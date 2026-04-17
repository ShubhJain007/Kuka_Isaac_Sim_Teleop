# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Kuka Med 7 LBR teleoperation in NVIDIA **Isaac Lab 5.1 / Isaac Sim 5.1** (Python 3.11). Supports keyboard and Apple Vision Pro (CloudXR) VR control, Flask MJPEG streaming, ROS 2 bone/point-cloud tracking, and ArUco-marker stylus tracking. Targets a medical/surgical planning use case (femur + tibia tracking, surgical guide line, bone opacity controls).

## Install

Editable install is required before any script will import `Kuka_Med_7`:

```bash
python -m pip install -e source/Kuka_Med_7
```

`setup.py` reads metadata from `source/Kuka_Med_7/config/extension.toml`.

## Run

All scripts must be launched with the **Isaac Lab / Isaac Sim Python** (3.11), not system Python.

- **Keyboard teleop** (Cartesian IK): `python scripts/teleop_med7.py` — add `--no-ros` if `rclpy` / bridge setup breaks startup.
- **VR teleop** (surgical planning, robot joints are *held*; probe/guide/opacity only):
  ```bash
  source scripts/cloudxr_env.sh   # auto-detects OPENXR_ROOT from the cloudxr-runtime container
  python scripts/teleop_med7_vr.py --experience "$(pwd)/apps/isaaclab.python.headless.cloudxr.kit"
  ```
  Flags: `--ros` (subscribe to bone topics), `--keyboard-fallback` (debug keys), `--sensitivity`.
- **VR + ROS wrapper** (handles env var cleanup so Isaac Sim's bundled ROS 2 is used, not `/opt/ros/`): `./scripts/run_teleop_ros.sh`.
- **CloudXR health check**: `bash scripts/check_cloudxr.sh` (looks for `ipc_cloudxr` socket).
- MJPEG dashboard served by `teleop_med7_vr.py` at `http://<host>:5000` (routes `/wrist`, `/room`).

The CloudXR `.kit` experience is **this repo's** file at `apps/isaaclab.python.headless.cloudxr.kit` — not the stock Isaac Lab one. Hand-tracking toggles differ from the Isaac Lab stock `isaaclab.python.xr.openxr.headless.kit`.

## Lint

`pyproject.toml` configures `ruff` (line length 120, py310 target, isort with custom Isaac Lab section order) and `pyright` (basic, py311, Linux). There is no `pytest` suite configured beyond the `isaacsim_ci` marker.

## Architecture

### Package layout

- `source/Kuka_Med_7/Kuka_Med_7/` — installable package.
  - `assets.py` — `MED7_CONFIG` `ArticulationCfg` built from `med7.urdf`; PD tuning (damping 80.0), arm gravity disabled for teleop stability.
  - `tasks/med7/med7_env_cfg.py` — `DirectRLEnvCfg`: scene (hospital bed, mount, 6500 K dome + warm distant sun), `wrist_camera` on `lbr_link_7`, static OR-view `room_camera`, `OpenXRDeviceCfg`, hand-stick prims under `/World/SurgicalPlan/HandStick/`.
  - `tasks/med7/med7_env.py` — `Med7Env(DirectRLEnv)`: `_apply_action` writes buffered joint targets; `update_femur_pose` / `update_pointcloud` / `sync_robot_joint_state` are the side channels for external trackers. Gym id **`Isaac-Med7-v0`** registered in `tasks/med7/__init__.py`.
- `scripts/` — entry points and helpers (see Run). `med7_ros_bones.py` and `med7_ros_pointcloud.py` define ROS 2 nodes; `mock_bone_publisher.py` is a standalone test publisher. `xr_pov_helpers.py` has VR POV camera math. `convert_stl_usd.py` wraps `MeshConverterCfg` (medical STL mm→m needs `scale=(0.001,0.001,0.001)`).
- `apps/isaaclab.python.headless.cloudxr.kit` — **project-local** kit file; ensure `persistent.isaac.asset_root.default` matches the installed Isaac Sim version or asset loading 404s.
- `med7/`, `Mount_with_EE.stl`, `*_cut_usd/` — meshes + converted USDs. `ir_tracking/` holds NDI Polaris tracker `.rom` files.

### Control-flow contract

1. External input (keyboard `Se3Keyboard`, OpenXR hand pose, or ArUco 6-DOF from `avp_stream.get_markers(10)`) produces a **delta pose**.
2. `DifferentialIKController` retargets that to joint targets.
3. Targets are buffered in `_pre_physics_step` and written in `_apply_action`.
4. Bone poses from ROS `/bone_pose_femur`, `/bone_pose_tibia` (`geometry_msgs/PoseStamped`) or `/tracked/femur` (`PointCloud2`) update **kinematic** rigid objects via `write_root_pose_to_sim` — they move without physics coupling.

The VR script (`teleop_med7_vr.py`) is planning-only: it updates the **surgical pointer / guide line / bone opacity** but does *not* drive arm IK. Use `teleop_med7.py` for keyboard arm teleop.

### ROS 2 "self-healing" library setup

`scripts/med7_ros_bones.py::setup_ros2_libs()` detects Isaac Sim's bundled `rclpy` (`isaacsim.ros2.bridge/humble`), injects `LD_LIBRARY_PATH`, **re-execs the Python process** if the path was missing. Required because system ROS Humble is Python 3.10 and Isaac Sim is 3.11 — mixing them produces `ModuleNotFoundError: rclpy._rclpy_pybind11`. `run_teleop_ros.sh` additionally unsets `ROS_DISTRO` / `AMENT_PREFIX_PATH` / `ROS_PYTHON_VERSION` but **preserves** `ROS_DOMAIN_ID`, `RMW_IMPLEMENTATION`, and system `LD_LIBRARY_PATH` (needed for DDS middleware discovery).

### Bone-pose smoothing (EKF)

Per-bone Extended Kalman Filter in `med7_ros_bones.py` fills the gap between ~20 Hz `/tf` measurements and 60–120 Hz render: 6-state constant-velocity for position, quaternion + angular-velocity propagation for orientation with SLERP correction. Re-initializes on >50 cm jumps. Key tuning knobs: `process_noise_pos` (0.05), `meas_noise_pos` (0.002), `process_noise_quat` (0.1), `meas_noise_quat` (0.01).

### ArUco stylus (6-DOF)

`ARUCO_MARKER_ID = 10` (DICT_5X5_50), `STYLUS_TIP_OFFSET = 0.10` m along local −Z. USD hierarchy at `/World/envs/env_0/SurgicalPlan/StylusTracker/{StylusTip, ApproachLine}` — children inherit the parent Xform, so only the tracker Xform is updated. Dwell-lock: stylus still (<5 mm) for 2 s → publish `/surgical_plan/probe_pose` + `/surgical_plan/guide_line`.

### Key caveats

- ROS point-cloud path replaces the old `/World/Bones/*` USD meshes; bone-lock keys (`K`, `1`, `2`) are removed.
- VR joint sync from `/lbr/joint_states` is **disabled by default** to avoid fighting the VR command stream — see the commented block around `teleop_med7_vr.py:515`.
- `cloudxr_env.sh` **auto-detects** `OPENXR_ROOT` from the running `cloudxr-runtime` Docker container; override with `CLOUDXR_OPENXR_ROOT` if the container name differs. The host path must match the container's `--mount src=...:dst=/openxr`.

## References in-repo

- `README.md` — usage walkthrough.
- `documentation.md` — chronological dev log, architecture diagrams, EKF and ArUco details.
- `scripts/ROS_INTEGRATION.md` — point-cloud + joint-state ROS wiring.
- `scripts/VR_PLANNING_WALKTHROUGH.md`, `scripts/AVP_CAMERA_STREAM_GUIDE.md` — VR-side user flow.
