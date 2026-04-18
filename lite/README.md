# Kuka Med7 Surgical Visualization — Lite

Pure renderer. No physics engine, no RL environment, no hand tracking.

## What it does

1. Loads the Kuka Med7 robot and sets joint angles from ROS `/lbr/joint_states` (forward kinematics, no PhysX)
2. Loads femur/tibia bone meshes and sets their transforms from ROS `/tf`
3. Muse stylus drives a surgical probe with lock/unlock via button hold
4. Streams the scene to Apple Vision Pro via avp_stream

## Old vs new

| | Old (`scripts/` + `source/`) | New (`lite/`) |
|---|---|---|
| Lines of code | ~3,700 | ~860 |
| Physics engine | PhysX at 120Hz | None — pure FK |
| RL environment | DirectRLEnv + gym | None |
| Cameras | 2x 224x224 every frame | None |
| Hand tracking | Computed then discarded | Removed |
| ICP/EKF bone tracking | Redundant re-filtering | Removed |
| Robot control | Frozen or mirroring ROS | FK from joint angles |
| Files | 10+ across 3 dirs | 5 in one dir |

## Files

```
lite/
├── run.py          # Main loop (195 lines)
├── scene.py        # USD scene + forward kinematics (260 lines)
├── ros_bridge.py   # /tf + /joint_states subscriber (138 lines)
├── stylus.py       # Muse stylus EKF + buttons (146 lines)
└── probe.py        # Lock/unlock/bone-attach (122 lines)
```

## Running

```bash
# Scene viewer only (no ROS, no AVP):
python lite/run.py

# With ROS (bone tracking + robot mirroring):
python lite/run.py --ros

# With Apple Vision Pro AR:
python lite/run.py --avp 192.168.1.42

# Full:
python lite/run.py --ros --avp 192.168.1.42 --avp-z 0.825
```

## ROS topics

Subscribed:
- `/tf` — bone poses (`tracked/femur_origin`, `tracked/tibia_origin`)
- `/lbr/joint_states` — robot joint positions

Published (when probe is locked):
- `/surgical_plan/probe_pose` (PoseStamped, ~10 Hz)
- `/surgical_plan/guide_line` (PoseArray, ~10 Hz)

## Prerequisites

- Isaac Sim 4.5+ (for URDF importer + USD rendering)
- `scipy` (`pip install scipy`)
- `avp_stream` (`pip install avp_stream`) — only for `--avp`
- `ir_tracking/scripts/pose_ekf.py` — only for stylus EKF

## Asset paths

Resolved relative to project root (no hardcoded absolute paths):
- `med7.urdf`
- `femur_cut_usd/Draw_Left_Femur_Plan_Array_V2.usd`
- `tibia_cut_usd/tibia_cut.usd`
