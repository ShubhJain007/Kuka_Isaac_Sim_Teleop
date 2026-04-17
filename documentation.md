# Kuka Med 7 Development & Technical Documentation

This document provides a comprehensive, chronological record of the development of the Kuka Med 7 robot environment in Isaac Lab. It covers every file, code change, and function implemented.

---

## Phase 1: Foundation (Robot & Assets)

The first step was defining the physical properties and visual model of the Kuka Med 7.

### 1.1 `med7.urdf`
- **Description**: The Unified Robot Description Format file defining the links, joints, and visual/collision meshes of the robot.
- **Key Change**: Modified to ensure mesh paths are correctly resolved within the Isaac Lab project structure.

### 1.2 `source/Kuka_Med_7/Kuka_Med_7/assets.py`
- **Description**: Configures the robot as an Isaac Lab `Articulation`.
- **Functions/Configs**:
    - `MED7_CONFIG`: An `ArticulationCfg` that spawns the robot from the URDF. It configures the PD controller (stiffness/damping) and disables default gravity for the arm to ensure stability during teleoperation.
- **Key Milestone**: Achieving stable joint control without "shaking" by tuning damping to `80.0`.

---

## Phase 2: Environment Logic & Scene Definition

Once the robot was defined, we created the clinical environment and the simulation logic.

### 2.1 `source/Kuka_Med_7/Kuka_Med_7/tasks/med7/med7_env_cfg.py`
- **Description**: Defines the world layout using `DirectRLEnvCfg`.
- **Key Components**:
    - **Hospital Bed & Mount**: Custom cuboids with realistic materials (`BED_COLOR`, `MOUNT_COLOR`).
    - **Lighting**: Realistic cool-white dome lighting (6500K) and warm distant sunlight.
    - **Cameras**: `wrist_camera` on `lbr_link_7`; `room_camera` static in front of the arm (OR-style view) for MJPEG. VR planning also draws OpenXR **hand stick figures** under `/World/SurgicalPlan/HandStick/`.
    - **XR Configuration**: Sets up the `OpenXRDeviceCfg` for VR integration.

### 2.2 `source/Kuka_Med_7/Kuka_Med_7/tasks/med7/med7_env.py`
- **Description**: The core execution logic class `Med7Env` (inherits from `DirectRLEnv`).
- **Function Documentation**:
    | Function | Description |
    | :--- | :--- |
    | `__init__` | Initializes the environment with the provided configuration. |
    | `_setup_scene` | Spawns the robot, props (bed, mount), lights, and cameras. Clones environments for parallel simulation. |
    | `_pre_physics_step` | Internal buffer for joint position targets received from the policy/user. |
    | `_apply_action` | Sends the buffered joint targets to the robot articulation. |
    | `_get_observations` | Returns joint positions and velocities as a tensor. |
    | `_get_rewards` | Placeholder for RL rewards (currently returns zeros). |
    | `_get_dones` | Returns the "done" state based on episode length. |
    | `_reset_idx` | Resets specific environments to their default joint states. |

---

## Phase 3: Basic Simulation & Teleoperation

With the environment ready, we implemented standard scripts for running and manually controlling the robot.

### 3.1 `scripts/run_med7.py`
- **Description**: A basic script to launch the environment and verify that the robot can be stepped.
- **Functions**:
    - `main()`: Sets up the environment and runs a simple loop to render the scene.

### 3.2 `scripts/teleop_med7.py`
- **Description**: Enables Cartesian (End-Effector) control using a keyboard.
- **Functions**:
    - `main()`: Initializes the `Se3Keyboard` device and a `DifferentialIKController` to map keyboard inputs to joint commands.
- **Key Logic**: Uses Inverse Kinematics (IK) to compute the required joint angles to move the end-effector in X, Y, Z space.

---

## Phase 4: VR Integration & Web Streaming (Latest Updates)

The most recent phase focused on **Apple Vision Pro** teleoperation, moving from complex DDS infrastructure to a lightweight web-streaming approach.

### 4.1 `scripts/teleop_med7_vr.py`
- **Description**: The advanced teleoperation script for VR. It starts a Flask server to stream camera views and uses OpenXR for hand tracking.
- **Function Documentation**:
    | Function | Description |
    | :--- | :--- |
    | `generate_mjpeg(camera_key)` | Pulls RGB frames from the shared buffer and yields them as an MJPEG stream. |
    | `wrist_stream()` / `room_stream()` | Flask routes for accessing the camera streams via `/wrist` and `/room`. |
    | `home()` | Provides a web dashboard for the Vision Pro Safari browser. |
    | `run_server()` | Runs the Flask server in a dedicated background thread to prevent simulation lag. |
    | `start_teleop()` / `stop_teleop()` | Callbacks for the VR "START/STOP" hand gestures. |
    | `main()` | Orchestrates the entire system: Flask server, OpenXR polling, Coordinate Transforms, and Simulation stepping. |

### 4.2 Streaming & Control Flowcharts

#### Architecture Overview
```mermaid
graph TD
    A[Isaac Lab Simulation] -->|RGB Frames| B(Shared Frame Buffer)
    B --> C[Flask Web Server]
    C -->|MJPEG Stream| D[Apple Vision Pro Safari]
    E[Apple Vision Pro / CloudXR] -->|Hand Tracking Data| F[OpenXR Device]
    F -->|Delta Pose| G[Inverse Kinematics / Transform]
    G -->|Joint Commands| A
```

#### Detailed Control Loop
```mermaid
sequenceDiagram
    participant VR as Vision Pro (CloudXR)
    participant SIM as Isaac Lab (Simulation)
    participant FLASK as Flask Streaming Server

    loop 每一步 (Every Step)
        VR->>SIM: Send Hand Pose (OpenXR)
        SIM->>SIM: retarget & Transform Pose (Wrist -> World)
        SIM->>SIM: Compute Pose Error & Step Physics
        SIM->>FLASK: Update Frame Buffer (RGB)
        FLASK-->>VR: Stream MJPEG to Safari Browser
    end
```

---

## Phase 5: Troubleshooting & Debugging

If the VR teleoperation or hand tracking is not working as expected, follow this checklist:

### 1. Launch Requirements
Ensure you are running the script with the CloudXR experience file and correct environment variables.

#### 1.1 Start CloudXR Runtime (Docker)
The CloudXR Monado service must be running in a background container to handle the IPC socket connection:
```bash
docker run -it --rm --name cloudxr-runtime \
    --user $(id -u):$(id -g) --gpus=all -e "ACCEPT_EULA=Y" \
    --mount type=bind,src=$(pwd)/openxr,dst=/openxr \
    -p 48010:48010 -p 47998-48000:47998-48000/udp \
    -p 48005:48005/udp -p 48008:48008/udp -p 48012:48012/udp \
    nvcr.io/nvidia/cloudxr-runtime:5.0.1
```

#### 1.2 Setup Environment Variables
In the terminal where you launch Isaac Lab:
```bash
export EXTERNAL_RENDERER=cloudxr
export XDG_RUNTIME_DIR=$(pwd)/openxr/run
export XR_RUNTIME_JSON=$(pwd)/openxr/share/openxr/1/openxr_cloudxr.json
```

#### 1.3 Launch Script
```bash
python scripts/teleop_med7_vr.py --num_envs 1 --teleop_device handtracking --experience apps/isaaclab.python.headless.cloudxr.kit
```

### 2. Common Errors
- **FileNotFoundError (Asset Loading)**: If you see a 404/FileNotFound for `Isaac/Environments/Grid/default_environment.usd`, ensure your `apps/*.kit` file has the correct `persistent.isaac.asset_root.default` version (e.g., `4.5` for Isaac Sim 4.5).
- **Connection Refused (IPC)**: This means the Docker container in Step 1.1 is not running or the `XDG_RUNTIME_DIR` is not correctly exported.
- **Hand Tracking Settings**: Verify that "Hand Tracking" is enabled in the CloudXR Client settings on your Vision Pro.
- **Xcode Checklist**:
    - Ensure your Xcode project has the **Hand Tracking** capability enabled in "Signing & Capabilities".
    - Check that `NSHandsTrackingUsageDescription` is present in your `Info.plist`.
    - Verify that the app is using the **OpenXR Hand Tracking Extension** (check the sample app code if you modified it).
- **START Gesture**: You must perform a **PINCH** gesture (Index finger + Thumb) to activate the robot. Watch the terminal for `[INFO]: Teleop Started`.

### 3. Visual Feedback (Hand Proxy)
I have added a **Hand Visualization Proxy** (a green semi-transparent sphere) to the scene. 
- If you see the green sphere moving but the robot arm is stationary, the hand tracking is working, but the robot might be "Stopped." Check if you have performed the start gesture.
- If you don't see the green sphere at all, the OpenXR device is likely not receiving data from CloudXR.

---

## Phase 6: Asset Conversion & Medical Models

To integrate custom medical data (e.g., STL files of bones), we use the Isaac Lab Mesh Converter utilities. This process is more than just a format change; it optimizes the model for simulation.

### 6.1 `scripts/convert_stl_usd.py`
- **Description**: Converts raw mesh files (STL/OBJ) into native USD objects.
- **Key Technical Requirements**:
    - **AppLauncher & SimulationApp**: The conversion requires the Omniverse engine to be running. This loads the Carbonite (`carb`) binary stack and the USD Stage runtime.
    - **MeshConverterCfg**:
        - `asset_path`: Path to the raw STL.
        - `usd_dir`: Directory for the output USD.
        - `force_usd_conversion`: Ensures a fresh conversion every time.
        - `scale`: Essential for medical models (usually `0.001, 0.001, 0.001` to convert mm to meters).
        - `rigid_props`: Applying `RigidBodyPropertiesCfg()` ensures the USD is recognized by the physics engine (PhysX) as a simulatable object.

### 6.2 Bone Model Configuration
- **Kinematic Bone**: In the environment config (`med7_env_cfg.py`), the bone is defined as a `RigidObjectCfg` with `kinematic_enabled=True`. This prevents gravity from affecting it while still allowing collision detection.
- **Dynamic Updates**: By using `self.femur.write_root_pose_to_sim()`, we can "teleport" the bone to any tracked position in real-world time without breaking physics stability.

---

## Phase 7: ROS 2 Bone Tracking Integration

This phase connects the simulation to external tracking systems using the ROS 2 (Robot Operating System) bridge.

### 7.1 "Self-Healing" Library Logic
Isaac Sim 5.1 bundles its own ROS 2 (`rclpy`) and libraries. To avoid manual `LD_LIBRARY_PATH` exports, we implemented a `setup_ros2_libs()` function at the top of our scripts:
- **Action**: It detects the bundled path, sets the environment variable, and **re-executes the Python process** if the path was missing. 
- **Benefit**: Makes the project completely portable across different machines.

### 7.2 RSS Subscriber Logic
- **Topic**: `/bone_pose`
- **Message Type**: `geometry_msgs/msg/PoseStamped`
- **Node**: `BoneTrackerNode` (in `teleop_med7.py`)
- **Flow**: The node stores the latest pose in `latest_pos` and `latest_quat`, which are then applied to the environment via `env.update_femur_pose()` in the main simulation loop.

### 7.3 EKF-Based Bone Pose Smoothing

ROS `/tf` messages arrive at ~20 Hz from the tracking system, but Isaac Sim renders at 60-120 Hz. To fill the gaps and produce smooth, continuous bone motion we use an **Extended Kalman Filter (EKF)** per bone (femur, tibia).

#### State Model

| Component | State Vector | Model |
|---|---|---|
| **Position** | `[x, y, z, vx, vy, vz]` (6-state) | Constant-velocity with process noise |
| **Orientation** | Quaternion `[w,x,y,z]` + angular velocity `[ωx,ωy,ωz]` | Angular-velocity propagation + SLERP correction |

#### Why EKF over LERP/SLERP interpolation

| LERP/SLERP (previous) | EKF (current) |
|---|---|
| Can only interpolate *between* two known samples | **Predicts forward** beyond the latest sample using velocity |
| Holds at the last value when measurements stop, then jumps | Smooth coast-down via velocity decay |
| No noise filtering — passes sensor jitter directly | Kalman gain balances process model vs measurement noise |
| Bag restart causes stale buffer problems | EKF auto-reinitializes on large position jumps (>50 cm) |

#### Predict–Update Cycle

1. **Predict** (every render frame, ~120 Hz): propagate position via `x += v·dt`, quaternion via angular-velocity integration.
2. **Update** (on each `/tf` measurement, ~20 Hz): Kalman correction for position; SLERP-blend correction for orientation. Angular velocity estimated from measurement delta with EMA smoothing (α=0.4).

#### Tuning Parameters

| Parameter | Default | Purpose |
|---|---|---|
| `process_noise_pos` | 0.05 | How much position is allowed to drift per second² |
| `process_noise_vel` | 2.0 | How much velocity is allowed to change per second |
| `meas_noise_pos` | 0.002 | Trust in position measurements (lower = more trust) |
| `process_noise_quat` | 0.1 | Orientation prediction uncertainty |
| `meas_noise_quat` | 0.01 | Trust in orientation measurements |

---

## 🗺️ Full Data Flow Architecture

This flowchart displays how every component in the project interacts, from raw input to physical simulation.

```mermaid
graph TD
    subgraph External_Inputs["Real World / External"]
        K([Keyboard User]) -- "W,A,S,D,Q,E,I,J,K,L" --> Teleop
        Tracker([Tracker / Mock Script]) -- "geometry_msgs/PoseStamped" --> Teleop
    end

    subgraph Scripts["Control Scripts (Python)"]
        Teleop[teleop_med7.py] -- "Joint Position Commands" --> Env
        Mock[mock_bone_publisher.py] -- "/bone_pose Topic" --> Teleop
        LibFix[setup_ros2_libs] -- "Set LD_LIBRARY_PATH" --> Teleop
    end

    subgraph Logic["Environment Logic (Isaac Lab)"]
        Env[med7_env.py] -- "update_femur_pose(pos, quat)" --> FemurPr[Femur Prim]
        Env -- "DiffIK Controller" --> RoboPr[Robot Articulation]
        Cfg[med7_env_cfg.py] -- "Asset Blueprints" --> Env
    end

    subgraph Simulation_Engine["Isaac Sim (PhysX)"]
        RTX[RTX Rendering] --> View[Viewport Display]
        PhysX[Physics Step] -->|Collision / State| Fabric[Fabric Data Layer]
        Fabric -->|Observations| Env
    end

    RoboPr --> PhysX
    FemurPr --> PhysX
```

---

## ArUco Marker–Based Surgical Stylus Tracking

### Overview

The surgical pointer is now tracked via a physical **ArUco marker** (DICT_5X5_50, ID 10, 50 × 50 mm outer / 38 × 38 mm inner) attached to the stylus.  The Apple Vision Pro detects the marker in its camera feed and streams a full **6-DOF pose** (4 × 4 SE(3) matrix) through `avp_stream.get_markers(10)`.

### Why this is better than hand tracking for the probe

| Hand tracking (previous) | ArUco marker (current) |
|---|---|
| Index-finger tip ≈ probe tip — no rotation info | Full 6-DOF: position **and** approach vector |
| Requires the surgeon's hand to hold a pose | Rigid stylus gives stable, repeatable pose |
| Cannot distinguish orientation of the needle | Physical orientation of the marker IS the needle direction |
| Approach vector had to be published separately as a fixed Euler angle | Approach vector is directly the marker's −Z axis in world frame |

### USD Scene Hierarchy

```
/World/envs/env_0/SurgicalPlan/
└── StylusTracker  (UsdGeom.Xform)          ← receives 6-DOF from get_markers(10)
    ├── StylusTip   (UsdGeom.Sphere, r=7.2mm, green)  ← at local [0,0,−tip_offset]
    └── ApproachLine (UsdGeom.Cylinder, r=0.8mm, green) ← along local −Z, length=tip_offset
```

`StylusTracker` is set **invisible** by default and becomes visible only when the AVP detects the marker.

### Coordinate convention

- Marker's **local −Z** axis → physical needle direction (approach vector into bone).
- `StylusTip` is offset `STYLUS_TIP_OFFSET = 0.10 m` (10 cm) along local −Z from the marker centre (adjust to match actual stylus geometry).
- Children (`StylusTip`, `ApproachLine`) follow the parent `StylusTracker` automatically — no separate transform math needed.

### Dwell-based lock (unchanged)

Holding the stylus still (< 5 mm displacement) for **2 seconds** locks both the position **and orientation** (full 6-DOF).  The locked pose is published on ROS.  Returning to the locked point and holding still for another 2 s unlocks.

### ROS output topics (when locked)

| Topic | Type | Contents |
|---|---|---|
| `/surgical_plan/probe_pose` | `geometry_msgs/PoseStamped` | Probe tip position + marker orientation as quaternion |
| `/surgical_plan/guide_line` | `geometry_msgs/PoseArray` | [tip + approach_depth offset, tip] — approach line segment |

When the ArUco marker is visible, the quaternion is derived directly from the marker's rotation matrix via `scipy.spatial.transform.Rotation.from_matrix()`, giving a physically accurate approach vector.  When the marker is not detected, the system falls back to the fixed Euler angles.

### Key parameters (`teleop_med7_vr.py`)

```python
ARUCO_MARKER_ID   = 10     # DICT_5X5_50 ID printed on the stylus
STYLUS_TIP_OFFSET = 0.10   # metres: marker centre → physical needle tip along −Z
```

---

## Summary of Code Evolution

1.  **Direct Joint Control**: Started with simple joint-space movements.
2.  **Differential IK**: Added Cartesian control for intuitive keyboard interaction and VR pose retargeting.
3.  **DDS Integration (Removed)**: Initially attempted RTI DDS for streaming, but replaced it due to complexity.
4.  **Flask + OpenXR**: High-performance streaming and VR control solution.
5.  **Asset Pipeline**: Established a workflow for converting and tracking custom medical meshes (STL to USD).
6.  **ROS 2 Bridge**: Completed real-time tracking for medical bones with self-healing dependency management.
7.  **ArUco Stylus Tracker**: Replaced hand-tracking probe with ArUco marker 6-DOF tracking for true approach-vector control.