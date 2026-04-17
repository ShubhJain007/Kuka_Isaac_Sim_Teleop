# Isaac Lab — Unofficial Basics Guide
> **Things the official docs gloss over, gotchas that will bite you, and the mental model you actually need.**  
> All examples are drawn from this project (`Kuka_Med_7`) so you can cross-reference real code.

---

## Table of Contents
1. [The Mental Model — What Isaac Lab Actually Is](#1-the-mental-model)
2. [USD & The Prim Hierarchy — The Foundation of Everything](#2-usd--the-prim-hierarchy)
3. [Project File Structure — What Goes Where and Why](#3-project-file-structure)
4. [The `@configclass` Pattern — How Every Setting Is Configured](#4-the-configclass-pattern)
5. [Environment Types — `DirectRLEnv` vs `ManagerBasedEnv`](#5-environment-types)
6. [The `_setup_scene()` Method — Where the World Is Built](#6-the-_setup_scene-method)
7. [The Simulation Loop — What Happens Every Step](#7-the-simulation-loop)
8. [Assets — Robots, Rigid Objects, and Static Props](#8-assets)
9. [Sensors — Cameras and More](#9-sensors)
10. [The Gym Interface — How to Launch and Control an Env](#10-the-gym-interface)
11. [Coordinate Systems & Quaternion Conventions](#11-coordinate-systems--quaternion-conventions)
12. [The USD Stage — Directly Manipulating Prims at Runtime](#12-the-usd-stage-directly-manipulating-prims)
13. [Physics Gotchas — Things That Will Drive You Crazy](#13-physics-gotchas)
14. [Integrating ROS 2 — The Bridge Pattern](#14-integrating-ros-2)
15. [First Project Walkthrough — Building a Robot Env from Scratch](#15-first-project-walkthrough)

---

## 1. The Mental Model

Isaac Lab is a **layer on top of Isaac Sim** (which itself sits on top of Omniverse/USD/PhysX).

```
Your Script (e.g., teleop_med7.py)
        ↓
  Isaac Lab (isaaclab)          ←  High-level: Env, Assets, Sensors, Controllers
        ↓
  Isaac Sim (isaacsim)          ←  Mid-level: App lifecycle, ROS bridge, XR
        ↓
  Omniverse Kit (omni.*)        ←  Low-level: USD stage, rendering pipeline
        ↓
  PhysX (pxr, usdrt)            ←  Physics simulation
```

**Key insight**: Isaac Lab abstracts away most of Omniverse, but when you need to do something the abstraction doesn't support (e.g., move a mesh without physics, or read raw prim attributes), you drop down to the `omni.*` and `pxr.*` APIs directly. Don't be afraid to do this.

---

## 2. USD & The Prim Hierarchy

**USD (Universal Scene Description)** is the file format and in-memory scene graph that drives everything. Every object in the simulation — robots, lights, cameras, the ground — is a **prim** (primitive) in a tree called the **stage**.

### The Stage Path Tree
```
/World
├── envs/
│   ├── env_0/
│   │   ├── Robot/
│   │   │   ├── lbr_link_0
│   │   │   └── lbr_link_7/
│   │   │       └── wrist_cam
│   │   ├── RobotMount
│   │   ├── HospitalBed
│   │   └── HandProxy
│   └── env_1/
│       └── ...  (cloned from env_0)
├── Bones/               ← Note: OUTSIDE /envs/ — PhysX ignores these
│   ├── femur
│   └── tibia
├── Light
├── DistantLight
└── ground_plane
```

### Why `/World/envs/env_.*/` matters
Isaac Lab uses the regex `env_.*` to identify and **clone** environments. Any prim placed at a path matching `/World/envs/env_.*/...` will be duplicated for each parallel simulation. If you put something **outside** `/World/envs/`, there is exactly **one** copy, shared by all environments — this is how the `Light`, `ground_plane`, and (in this project) the bone meshes work.

### Prim Types You'll Encounter
| Prim Type | What It Is |
|-----------|-----------|
| `Xform` | Empty transform node — a parent/group |
| `Scope` | Namespace group, no transform (use for logical groupings) |
| `Mesh` | 3D geometry |
| `PhysicsScene` | Defines the PhysX simulation domain |
| `RigidBody` | A prim controlled by the physics engine |
| `Articulation` | A kinematic chain (robot) |
| `Camera` | Render camera |

---

## 3. Project File Structure

A well-structured Isaac Lab project follows this layout (this project is a good reference):

```
MyProject/
├── pyproject.toml            # Ruff/Pyright config; also defines the Python package
├── med7.urdf                 # Robot description file
├── femur_cut_usd/            # USD asset files (.usd, .usda)
│   └── Draw_Left_Femur_Plan_Array_V2.usd
│
├── scripts/                  # Runnable entry points (not importable Python packages)
│   ├── teleop_med7.py        # Main executable — launches AppLauncher here
│   └── train.py              # RL training entry point
│
├── ir_tracking/              # Domain-specific utilities (IR camera, ROS publishers)
│   └── scripts/
│       └── vega_ros2_publisher.py
│
└── source/
    └── MyProject/            # The Python package registered with Isaac Lab
        └── MyProject/
            ├── __init__.py           # Package init — registers gym envs here
            ├── assets.py             # ArticulationCfg definitions
            ├── tasks/
            │   ├── __init__.py
            │   ├── direct/           # DirectRLEnv tasks
            │   │   └── my_task/
            │   │       ├── __init__.py
            │   │       ├── my_env.py      # Env logic
            │   │       └── my_env_cfg.py  # Env config
            │   └── manager_based/    # ManagerBasedEnv tasks (optional)
            └── ui_extension_example.py
```

### The `__init__.py` at the package root
This file is where you **register** your environment with Gymnasium so `gym.make()` can find it:

```python
# source/MyProject/MyProject/__init__.py
import gymnasium as gym

gym.register(
    id="Isaac-Med7-v0",
    entry_point="Kuka_Med_7.tasks.med7.med7_env:Med7Env",
    kwargs={"cfg": Med7EnvCfg()},
)
```

> **Gotcha**: The `id` string you pass to `gym.register` is what you use in `gym.make("Isaac-Med7-v0")`. They must match exactly. If you add a second variant (e.g., for RL training vs teleoperation), just register a second ID.

---

## 4. The `@configclass` Pattern

Almost everything in Isaac Lab is configured with **dataclass-style config objects** decorated with `@configclass`. This is Isaac Lab's way of making configs type-safe, composable, and overridable.

```python
from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.scene import InteractiveSceneCfg

@configclass
class MyEnvCfg(DirectRLEnvCfg):
    # Env parameters
    decimation: int = 2              # Step sim this many times per RL step
    episode_length_s: float = 10.0   # Episode length in seconds
    action_space: int = 7
    observation_space: int = 14
    state_space: int = 0             # For asymmetric actor-critic; 0 = disabled

    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=decimation)

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=5.0,       # Distance between parallel environments (meters)
        replicate_physics=True # Reuse physics data across envs for speed
    )
```

### Overriding configs at launch time
You never edit the config class — you **instantiate it and mutate it** in your script:

```python
env_cfg = Med7EnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs  # Override before creating the env
env = gym.make("Isaac-Med7-v0", cfg=env_cfg)
```

### The `MISSING` sentinel
Some config fields use `MISSING` as a default, which means they **must** be set before the env is created. If you forget, you get a runtime error.

```python
from dataclasses import MISSING
prim_path: str = MISSING  # Must be set before use
```

---

## 5. Environment Types

Isaac Lab has two main environment APIs:

### `DirectRLEnv` — Use this for teleop and custom projects
You manually implement the physics, observations, rewards. It's explicit and gives you full control.

```
DirectRLEnv
├── _setup_scene()       — called once at startup
├── _pre_physics_step()  — called every RL step, before physics
├── _apply_action()      — apply joint targets etc.
├── _get_observations()  — return obs dict
├── _get_rewards()       — return reward tensor
├── _get_dones()         — return (terminated, truncated) tensors
└── _reset_idx()         — reset specific envs by index
```

### `ManagerBasedRLEnv` — Use this for standard RL training
Uses a manager architecture (ObservationManager, RewardManager, etc.). Each component is a separate config + class. Much more modular, good for training tasks, but more boilerplate.

**Verdict**: For teleoperation, custom robotics demos, or anything that isn't pure RL training → use `DirectRLEnv`. It's simpler and you have more control.

---

## 6. The `_setup_scene()` Method

This is the **most important method** to understand. It's called once when the environment starts and is where you build your world.

### Rule 1: Register assets with `self.scene`
After creating assets, you must register them:

```python
def _setup_scene(self):
    self.robot = Articulation(self.cfg.robot)
    self.hand_proxy = RigidObject(self.cfg.hand_proxy)

    # MUST register so Isaac Lab tracks them
    self.scene.articulations["robot"] = self.robot
    self.scene.rigid_objects["hand_proxy"] = self.hand_proxy

    # Clone environments (creates env_1, env_2, ... copies)
    self.scene.clone_environments(copy_from_source=False)
    self.scene.filter_collisions(global_prim_paths=[])
```

### Rule 2: `clone_environments()` must come AFTER all env-level asset creation
Any prim you create under `/World/envs/env_0/...` will be cloned. Prims created after the clone call won't be automatically replicated.

### Rule 3: Static props don't need to be registered
Props like a `CuboidCfg` ground plane or a mesh that has no runtime interaction don't need to go into `self.scene.rigid_objects`. You just spawn them:

```python
self.cfg.hospital_bed.spawn.func(
    self.cfg.hospital_bed.prim_path,
    self.cfg.hospital_bed.spawn,
    translation=self.cfg.hospital_bed.init_state.pos,
    orientation=self.cfg.hospital_bed.init_state.rot,
)
```

### Rule 4: To opt out of physics entirely, go outside `/World/envs/`
If you want to move a mesh at runtime via pure USD transforms (no PhysX), spawn it **outside** the env tree. PhysX only simulates prims under its registered scene scope.

```python
# Spawned outside /World/envs/ — PhysX never touches this
UsdGeom.Scope.Define(stage, "/World/Bones")
self.cfg.femur.spawn.func("/World/Bones/femur", self.cfg.femur.spawn, ...)
```

---

## 7. The Simulation Loop

Every call to `env.step(action)` triggers this sequence:

```
env.step(action)
    ↓
_pre_physics_step(action)    ← store/preprocess the action
    ↓
    for _ in range(decimation):
        _apply_action()      ← write joint targets to the articulation
        scene.step()         ← advance PhysX by dt
    ↓
_get_observations()          ← read robot.data, sensor.data, etc.
_get_rewards()
_get_dones()
    ↓
(obs, reward, terminated, truncated, info)  ← returned to your script
```

**`decimation`**: The sim runs at `1/dt` Hz (e.g., 120 Hz), but your policy loop runs at `1/(dt * decimation)` Hz. With `dt=1/120` and `decimation=2`, the policy runs at 60 Hz.

---

## 8. Assets

### Articulations (Robots)
Defined via `ArticulationCfg`. The key parameters:

```python
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils

MED7_CONFIG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path="/path/to/robot.urdf",
        fix_base=True,          # Fixed-base vs floating-base robot
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,   # Joints don't sag under gravity
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos={"joint_name": 0.0, ...}  # Initial joint angles (radians)
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["lbr_A.*"],  # Regex to match joint names
            stiffness=400.0,
            damping=80.0,
        ),
    },
)
```

**`ImplicitActuatorCfg`** uses PhysX's built-in PD controller. The sim computes torques internally. This is the simplest and most stable actuator model.

### Reading Robot Data
After `env.reset()` or `env.step()`, the robot's latest state is available:

```python
# All returns are (num_envs, ...) tensors on GPU
joint_pos  = env.robot.data.joint_pos          # (N, num_joints)
joint_vel  = env.robot.data.joint_vel
ee_pose_w  = env.robot.data.body_pose_w[:, body_idx]  # (N, 7) [pos + quat]
root_pose  = env.robot.data.root_pose_w        # (N, 7)
jacobian   = env.robot.root_physx_view.get_jacobians()  # For IK
```

### Rigid Objects
For objects that have physics (can be pushed, fall under gravity):

```python
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils

hand_proxy = RigidObjectCfg(
    prim_path="/World/envs/env_.*/HandProxy",
    spawn=sim_utils.SphereCfg(
        radius=0.05,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,    # Moved by you, not physics
            disable_gravity=True,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0.8)),
)
```

**`kinematic_enabled=True`**: The object has a physics body but it's driven by you (not simulated). Perfect for hand trackers, markers, objects you want to set poses on directly.

### Static Visual Props
For things that only need visuals (no physics interaction):

```python
from isaaclab.assets import AssetBaseCfg

hospital_bed = AssetBaseCfg(
    prim_path="/World/envs/env_.*/HospitalBed",
    spawn=sim_utils.CuboidCfg(
        size=(2.0, 1.0, 0.55),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.85, 0.88, 0.90),
            metallic=0.1,
            roughness=0.7,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),  # Optional: add collision
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(1.3, 0.0, 0.275)),
)
```

### Loading a USD File
```python
sim_utils.UsdFileCfg(
    usd_path="/absolute/path/to/asset.usd",
    scale=(1.0, 1.0, 1.0),   # Scale factor (1.0 = no change)
)
```

> **Gotcha**: USD paths must be **absolute**. Relative paths will silently fail or cause confusing errors.

### Loading a URDF
```python
sim_utils.UrdfFileCfg(
    asset_path="/absolute/path/to/robot.urdf",
    fix_base=True,
)
```
The URDF is **converted to USD** the first time it is loaded and cached. If you change the URDF, delete the cached USD (usually next to the URDF file or in `/tmp/`) to force re-conversion.

---

## 9. Sensors

### Camera
```python
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils

wrist_camera = CameraCfg(
    prim_path="/World/envs/env_.*/Robot/lbr_link_7/wrist_cam",
    update_period=0.0,     # 0.0 = update every sim step
    height=224,
    width=224,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=12.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 1.0e5),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.0, 0.05, 0.03),       # Offset from parent link frame
        rot=(0.5, -0.5, 0.5, -0.5), # Quaternion (w, x, y, z) in ROS convention
        convention="ros",
    ),
)
```

**Reading camera data**:
```python
# After env.step(), in _get_observations():
rgb = self.wrist_camera.data.output["rgb"]       # (N, H, W, 4) RGBA
depth = self.wrist_camera.data.output["distance_to_image_plane"]  # (N, H, W)
```

**Camera conventions**:
- `convention="ros"`: X right, Y down, Z forward (standard ROS/OpenCV).
- `convention="world"`: X right, Y up, Z backward (OpenGL, default if unset).
- When attaching to a robot link, almost always use `"ros"`.

---

## 10. The Gym Interface

### Launching the App
**Critical**: `AppLauncher` must be created before any `isaaclab` imports (other than `AppLauncher` itself). This is because Isaac Sim hooks into Python at import time.

```python
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)    # Adds --headless, --device, etc.
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ↑ Everything above MUST come before any isaaclab.* or isaacsim.* imports ↑
import gymnasium as gym
import torch
# ... rest of imports
```

### Creating and Running the Environment
```python
env_cfg = Med7EnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs

env = gym.make("Isaac-Med7-v0", cfg=env_cfg)
env = env.unwrapped   # Remove gym wrappers to access env attributes directly

obs, _ = env.reset()

while simulation_app.is_running():
    with torch.inference_mode():
        action = compute_action(obs)   # Your logic here
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated.any():
            obs, _ = env.reset()

env.close()
simulation_app.close()
```

### Common CLI arguments from `AppLauncher`
| Flag | Effect |
|------|--------|
| `--headless` | Run without a display (faster, for training) |
| `--device cuda:0` | Which GPU to use |
| `--livestream 1` | Stream the viewport to a browser |
| `--num_envs N` | Number of parallel environments |

---

## 11. Coordinate Systems & Quaternion Conventions

This is the #1 source of bugs in robot simulation. Memorize these.

### Isaac Lab World Frame
- **X** = forward
- **Y** = left
- **Z** = up (Z-up, same as USD/Omniverse, different from ROS!)

### ROS Frame
- **X** = forward
- **Y** = left
- **Z** = up ... wait, this is the same. The difference is in **camera** convention.

### Quaternion Convention
Isaac Lab uses **(w, x, y, z)** format everywhere.  
ROS uses **(x, y, z, w)** format.

**Converting from ROS `PoseStamped` to Isaac Lab**:
```python
def ros_pose_to_isaaclab(msg):
    pos = torch.tensor([
        msg.pose.position.x,
        msg.pose.position.y,
        msg.pose.position.z,
    ], device="cuda:0").unsqueeze(0)
    
    quat = torch.tensor([
        msg.pose.orientation.w,   # ← w comes FIRST in Isaac Lab
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
    ], device="cuda:0").unsqueeze(0)
    
    return pos, quat
```

### Body Frame vs World Frame
- `body_pose_w` = pose in **world** frame
- `joint_pos` = joint angles in **joint** (1D) space
- `root_pose_w` = root link pose in world frame
- The `subtract_frame_transforms()` utility converts EE pose from world frame to robot root frame (needed for IK).

---

## 12. The USD Stage — Directly Manipulating Prims

When you need to move something that Isaac Lab doesn't manage (e.g., a visual-only bone mesh), you use the raw USD APIs.

### Getting the Stage
```python
import omni.usd
stage = omni.usd.get_context().get_stage()
```

### Moving a Prim by Setting Its Transform Matrix
```python
from pxr import Gf, UsdGeom

prim = stage.GetPrimAtPath("/World/Bones/femur")
if not prim.IsValid():
    print("Prim not found!")

# Build rotation from quaternion (w, x, y, z)
rotation = Gf.Rotation(
    Gf.Quaternion(float(quat_w), Gf.Vec3d(float(qx), float(qy), float(qz)))
)
translation = Gf.Vec3d(float(x), float(y), float(z))

matrix = Gf.Matrix4d()
matrix.SetRotateOnly(rotation)
matrix.SetTranslateOnly(translation)

UsdGeom.Xformable(prim).MakeMatrixXform().Set(matrix)
```

### Creating a Scope (Namespace Group)
```python
from pxr import UsdGeom
if not stage.GetPrimAtPath("/World/MyGroup").IsValid():
    UsdGeom.Scope.Define(stage, "/World/MyGroup")
```

### Why Use This Instead of RigidObject API?
- **No physics jitter**: Prims outside of PhysX scope are never touched by the physics engine.
- **Direct control**: You set exactly what you want, no velocity/inertia to fight.
- **Simpler for visualization**: If you only need to show where something is (e.g., IR-tracked bones), you don't want physics at all.

---

## 13. Physics Gotchas

### 1. Objects vibrate / jitter even when "stationary"
**Cause**: The physics engine is applying residual forces (gravity, contact resolution).  
**Fix options**:
- Set `disable_gravity=True` in `RigidBodyPropertiesCfg`.
- Set `kinematic_enabled=True` to make the object driven-by-you, not simulated.
- Disable collision with `CollisionPropertiesCfg(collision_enabled=False)`.
- Move the object **outside** `/World/envs/` and use raw USD transforms.

### 2. Prim appears but physics doesn't apply
**Cause**: The prim is missing a `RigidBody` schema or is outside the PhysX scene scope.  
**Fix**: Ensure the prim was created via an Isaac Lab asset config (`RigidObjectCfg`, `ArticulationCfg`), **or** manually apply physics schemas if going raw USD.

### 3. URDF loads with wrong joint limits
The URDF importer sometimes ignores or mis-applies effort/velocity limits. Always verify joint behavior after loading. Setting `disable_gravity=True` on the entire robot is often required to stop the robot from collapsing under its own weight before the controller engages.

### 4. `scene.clone_environments()` must come before accessing physics views
Don't call `env.robot.root_physx_view.get_jacobians()` in `_setup_scene()`. Physics views are only valid after the first `env.reset()` / `env.step()`.

### 5. GPU tensor device mismatch
All tensors fed to Isaac Lab must be on the same CUDA device the simulation is using. Pass `device=env.device` when creating tensors if you don't know which GPU was selected.

```python
action = torch.zeros(env.num_envs, 7, device=env.device)
```

### 6. `replicate_physics=True` gotcha
With this flag on, all environments share the same physics data (faster). But it means **you cannot set different states per environment**. For teleoperation with a single robot, this is fine. For RL training with episodic resets, be careful.

---

## 14. Integrating ROS 2

Isaac Sim ships a bundled ROS 2 Humble bridge. The trick is that its libraries are **not on Python's default path**, so you must set `LD_LIBRARY_PATH` before importing `rclpy`.

### The Self-Patching Pattern (from `teleop_med7.py`)
```python
import sys, os

def setup_ros2_libs():
    import isaacsim
    isaacsim_path = os.path.dirname(isaacsim.__file__)
    ros2_bridge_root = os.path.join(isaacsim_path, "exts/isaacsim.ros2.bridge/humble")
    lib_path = os.path.join(ros2_bridge_root, "lib")
    python_path = os.path.join(ros2_bridge_root, "rclpy")

    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_path not in current_ld:
        # Re-exec the process with the correct env
        os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{current_ld}"
        os.execv(sys.executable, [sys.executable] + sys.argv)

    if python_path not in sys.path:
        sys.path.append(python_path)

setup_ros2_libs()   # Must run BEFORE AppLauncher
```

The `os.execv` call restarts the Python process with the updated environment. This is the cleanest way to avoid needing to manually source ROS before running the script.

### The ROS-in-the-Loop Pattern
Isaac Lab's simulation loop is synchronous. To handle ROS callbacks:

```python
import rclpy

rclpy.init()
ros_node = MyRosNode()

while simulation_app.is_running():
    # Non-blocking spin — process any waiting callbacks, then continue
    rclpy.spin_once(ros_node, timeout_sec=0.001)

    with torch.inference_mode():
        # ... your sim step logic
        obs, _, _, _, _ = env.step(action)

env.close()
ros_node.destroy_node()
rclpy.shutdown()
```

**`timeout_sec=0.001`**: Waits up to 1ms for ROS messages. Small enough that it doesn't block the sim, large enough to actually process incoming data.

### ROS Node Pattern for Subscriptions
```python
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class MyTrackerNode(Node):
    def __init__(self):
        super().__init__("my_tracker_node")
        self.latest_pos = None
        self.latest_quat = None
        self.has_new_data = False

        self.create_subscription(PoseStamped, "/my_topic", self._callback, 10)

    def _callback(self, msg):
        self.latest_pos = torch.tensor([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], device="cuda:0").unsqueeze(0)
        self.latest_quat = torch.tensor([
            msg.pose.orientation.w,  # Isaac Lab order: w, x, y, z
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        ], device="cuda:0").unsqueeze(0)
        self.has_new_data = True

# In main loop:
rclpy.spin_once(ros_node, timeout_sec=0.001)
if ros_node.has_new_data:
    env.update_object_pose(ros_node.latest_pos, ros_node.latest_quat)
    ros_node.has_new_data = False   # Consume the flag
```

> **Only update on new data.** If you update every step even when no new ROS message arrived, you create physics drift because you're writing stale data each tick.

---

## 15. First Project Walkthrough

Here's a complete recipe for adding a new robot simulation to this project structure.

### Step 1: Define the Robot Asset Config (`assets.py`)
```python
# source/MyProject/MyProject/assets.py
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

MY_ROBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path="/path/to/my_robot.urdf",
        fix_base=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos={"joint_1": 0.0, "joint_2": 0.1},  # joint name → radians
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],   # match all joints
            stiffness=200.0,
            damping=40.0,
        ),
    },
)
```

### Step 2: Define the Env Config (`my_env_cfg.py`)
```python
# source/MyProject/MyProject/tasks/direct/my_task/my_env_cfg.py
from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg

from MyProject.assets import MY_ROBOT_CONFIG

@configclass
class MyEnvCfg(DirectRLEnvCfg):
    decimation: int = 2
    episode_length_s: float = 20.0
    action_space: int = 6          # Number of joints you control
    observation_space: int = 12    # E.g., joint_pos + joint_vel for each joint
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=3.0
    )

    robot: ArticulationCfg = MY_ROBOT_CONFIG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    ground_plane: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/ground_plane",
        spawn=sim_utils.GroundPlaneCfg(color=(0.8, 0.8, 0.8)),
    )

    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=600.0, color=(1.0, 1.0, 1.0)),
    )
```

### Step 3: Implement the Env (`my_env.py`)
```python
# source/MyProject/MyProject/tasks/direct/my_task/my_env.py
from __future__ import annotations
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from .my_env_cfg import MyEnvCfg

class MyEnv(DirectRLEnv):
    cfg: MyEnvCfg

    def __init__(self, cfg: MyEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)

        # Spawn static elements
        self.cfg.ground_plane.spawn.func(
            self.cfg.ground_plane.prim_path,
            self.cfg.ground_plane.spawn,
        )
        self.cfg.light.spawn.func(
            self.cfg.light.prim_path,
            self.cfg.light.spawn,
        )

        # Register and clone
        self.scene.articulations["robot"] = self.robot
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions)

    def _get_observations(self) -> dict:
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        return {"policy": torch.cat([joint_pos, joint_vel], dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        # Return zeros for teleop; implement for RL training
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        # Reset to default pose
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
```

### Step 4: Register with Gym (`__init__.py`)
```python
# source/MyProject/MyProject/__init__.py
import gymnasium as gym
from .tasks.direct.my_task.my_env_cfg import MyEnvCfg

gym.register(
    id="Isaac-MyRobot-v0",
    entry_point="MyProject.tasks.direct.my_task.my_env:MyEnv",
    kwargs={"cfg": MyEnvCfg()},
)
```

### Step 5: Write the Script (`scripts/run_my_robot.py`)
```python
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ↓ All isaaclab imports MUST come after AppLauncher
import gymnasium as gym
import torch
import MyProject  # triggers __init__.py, registers the gym env

env_cfg = MyEnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs
env = gym.make("Isaac-MyRobot-v0", cfg=env_cfg).unwrapped

obs, _ = env.reset()

while simulation_app.is_running():
    with torch.inference_mode():
        action = torch.zeros(env.num_envs, env.cfg.action_space, device=env.device)
        obs, reward, terminated, truncated, info = env.step(action)

env.close()
simulation_app.close()
```

### Step 6: Run It
```bash
# From the project root (Kuka_Med_7/), using the Isaac Lab python:
~/.local/share/ov/pkg/isaac-sim-4.x.x/python.sh scripts/run_my_robot.py --num_envs 1
```

---

## Quick Reference — Common Patterns

### Convert Euler Angles to Quaternion
```python
from isaacsim.core.utils.torch.rotations import euler_angles_to_quats
import torch
# Input: degrees, output: (w, x, y, z) tensor
quat = euler_angles_to_quats(torch.tensor([0.0, 0.0, 90.0]), degrees=True)
```

### Normalize a Quaternion
```python
quat = quat / quat.norm(dim=-1, keepdim=True)
```

### SceneEntityCfg — Select Specific Joints/Bodies
```python
from isaaclab.managers import SceneEntityCfg
robot_cfg = SceneEntityCfg(
    "robot",
    joint_names=["lbr_A.*"],  # Regex
    body_names=["lbr_link_7"]
)
robot_cfg.resolve(env.scene)
# Now use:
robot_cfg.joint_ids   # list of matching joint indices
robot_cfg.body_ids    # list of matching body indices
```

### Subtract Frame Transforms (for IK)
```python
from isaaclab.utils.math import subtract_frame_transforms
ee_pos_b, ee_quat_b = subtract_frame_transforms(
    root_pose_w[:, :3], root_pose_w[:, 3:7],
    ee_pose_w[:, :3],   ee_pose_w[:, 3:7],
)
```

### Differential IK Controller
```python
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

ik_cfg = DifferentialIKControllerCfg(
    command_type="pose",          # "pose" or "position"
    use_relative_mode=True,       # Delta from current pose
    ik_method="dls",              # Damped least squares
)
ik_ctrl = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)

# Each step:
ik_ctrl.set_command(delta_pose, ee_pos_b, ee_quat_b)
joint_targets = ik_ctrl.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
obs, _, _, _, _ = env.step(joint_targets)
```

---

*This guide is intentionally opinionated and based on real patterns from this project. Cross-reference with `med7_env.py`, `med7_env_cfg.py`, `assets.py`, and `teleop_med7.py` for working implementations of every concept described here.*
