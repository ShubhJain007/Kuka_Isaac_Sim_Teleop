
from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaacsim.core.utils.torch.rotations import euler_angles_to_quats
import torch

from isaaclab.sensors import CameraCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg

from Kuka_Med_7.assets import MED7_CONFIG

# Hospital color palette (linear sRGB)
# Bed frame: clinical white/off-white
BED_COLOR = (0.85, 0.88, 0.90)
# Mount/stand: light steel blue-grey (typical hospital equipment)
MOUNT_COLOR = (0.72, 0.78, 0.82)


@configclass
class Med7EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5000.0
    action_space = 7
    observation_space = 14
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = MED7_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")

    # Robot Mount — steel-blue cuboid (40cm x 40cm x 50cm tall)
    robot_mount = AssetBaseCfg(
        prim_path="/World/envs/env_.*/RobotMount",
        spawn=sim_utils.CuboidCfg(
            size=(0.40, 0.40, 0.50),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=MOUNT_COLOR,
                metallic=0.4,
                roughness=0.5,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25)),
    )

    # Hospital Bed — white cuboid (2.0m x 1.0m x 0.55m tall)
    hospital_bed = AssetBaseCfg(
        prim_path="/World/envs/env_.*/HospitalBed",
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 1.0, 0.55),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=BED_COLOR,
                metallic=0.05,
                roughness=0.7,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.3, 0.0, 0.275)),
    )

    # Realistic hospital lighting:
    # Primary: cool-white overhead dome (fluorescent-like, 6500K)
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=800.0,
            color=(0.92, 0.95, 1.0),   # cool white / daylight fluorescent
            enable_color_temperature=True,
            color_temperature=6500.0,
        ),
    )

    # Distant fill light from a window direction (warm sunlight)
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=2000.0,
            color=(1.0, 0.97, 0.88),   # warm daylight
            angle=0.53,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=(0.906, 0.0, -0.423, 0.0),  # ~50 deg elevation from side
        ),
    )

    # ground plane (light grey hospital floor)
    ground_plane = AssetBaseCfg(
        prim_path="/World/ground_plane",
        spawn=sim_utils.GroundPlaneCfg(
            color=(0.78, 0.80, 0.82),   # light grey linoleum
        ),
    )

    # --- Teleoperation Cameras ---

    # Wrist Camera — attached to robot end-effector (lbr_link_7)
    wrist_camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/lbr_link_7/wrist_cam",
        update_period=0.0,
        height=224,
        width=224,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=100.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.05, 0.03),  # offset from the link flange
            rot=(0.5, -0.5, 0.5, -0.5),  # looking "forward" along the tool axis (ROS convention)
            convention="ros",
        ),
    )

    # Room Camera — static third-person view
    room_camera = CameraCfg(
        prim_path="/World/envs/env_.*/room_camera",
        update_period=0.0,
        height=224,
        width=224,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=100.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-1.5, 1.0, 1.5),   # Opposite corner of the bed
            rot=euler_angles_to_quats(torch.tensor([200.0, 0.0, 180.0]), degrees=True),
            convention="ros",
        ),
    )

    # --- XR / OpenXR Configuration ---
    # Used by OpenXRDevice in scripts/teleop_med7_vr.py
    # Used by OpenXRDevice in scripts/teleop_med7_vr.py
    xr = OpenXRDeviceCfg(
        xr_cfg=XrCfg(
            anchor_pos=(0.2, 0.0, 0.2),
            anchor_rot=(0.7071, 0.0, 0.0, -0.7071),
        )
    )

    # Hand Visualization Proxy - a semi-transparent sphere to show tracked hand position
    hand_proxy = RigidObjectCfg(
        prim_path="/World/envs/env_.*/HandProxy",
        spawn=sim_utils.SphereCfg(
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0), # Lime green
                opacity=0.5,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5), rot=(1.0, 0.0, 0.0, 0.0)), # Visible initially
    )
    femur = RigidObjectCfg(
        prim_path="/World/envs/env_.*/femur",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/kneepolean/Isaac_Lab_projects/Kuka_Med_7/femur_cut_usd/Draw_Left_Femur_Plan_Array_V2.usd",
            scale=(1.0, 1.0, 1.0), # USD is already scaled to meters
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.3, 0.0, 0.6), rot=(1.0, 0.0, 0.0, 0.0)), # On the bed
    )

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # Robot sits on top of the mount (mount top = 0.50m)
        self.robot.init_state.pos = (0.0, 0.0, 0.50)
