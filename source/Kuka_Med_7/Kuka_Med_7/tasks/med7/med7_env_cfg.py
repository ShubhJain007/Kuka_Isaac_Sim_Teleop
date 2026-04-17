
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg

from Kuka_Med_7.assets import MED7_CONFIG

# XR anchor — default (0,0,0) + identity quat = Isaac Lab / Kit default camera view.
XR_ANCHOR_X_M = 0.0
XR_ANCHOR_Y_M = 0.0
XR_ANCHOR_Z_M = 0.0


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

    # robot — link_0 at world origin (0,0,0). All coordinates are relative to arm base.
    robot: ArticulationCfg = MED7_CONFIG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),  # link_0 = world origin
            rot=(1.0, 0.0, 0.0, 0.0),  # Identity — no rotation
            joint_pos={
                "lbr_A1": 0.0,
                "lbr_A2": 0.5,
                "lbr_A3": 0.0,
                "lbr_A4": -1.0,
                "lbr_A5": 0.0,
                "lbr_A6": 1.0,
                "lbr_A7": 0.0,
            },
        ),
    )

    # Hospital bed removed from AR scene — replaced by the real physical table detected by
    # Apple Vision Pro ARKit plane detection.  In non-AVP (Isaac Sim only) mode you can
    # re-enable this by un-commenting the block below.
    #
    # hospital_bed = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/HospitalBed",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(2.0, 1.0, 1.651),
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #             diffuse_color=(0.0, 0.0, 0.0),
    #             metallic=0.05,
    #             roughness=0.7,
    #         ),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.8255)),
    # )

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

    # Black cuboid pedestal on which the arm is mounted.
    #   Length 42 in = 1.0668 m, breadth 30 in = 0.762 m.
    #   Centre shifted 5 cm down: -0.31355 - 0.05 = -0.36355.
    #   Bottom 20 cm below ground plane (-0.8271 - 0.20 = -1.0271).
    #   Top = 2*centre - bottom = +0.30. Size Z = top - bottom = 1.3271 m.
    # Visual-only: no collision_props → arm links pass through without contact.
    base_pedestal = AssetBaseCfg(
        prim_path="/World/envs/env_.*/BasePedestal",
        spawn=sim_utils.CuboidCfg(
            size=(1.0668, 0.762, 1.3271),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.53, 0.81, 0.98),
                metallic=0.05,
                roughness=0.7,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.36355)),
    )

    # ground plane (black) — unchanged at z = -0.8271; pedestal now pokes
    # 20 cm below it.
    ground_plane = AssetBaseCfg(
        prim_path="/World/ground_plane",
        spawn=sim_utils.GroundPlaneCfg(
            color=(0.0, 0.0, 0.0),   # black
            usd_path="/home/kneepolean/IsaacLab/source/isaaclab_assets/isaaclab_assets/Environments/Grids/default_environment.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.8271)),
    )

    # --- XR / OpenXR Configuration ---
    # Used by OpenXRDevice in scripts/teleop_med7_vr.py (see Isaac Lab XrCfg docstring).
    #
    # anchor_pos: WORLD (x,y,z) mapped to XR playspace origin. Prim path: /World/XRAnchor (default).
    # Adjust POV: edit XR_ANCHOR_*_M in module header. See xr_pov_helpers.py for runtime tweaks.
    # anchor_prim_path: set to a parent prim (e.g. robot root) only if you want dynamic follow — then
    # anchor_pos is an OFFSET from that prim, not world coordinates (see Isaac Lab XrCfg).
    # anchor_rot: (w,x,y,z); identity keeps world upright.
    xr = OpenXRDeviceCfg(
        xr_cfg=XrCfg(
            anchor_pos=(XR_ANCHOR_X_M, XR_ANCHOR_Y_M, XR_ANCHOR_Z_M),
            anchor_rot=(1.0, 0.0, 0.0, 0.0),
        )
    )

    # Hand Visualization Proxy - a semi-transparent sphere to show tracked hand position
    hand_proxy = RigidObjectCfg(
        prim_path="/World/envs/env_.*/HandProxy",
        spawn=sim_utils.SphereCfg(
            radius=0.006,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),
                opacity=0.3,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.80), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Surgical probe — registered as RigidObject so avp_stream tracks its pose
    probe = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Probe",
        spawn=sim_utils.SphereCfg(
            radius=0.0072,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 1.0),
                opacity=1.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.263, 0.181, 0.449), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # ---------------------------------------------------------------
    # Bone initial poses — LEFT/RIGHT knee joint, 45° interior angle.
    # Knee joint at (0.90, 0.0, 0.75), bones touching at that point.
    # Matches mock_bone_publisher.py t=0 pose (no swing).
    #
    # Front view (from robot, looking along +X):
    #
    #      femur /   (up-left, direction (0, -0.707, +0.707))
    #           /
    #          ●───────────  tibia (horizontal +Y, direction (0, 1, 0))
    #
    # Interior angle = 45°  ✓   Rotation axis = X  ✓
    #
    # USD mesh 'up' axis = +Z.  Orientations:
    #   Femur  : rotate +Z → (0, −0.7071, +0.7071) = +X rotation by 45°
    #     quat (w,x,y,z) = (cos22.5°,  sin22.5°, 0, 0) = (0.9239,  0.3827, 0, 0)
    #   Tibia  : rotate +Z → (0, 1, 0) = −X rotation by 90°
    #     quat (w,x,y,z) = (cos45°, −sin45°, 0, 0) = (0.7071, −0.7071, 0, 0)
    #
    # Positions seeded from STL geometric centres (both centre at knee joint):
    #   Femur STL centre (m): (−0.004, 0.0147, −0.0026)  → rotate by femur_quat
    #   Tibia STL centre (m): (−0.0892, −0.0374, 0.0833) → rotate by tibia_quat
    # ◄ After rotate, subtract from knee to get USD origin world pos.
    # Pre-computed below (rounded to 4 dp).
    # ---------------------------------------------------------------
    femur = AssetBaseCfg(
        prim_path="/World/envs/env_.*/femur",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/kneepolean/Isaac_Lab_projects/Kuka_Med_7/femur_cut_usd/Draw_Left_Femur_Plan_Array_V2.usd",
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            # Rotated additional 180° about Y (total = 360° = identity)
            pos=(0.904,  0.133,  0.7442),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    tibia = AssetBaseCfg(
        prim_path="/World/envs/env_.*/tibia",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/kneepolean/Isaac_Lab_projects/Kuka_Med_7/tibia_cut_usd/tibia_cut.usd",
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            # q_tibia = (0.7071, −0.7071, 0, 0) → rotate((−0.0892,−0.0374,0.0833))
            # +Z→+Y:  x'=x, y'=−z, z'=y  (90° about −X)
            # rotated offset = (−0.0892, −0.0833, −0.0374)
            # pos = (0.90−0.0892*(−0.0892), 0.0−(−0.0833), 0.75−(−0.0374))
            # simpler: pos = (0.90, 0.0, 0.75) − rotate(offset, q_tibia)
            # rotate((−0.0892,−0.0374,0.0833), (0.7071,−0.7071,0,0)):
            #   Using R_x(-90°): x'=x, y'=z, z'=−y
            #   = (−0.0892, 0.0833, 0.0374)
            # pos = (0.90−(−0.0892), 0.0−0.0833, 0.75−0.0374)
            #      = (0.9892, −0.0833, 0.7126)
            pos=(0.9892, -0.0833,  0.7126),
            rot=(0.0, 0.0, 0.0, 0.7071),
        ),
    )
    # End-effector mount — registered as RigidObject (kinematic, no physics) so that
    # avp_stream streams its pose every frame (streamer.py:3261-3272). Static AssetBaseCfg
    # poses are only sent for the first 5 frames — useless for something that moves.
    # Pose is synced to lbr_link_7 each step in Med7Env._pre_physics_step.
    # Offset from link_7 frame: xyz=(0, 0, 0.189), rpy=(0, π, 0)  →  quat (w,x,y,z) = (0, 0, 1, 0).
    ee_mount = RigidObjectCfg(
        prim_path="/World/envs/env_.*/EEMount",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/kneepolean/Isaac_Lab_projects/Kuka_Med_7/final_mount_usd/Final_Mount.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.5),
            rot=(0.0, 0.0, 1.0, 0.0),
        ),
    )

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
