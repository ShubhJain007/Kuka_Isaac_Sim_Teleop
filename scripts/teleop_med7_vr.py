# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# VR surgical planning (Vision Pro + CloudXR): pointer, guide line, bone translucency.
# Default: **hand gestures only** (no keyboard). Use --keyboard-fallback for dev keys.
#
# POINTER (right hand, Play on):
#   1) Right wrist — move / rotate the cyan probe (until you lock it).
#   2) Left thumb–index **short pinch** — lock probe position & orientation where it is.
#   3) Right thumb–index **distance** — adjust guide line length (only after position lock).
#   4) Left thumb–index **short pinch** again — lock & fix guide line length (saves line anchor).
#   5) Left thumb–index **hold ≥ 3 s** — unlock probe so you can move it again (clears both locks).
#
# Left thumb–little spread — bone opacity (wider = more opaque).

from __future__ import annotations

import contextlib
import argparse
import pathlib
import sys
import time

# ── NumPy 2.x compatibility shim ─────────────────────────────────────────────
# numpy 2.0 removed / renamed several symbols that avp_stream dependencies AND
# Isaac Sim's bundled numpy 1.x testing code still reference when running under
# numpy 2.x.  Patch everything before any other import.
import numpy as _np
import numpy.lib.stride_tricks as _nst

# broadcast_to removed from numpy.lib.stride_tricks in numpy 2.0
if not hasattr(_nst, "broadcast_to"):
    _nst.broadcast_to = _np.broadcast_to

# _no_nep50_warning: numpy 1.x internal used as @np._no_nep50_warning() in
# Isaac Sim's bundled numpy testing utilities.  Must be a callable that returns
# a decorator (i.e. a decorator factory).  Provide a no-op version.
if not hasattr(_np, "_no_nep50_warning"):
    def _nep50_noop_factory():
        def _deco(f): return f
        return _deco
    _np._no_nep50_warning = _nep50_noop_factory
    del _nep50_noop_factory

# Aliases removed in numpy 2.0
_NP2_ALIASES = {
    # type aliases
    "float_":    "float64",
    "complex_":  "complex128",
    "bool8":     "bool_",
    "int0":      "intp",
    "uint0":     "uintp",
    "object0":   "object_",
    # function aliases
    "round_":    "round",
    "product":   "prod",
    "cumproduct":"cumprod",
    "sometrue":  "any",
    "alltrue":   "all",
    "in1d":      "isin",
    "row_stack": "vstack",
}
for _old, _new in _NP2_ALIASES.items():
    if not hasattr(_np, _old):
        try:
            _v = _np
            for _p in _new.split("."):
                _v = getattr(_v, _p)
            setattr(_np, _old, _v)
        except AttributeError:
            pass

del _nst, _NP2_ALIASES, _old, _new, _v, _p, _np
# ─────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# ROS setup MUST happen before any other imports to avoid Python version conflicts
if "--ros" in sys.argv:
    # Remove system ROS from Python path to avoid conflicts with Isaac Sim's bundled ROS2
    sys.path = [p for p in sys.path if "/opt/ros/" not in p and "site-packages/ros" not in p]
    
    from med7_ros_bones import setup_ros2_libs
    setup_ros2_libs()  # Sets up Isaac Sim's bundled ROS2 and may re-exec
    
    from med7_ros_pointcloud import make_ros_integration_node
    from usd_pointcloud_viz import create_or_update_pointcloud_prim

import os

from udp_rtt_probe import UdpRttProbe

import gymnasium as gym
import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="VR surgical planning — pointer & line; gestures by default."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="VR delta scale.")
parser.add_argument(
    "--ros",
    action="store_true",
    help="ROS integration: point clouds (/tracked/femur), robot sync (/lbr/joint_states).",
)
parser.add_argument(
    "--keyboard-fallback",
    action="store_true",
    help="Enable host keyboard (arrows, K, P, …) in addition to gestures.",
)
parser.add_argument(
    "--avp",
    type=str,
    default=None,
    metavar="IP_OR_CODE",
    help=(
        "Enable Apple Vision Pro AR mode via avp_stream. "
        "Provide Vision Pro IP (e.g. 192.168.1.42) for local Wi-Fi, "
        "or a room code (e.g. ABC-1234) for cross-network. "
        "Hand tracking switches to Vision Pro ARKit; scene streams to AR."
    ),
)
parser.add_argument(
    "--avp-z",
    type=float,
    default=0.9271,
    metavar="METRES",
    help=(
        "Height (m) of robot arm base above the real floor detected by AVP. "
        "Default 0.9271 m = 36.5 in pedestal height, so the virtual pedestal "
        "bottom lands on the AVP floor and the arm sits on top. Adjust if "
        "the real pedestal / table you're standing next to has a different "
        "height (so the virtual and real surfaces line up)."
    ),
)
parser.add_argument(
    "--avp-yaw",
    type=float,
    default=0.0,
    metavar="DEGREES",
    help="Yaw rotation of sim X-axis relative to Vision Pro X-axis (default: 0).",
)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--debug-xr", action="store_true", help="Extra diagnostics.")
parser.add_argument(
    "--log-render-bones",
    type=float,
    default=0.0,
    metavar="SECONDS",
    help="Record actual rendered femur/tibia USD poses for DUR seconds, "
         "write CSV to scripts/rendered_bones.csv (plot with plot_rendered_bone.py).",
)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.utils.math as math_utils
from isaaclab.devices import OpenXRDevice
from isaaclab.devices.openxr.retargeters.manipulator import Se3RelRetargeter, Se3RelRetargeterCfg
from pxr import Gf, UsdGeom, UsdShade

import Kuka_Med_7  # noqa: F401
from isaaclab.managers import SceneEntityCfg
from Kuka_Med_7.tasks.med7.med7_env_cfg import Med7EnvCfg

# PoseEKF lives under ir_tracking/scripts — reuse for stylus smoothing.
_IR_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "ir_tracking" / "scripts"
if str(_IR_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_IR_SCRIPTS))
try:
    from pose_ekf import PoseEKF  # noqa: E402
except ModuleNotFoundError:
    import importlib.util
    _spec = importlib.util.spec_from_file_location("pose_ekf", str(_IR_SCRIPTS / "pose_ekf.py"))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    PoseEKF = _mod.PoseEKF

def vr_delta_to_probe_deltas(
    delta_pose: torch.Tensor,
    transform_matrix: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    d = delta_pose.to(device)
    if d.dim() == 2:
        d = d[0]
    delta_pos_h = torch.cat([d[:3], torch.ones(1, device=device)])
    delta_pos = torch.matmul(delta_pos_h, transform_matrix)[:3]
    delta_rot_quat = math_utils.quat_from_euler_xyz(d[3].unsqueeze(0), d[4].unsqueeze(0), d[5].unsqueeze(0))
    delta_rot_mat = math_utils.matrix_from_quat(delta_rot_quat)
    t = transform_matrix[:3, :3]
    delta_rot_mat = torch.matmul(t, torch.matmul(delta_rot_mat, t.T))
    delta_rot_quat = math_utils.quat_from_matrix(delta_rot_mat)
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(delta_rot_quat)
    euler = torch.stack([roll[0], pitch[0], yaw[0]])
    return delta_pos, euler


def _pinch_dist(hand: dict) -> float | None:
    tt = hand.get("thumb_tip")
    it = hand.get("index_tip")
    if tt is None or it is None or len(tt) < 3 or len(it) < 3:
        return None
    return float(np.linalg.norm(np.array(tt[:3], dtype=np.float64) - np.array(it[:3], dtype=np.float64)))


# OpenXR hand topology — line segments for stick-figure visualization
_HAND_STICK_EDGES: list[tuple[str, str]] = [
    ("wrist", "palm"),
    ("palm", "thumb_metacarpal"),
    ("thumb_metacarpal", "thumb_proximal"),
    ("thumb_proximal", "thumb_distal"),
    ("thumb_distal", "thumb_tip"),
    ("palm", "index_metacarpal"),
    ("index_metacarpal", "index_proximal"),
    ("index_proximal", "index_intermediate"),
    ("index_intermediate", "index_distal"),
    ("index_distal", "index_tip"),
    ("palm", "middle_metacarpal"),
    ("middle_metacarpal", "middle_proximal"),
    ("middle_proximal", "middle_intermediate"),
    ("middle_intermediate", "middle_distal"),
    ("middle_distal", "middle_tip"),
    ("palm", "ring_metacarpal"),
    ("ring_metacarpal", "ring_proximal"),
    ("ring_proximal", "ring_intermediate"),
    ("ring_intermediate", "ring_distal"),
    ("ring_distal", "ring_tip"),
    ("palm", "little_metacarpal"),
    ("little_metacarpal", "little_proximal"),
    ("little_proximal", "little_intermediate"),
    ("little_intermediate", "little_distal"),
    ("little_distal", "little_tip"),
]

def _joint_xyz(hand: dict, name: str) -> np.ndarray | None:
    j = hand.get(name)
    if j is None or len(j) < 3:
        return None
    return np.array(j[:3], dtype=np.float64)


def _ensure_hand_stick_curves(stage, path: str, color_rgb: tuple[float, float, float]) -> UsdGeom.BasisCurves:
    p = stage.GetPrimAtPath(path)
    if not p.IsValid():
        bc = UsdGeom.BasisCurves.Define(stage, path)
        try:
            bc.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
        except Exception:
            bc.CreateTypeAttr().Set("linear")
        bc.GetDisplayColorAttr().Set([color_rgb])
        p = bc.GetPrim()
    return UsdGeom.BasisCurves(p)


def _update_hand_stick_figure(stage, path: str, hand: dict, color_rgb: tuple[float, float, float]) -> None:
    if not hand:
        return
    points: list[Gf.Vec3f] = []
    counts: list[int] = []
    for a, b in _HAND_STICK_EDGES:
        pa = _joint_xyz(hand, a)
        pb = _joint_xyz(hand, b)
        if pa is None or pb is None:
            continue
        points.append(Gf.Vec3f(float(pa[0]), float(pa[1]), float(pa[2])))
        points.append(Gf.Vec3f(float(pb[0]), float(pb[1]), float(pb[2])))
        counts.append(2)
    if not points:
        return
    try:
        bc = _ensure_hand_stick_curves(stage, path, color_rgb)
        bc.CreatePointsAttr().Set(points)
        bc.CreateCurveVertexCountsAttr().Set(counts)
        n = len(points)
        bc.CreateWidthsAttr().Set([0.006] * n)
        bc.GetDisplayColorAttr().Set([color_rgb] * n)
    except Exception as e:
        if not getattr(_update_hand_stick_figure, "_warned", False):
            print(f"[WARN] Hand stick figure update failed (non-fatal): {e}")
            _update_hand_stick_figure._warned = True  # type: ignore[attr-defined]


def _spread_thumb_pinky(hand: dict) -> float | None:
    tt = hand.get("thumb_tip")
    pt = hand.get("little_tip")
    if tt is None or pt is None or len(tt) < 3 or len(pt) < 3:
        return None
    return float(np.linalg.norm(np.array(tt[:3], dtype=np.float64) - np.array(pt[:3], dtype=np.float64)))


# ── Apple Vision Pro (avp_stream) hand adapter ────────────────────────────
# Maps avp_stream HandData (27×4×4 world-frame transforms) → the OpenXR-style
# dict that _joint_xyz / _pinch_dist / _spread_thumb_pinky / _update_hand_stick_figure
# already understand, so zero downstream code changes are needed.

_AVP_TO_OPENXR: dict[str, str] = {
    "wrist":              "wrist",
    "thumbKnuckle":       "thumb_metacarpal",
    "thumbIntermediateBase": "thumb_proximal",
    "thumbIntermediateTip":  "thumb_distal",
    "thumbTip":           "thumb_tip",
    "indexMetacarpal":    "index_metacarpal",
    "indexKnuckle":       "index_proximal",
    "indexIntermediateBase": "index_intermediate",
    "indexIntermediateTip":  "index_distal",
    "indexTip":           "index_tip",
    "middleMetacarpal":   "middle_metacarpal",
    "middleKnuckle":      "middle_proximal",
    "middleIntermediateBase": "middle_intermediate",
    "middleIntermediateTip":  "middle_distal",
    "middleTip":          "middle_tip",
    "ringMetacarpal":     "ring_metacarpal",
    "ringKnuckle":        "ring_proximal",
    "ringIntermediateBase":  "ring_intermediate",
    "ringIntermediateTip":   "ring_distal",
    "ringTip":            "ring_tip",
    "littleMetacarpal":   "little_metacarpal",
    "littleKnuckle":      "little_proximal",
    "littleIntermediateBase": "little_intermediate",
    "littleIntermediateTip":  "little_distal",
    "littleTip":          "little_tip",
}

# avp_stream joint index → avp_stream attribute name (for iteration)
_AVP_JOINT_NAMES = [
    "wrist",
    "thumbKnuckle", "thumbIntermediateBase", "thumbIntermediateTip", "thumbTip",
    "indexMetacarpal", "indexKnuckle", "indexIntermediateBase", "indexIntermediateTip", "indexTip",
    "middleMetacarpal", "middleKnuckle", "middleIntermediateBase", "middleIntermediateTip", "middleTip",
    "ringMetacarpal", "ringKnuckle", "ringIntermediateBase", "ringIntermediateTip", "ringTip",
    "littleMetacarpal", "littleKnuckle", "littleIntermediateBase", "littleIntermediateTip", "littleTip",
]


def _avp_hand_to_openxr_dict(hand_data) -> dict:
    """Convert avp_stream HandData (N×4×4) → OpenXR-style flat dict.

    Each value is a list [x, y, z, qw, qx, qy, qz] matching what the
    existing _joint_xyz / _pinch_dist / wrist pose helpers expect.
    Returns {} if hand_data is None.
    """
    if hand_data is None:
        return {}
    out = {}
    from scipy.spatial.transform import Rotation as _R
    for avp_name, oxr_name in _AVP_TO_OPENXR.items():
        idx = _AVP_JOINT_NAMES.index(avp_name) if avp_name in _AVP_JOINT_NAMES else -1
        if idx < 0 or idx >= hand_data.shape[0]:
            continue
        mat = hand_data[idx]          # 4×4 world-frame transform
        pos = mat[:3, 3]              # xyz
        rot_mat = mat[:3, :3]
        quat_xyzw = _R.from_matrix(rot_mat).as_quat()   # [x,y,z,w]
        qw, qx, qy, qz = quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]
        out[oxr_name] = [float(pos[0]), float(pos[1]), float(pos[2]), qw, qx, qy, qz]
    return out


# ── Bone ICP tracking helper ──────────────────────────────────────────

def _extract_mesh_verts_world(stage, prim_path: str) -> np.ndarray | None:
    """Extract all Mesh vertices under a prim, transformed to world frame.

    Uses TraverseInstanceProxies to see through USD instances (the bone
    USDs are generated with make_instanceable=true).
    """
    from pxr import UsdGeom, Gf, Usd
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    cache = UsdGeom.XformCache()
    all_pts = []
    for desc in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not desc.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(desc)
        pts_attr = mesh.GetPointsAttr().Get()
        if pts_attr is None or len(pts_attr) == 0:
            continue
        world_xf = cache.GetLocalToWorldTransform(desc)
        pts_np = np.array([[p[0], p[1], p[2]] for p in pts_attr], dtype=np.float64)
        # Vectorized transform: build 4xN matrix, multiply by world_xf
        ones = np.ones((pts_np.shape[0], 1), dtype=np.float64)
        pts_h = np.hstack([pts_np, ones])  # Nx4
        # Gf.Matrix4d is row-major; numpy @ needs column-major, so transpose
        xf_np = np.array(world_xf, dtype=np.float64).reshape(4, 4)
        transformed = (xf_np @ pts_h.T).T[:, :3]  # Nx3
        all_pts.append(transformed)
    if not all_pts:
        return None
    return np.vstack(all_pts)


def _track_bone_icp(bone_name: str, pc_points: np.ndarray, env, stage,
                     ref_pcds: dict, icp_counts: dict, mesh_verts_cache: dict):
    """Move a bone USD prim to match a new point cloud via ICP.

    First call: extracts mesh verts, saves reference PC, computes initial alignment.
    Subsequent calls: ICP reference→new PC, applies transform to bone prim.
    """
    import open3d as o3d
    import time as _t

    icp_counts[bone_name] = icp_counts.get(bone_name, 0) + 1
    count = icp_counts[bone_name]

    if bone_name not in ref_pcds:
        # First point cloud — save as reference and compute initial alignment
        t0 = _t.time()

        # Extract mesh vertices from the bone USD prim (in world frame)
        prim_path = env._femur_prim_path if bone_name == "femur" else env._tibia_prim_path
        mesh_pts = mesh_verts_cache.get(bone_name)
        if mesh_pts is None:
            mesh_pts = _extract_mesh_verts_world(stage, prim_path)
            if mesh_pts is not None:
                mesh_verts_cache[bone_name] = mesh_pts
                print(f"[BoneTrack] {bone_name}: extracted {mesh_pts.shape[0]} mesh verts from USD")
            else:
                print(f"[BoneTrack] {bone_name}: WARNING - no mesh verts found in {prim_path}")
                return

        # Downsample mesh verts for faster ICP (use every Nth point)
        n_mesh = mesh_pts.shape[0]
        stride = max(1, n_mesh // 2000)
        mesh_ds = mesh_pts[::stride]

        mesh_pcd = o3d.geometry.PointCloud()
        mesh_pcd.points = o3d.utility.Vector3dVector(mesh_ds)

        pc_pcd = o3d.geometry.PointCloud()
        pc_pcd.points = o3d.utility.Vector3dVector(pc_points.astype(np.float64))

        # ICP: find transform that maps mesh → point cloud
        result = o3d.pipelines.registration.registration_icp(
            source=mesh_pcd, target=pc_pcd,
            max_correspondence_distance=0.05,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        T = np.array(result.transformation)

        if result.fitness > 0.1:
            env.set_bone_xform(bone_name, T)
            print(f"[BoneTrack] {bone_name}: initial alignment, fitness={result.fitness:.3f} "
                  f"({(_t.time()-t0)*1000:.0f} ms)")
        else:
            print(f"[BoneTrack] {bone_name}: initial ICP fitness too low ({result.fitness:.3f}), "
                  "bone may not overlap with PC — try adjusting initial pose in env_cfg")

        # Save reference PC for subsequent ICP updates
        ref_pcds[bone_name] = pc_pcd
        # Store the initial alignment transform so we can compose with future deltas
        mesh_verts_cache[bone_name + "_T"] = T
    else:
        # Subsequent: ICP reference → new PC → delta transform
        t0 = _t.time()
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(pc_points.astype(np.float64))

        result = o3d.pipelines.registration.registration_icp(
            source=ref_pcds[bone_name], target=new_pcd,
            max_correspondence_distance=0.02,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        if result.fitness < 0.3:
            if count <= 5:
                print(f"[BoneTrack] {bone_name}: ICP fitness low ({result.fitness:.2f}), skipping")
            return

        # Compose: initial_alignment @ delta → total transform on the mesh
        T_init = mesh_verts_cache.get(bone_name + "_T", np.eye(4))
        T_delta = np.array(result.transformation)
        T_total = T_delta @ T_init
        env.set_bone_xform(bone_name, T_total)

        dt_ms = (_t.time() - t0) * 1000
        if count <= 3 or count % 100 == 0:
            print(f"[BoneTrack] {bone_name}: ICP #{count}, fitness={result.fitness:.3f} ({dt_ms:.1f} ms)")


# Cyan pointer sphere radius (m); diameter = 2 * this. 10 mm diameter.
PROBE_SPHERE_RADIUS_M = 0.005

# Left pinch: short tap locks probe then line length; hold ≥3 s unlocks (see header).
LEFT_PINCH_TAP_MIN_S = 0.03
LEFT_PINCH_TAP_MAX_S = 0.38
LEFT_PINCH_HOLD_UNLOCK_S = 3.0
LEFT_PINCH_COOLDOWN_S = 0.45
LEFT_DOUBLE_TAP_WINDOW_S = 0.5  # Max gap between two taps to count as double-tap


def main():
    import carb
    import omni.appwindow
    import omni.usd

    env_cfg = Med7EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make("Isaac-Med7-v0", cfg=env_cfg)
    env = env.unwrapped

    ros_node = None
    rclpy_mod = None
    probe_pose_pub = None
    guide_line_pub = None
    if args_cli.ros:
        import rclpy as _rclpy
        import omni.usd

        rclpy_mod = _rclpy
        try:
            rclpy_mod.init()
        except RuntimeError as e:
            if "already" not in str(e).lower():
                raise
        ros_node = make_ros_integration_node(env.device)
        print("[INFO] ROS: /tracked/femur, /tracked/tibia (PointCloud2), /lbr/robot_description, /lbr/joint_states")

        from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Point, Quaternion
        probe_pose_pub = ros_node.create_publisher(PoseStamped, "/surgical_plan/probe_pose", 10)
        guide_line_pub = ros_node.create_publisher(PoseArray, "/surgical_plan/guide_line", 10)
        print("[INFO] ROS publishers: /surgical_plan/probe_pose (PoseStamped), /surgical_plan/guide_line (PoseArray)")

        stage = omni.usd.get_context().get_stage()

    sens = args_cli.sensitivity
    teleoperation_active = False

    def start_teleop():
        nonlocal teleoperation_active
        teleoperation_active = True
        print("[INFO] VR planning active (Play).")

    def stop_teleop():
        nonlocal teleoperation_active
        teleoperation_active = False
        print("[INFO] VR planning paused (Stop).")

    teleop_interface = None
    if args_cli.avp:
        print("[INFO] --avp mode: skipping OpenXR/CloudXR device (using AVP hand tracking instead).")
    else:
        try:
            retargeter_r = Se3RelRetargeter(
                Se3RelRetargeterCfg(
                    bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
                    zero_out_xy_rotation=False,
                    use_wrist_position=True,
                    use_wrist_rotation=True,
                    delta_rot_scale_factor=5.0 * sens,
                    delta_pos_scale_factor=5.0 * sens,
                )
            )
            retargeter_l = Se3RelRetargeter(
                Se3RelRetargeterCfg(
                    bound_hand=OpenXRDevice.TrackingTarget.HAND_LEFT,
                    zero_out_xy_rotation=False,
                    use_wrist_position=True,
                    use_wrist_rotation=True,
                    delta_rot_scale_factor=4.0 * sens,
                    delta_pos_scale_factor=0.02 * sens,
                )
            )
            teleop_interface = OpenXRDevice(env_cfg.xr, retargeters=[retargeter_r, retargeter_l])
            teleop_interface.add_callback("START", start_teleop)
            teleop_interface.add_callback("STOP", stop_teleop)
            teleop_interface.reset()
            _xc = env_cfg.xr.xr_cfg
            if _xc is not None:
                print(
                    "[INFO] XR view: XrCfg.anchor_pos is the WORLD point mapped to your headset origin "
                    f"(not eye position): {_xc.anchor_pos}; anchor_rot (wxyz) {_xc.anchor_rot}. "
                    "Edit in med7_env_cfg.py. Right controller A toggles anchor rotation (Isaac Lab default)."
                )
        except Exception as _xr_err:
            print(f"[WARNING] OpenXR device not available: {_xr_err}")
            print("[WARNING] Running without VR hand tracking. AVP hand tracking will still work if --avp is set.")
            teleop_interface = None

    transform_matrix = torch.tensor(
        [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=env.device,
    )

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["lbr_A.*"], body_names=["lbr_link_7"])
    robot_entity_cfg.resolve(env.scene)
    joint_ids = robot_entity_cfg.joint_ids

    settings = carb.settings.get_settings()
    settings.set_bool("/xr/openxr/components/omni.kit.xr.openxr.ext.hand_tracking/enabled", True)
    settings.set_bool("/xr/openxr/components/isaacsim.xr.openxr.hand_tracking/enabled", True)
    settings.set_bool("/persistent/xr/profile/ar/enabled", True)

    if os.environ.get("EXTERNAL_RENDERER") != "cloudxr":
        print("[WARNING] EXTERNAL_RENDERER is not 'cloudxr'. Run: source scripts/cloudxr_env.sh")

    # Probe is now a registered RigidObject at /World/envs/env_0/Probe
    probe_prim_path    = "/World/envs/env_0/Probe"
    line_prim_path     = "/World/envs/env_0/SurgicalPlan/GuideLine"
    stylus_prim_path   = "/World/envs/env_0/SurgicalPlan/StylusTracker"
    probe_prim  = None
    line_prim   = None
    stylus_prim = None   # root Xform that receives Muse stylus 6-DOF pose

    # Muse stylus geometry: physical tip is this far from the pose origin along local -Z
    STYLUS_TIP_OFFSET = 0.045   # metres

    def _set_translucency(prim, opacity=0.5):
        if prim.IsA(UsdGeom.Gprim):
            UsdGeom.Gprim(prim).GetDisplayOpacityAttr().Set([opacity])
        mat_binding = UsdShade.MaterialBindingAPI(prim)
        if mat_binding:
            material = mat_binding.GetDirectBinding().GetMaterial()
            if material:
                shader, _, _ = material.ComputeSurfaceSource()
                if shader:
                    op_input = shader.GetInput("opacity")
                    if op_input:
                        op_input.Set(opacity)
                    else:
                        shader.CreateInput("opacity", UsdShade.SdfValueTypeNames.Float).Set(opacity)
        for child in prim.GetChildren():
            _set_translucency(child, opacity)

    def _setup_surgical_plan_prims(st):
        """Create hand-stick scope, guide line, and Muse stylus assembly.

        StylusTracker (Xform) — receives full 6-DOF from Muse stylus
            └── StylusTip   (Sphere)    — at -Z offset = physical tip
            └── ApproachLine (Cylinder) — along -Z axis = approach vector
        """
        for scope in (
            "/World/envs/env_0/SurgicalPlan",
            "/World/envs/env_0/SurgicalPlan/HandStick",
        ):
            if not st.GetPrimAtPath(scope).IsValid():
                UsdGeom.Scope.Define(st, scope)

        # ── Guide line ─────────────────────────────────────────────────────
        pp = st.GetPrimAtPath("/World/envs/env_0/Probe")
        lp = st.GetPrimAtPath(line_prim_path)
        if not lp.IsValid():
            line_cyl = UsdGeom.Cylinder.Define(st, line_prim_path)
            line_cyl.GetRadiusAttr().Set(0.0012)
            line_cyl.GetHeightAttr().Set(0.2)
            line_cyl.GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)])
            lp = line_cyl.GetPrim()
            with contextlib.suppress(Exception):
                UsdGeom.Gprim(lp).CreateDoubleSidedAttr().Set(True)
        UsdGeom.Imageable(lp).MakeInvisible()   # hidden until probe is locked

        # ── StylusTracker root Xform ───────────────────────────────────────
        sp = st.GetPrimAtPath(stylus_prim_path)
        if not sp.IsValid():
            sp = UsdGeom.Xform.Define(st, stylus_prim_path).GetPrim()

        # Probe tip sphere (child of StylusTracker, offset -Z by tip length)
        tip_path = stylus_prim_path + "/StylusTip"
        if not st.GetPrimAtPath(tip_path).IsValid():
            tip_sphere = UsdGeom.Sphere.Define(st, tip_path)
            tip_sphere.GetRadiusAttr().Set(0.005)    # same radius as probe (10 mm diameter)
            tip_sphere.GetDisplayColorAttr().Set([(0.0, 1.0, 0.4)])  # green = approach
            # Offset the sphere by STYLUS_TIP_OFFSET along local -Z
            tip_xf = UsdGeom.Xformable(tip_sphere.GetPrim())
            tip_xf.AddTranslateOp().Set(
                Gf.Vec3d(0.0, 0.0, -STYLUS_TIP_OFFSET))

        # Approach line cylinder (child of StylusTracker, along -Z axis)
        aline_path = stylus_prim_path + "/ApproachLine"
        if not st.GetPrimAtPath(aline_path).IsValid():
            aline_cyl = UsdGeom.Cylinder.Define(st, aline_path)
            aline_cyl.GetRadiusAttr().Set(0.0008)          # 0.8 mm radius
            aline_cyl.GetHeightAttr().Set(STYLUS_TIP_OFFSET)  # fills marker→tip
            aline_cyl.GetDisplayColorAttr().Set([(0.0, 1.0, 0.4)])  # green
            # Centre the cylinder between marker and tip (-Z/2)
            aline_xf = UsdGeom.Xformable(aline_cyl.GetPrim())
            aline_xf.AddTranslateOp().Set(
                Gf.Vec3d(0.0, 0.0, -STYLUS_TIP_OFFSET / 2.0))

        # Start invisible — becomes visible once AVP tracks the Muse stylus
        UsdGeom.Imageable(sp).MakeInvisible()

        return pp, lp, sp

    probe_pos = [-0.263, 0.181, 0.449]
    probe_euler_deg = [180.0, 0.0, -90.0]
    probe_roll_deg = 0.0  # Roll around the finger/probe axis (Option C: wrist twist)
    prev_wrist_twist = None  # Previous wrist twist angle for delta computation
    line_depth = 0.20
    line_pos = None
    line_rot = None
    bone_opacity = 0.5

    probe_pos_locked = False
    probe_angle_locked = False
    line_len_locked = False
    left_pinch_prev = False

    # Logitech Muse stylus — edge-tracking for primary/secondary buttons.
    # Primary press   → lock probe position + orientation.
    # Secondary press → unlock probe + clear guide line.
    prev_primary_btn    = False
    prev_secondary_btn  = False
    _stylus_pose_world  = None   # last-seen stylus 4×4 (for ROS approach vector)
    _stylus_debug_printed = False

    # EKF for stylus 6-DOF smoothing. Tuning is meter-scale (PoseEKF is
    # unit-agnostic — we just pick noises that match): trust measurements
    # ~5 mm / ~1° and predictions heavily so jitter washes out.
    stylus_ekf = PoseEKF(
        max_misses=30,
        q_pos=1.0e-6,    # m²   process noise position
        q_vel=1.0e-2,    # m²/s process noise velocity
        q_ang=1.0e-3,    # rad² process noise angle
        q_angvel=1.0,    # rad²/s
        r_pos=2.5e-5,    # m²   measurement noise (~5 mm stddev)
        r_ang=3.0e-4,    # rad² measurement noise (~1° stddev)
    )
    # m33 bit-decode in avp_stream is corrupted by the timestamp fraction, so
    # noise bursts can hold a button "True" for up to ~16 frames. We require
    # the button to be held True continuously for STYLUS_BTN_HOLD_S seconds
    # before triggering — far longer than any noise burst.
    STYLUS_BTN_HOLD_S = 2.0
    _prim_true_since = None   # monotonic time when primary went True, or None
    _sec_true_since  = None
    _prim_triggered  = False  # latched so one hold only fires once
    _sec_triggered   = False

    # Dwell-lock state
    _dwell_start        = time.time()
    _dwell_ref_pos      = np.array([-0.263, 0.181, 0.449])  # matches probe_pos init
    _dwell_printed      = -1.0   # last printed bucket (avoids spam)
    _locked_pos         = np.zeros(3)
    _unlock_dwell_start = time.time() + 1e9  # far future = inactive
    _unlock_ref_pos     = np.zeros(3)
    t_left_down = 0.0
    long_unlock_fired = False
    last_left_tap_time = 0.0
    left_tap_count = 0  # Counts rapid taps for double-tap detection

    

    # Lock-to-bone: keep probe 6-DOF fixed relative to bone point cloud
    lock_to_bone = False
    lock_bone = "femur"  # "femur" or "tibia"
    lock_offset_local = None  # 3-vec: probe offset in bone-local frame at lock time
    lock_euler_local = None   # 3-vec: probe euler in bone-local frame at lock time
    prev_centroid = None      # Previous centroid for fallback
    prev_axes = None          # Previous PCA axes (3x3 rotation matrix)

    # Auto-attach locked probe to the nearest bone (femur/tibia).
    # On lock: pick nearer bone by distance from probe to bone origin, store probe pose
    # in that bone's local frame. Every frame thereafter: probe world pose = bone world
    # pose ⊕ stored offset, so the probe rigidly follows bone motion. ROS publishing
    # uses the updated probe_pos — so downstream consumers see live world coordinates.
    _auto_bone_name = None       # "femur" | "tibia" | None
    _auto_bone_off  = None       # np.array(3,) position in bone-local frame
    _auto_bone_qoff = None       # np.array(4,) wxyz quaternion in bone-local frame

    def _read_bone_world_pose(bone_name: str):
        """Return (pos, quat_wxyz) of the bone's root xform in world frame, or (None, None)."""
        prim = stage.GetPrimAtPath(f"/World/envs/env_0/{bone_name}")
        if not prim or not prim.IsValid():
            return None, None
        from pxr import UsdGeom as _Ug
        xformable = _Ug.Xformable(prim)
        mat = xformable.ComputeLocalToWorldTransform(0.0)  # Gf.Matrix4d
        pos = np.array(mat.ExtractTranslation())
        rot = mat.ExtractRotation().GetQuaternion()
        w = float(rot.GetReal())
        im = rot.GetImaginary()
        return pos, np.array([w, float(im[0]), float(im[1]), float(im[2])])

    def _qconj(q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def _qmul(a, b):
        aw, ax, ay, az = a; bw, bx, by, bz = b
        return np.array([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ])

    def _qrot(q, v):
        """Rotate 3-vec v by quaternion q (wxyz)."""
        w, x, y, z = q
        vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
        return np.array([
            (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y - z * w) * vy + 2 * (x * z + y * w) * vz,
            2 * (x * y + z * w) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z - x * w) * vz,
            2 * (x * z - y * w) * vx + 2 * (y * z + x * w) * vy + (1 - 2 * (x * x + y * y)) * vz,
        ])

    def _euler_xyz_to_quat(edeg):
        import math as _m
        rx, ry, rz = (_m.radians(d) / 2 for d in edeg)
        cx, sx = _m.cos(rx), _m.sin(rx)
        cy, sy = _m.cos(ry), _m.sin(ry)
        cz, sz = _m.cos(rz), _m.sin(rz)
        return np.array([
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ])

    def _quat_to_euler_xyz_deg(q):
        import math as _m
        w, x, y, z = q
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        rx = _m.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        ry = _m.copysign(_m.pi / 2, sinp) if abs(sinp) >= 1 else _m.asin(sinp)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        rz = _m.atan2(siny_cosp, cosy_cosp)
        return [_m.degrees(rx), _m.degrees(ry), _m.degrees(rz)]

    def _bone_world_aabb_center(bone_name: str):
        """Compute the world-space AABB center of the bone USD mesh.
        Better proxy than the xform origin (which is pinned at the knee for both bones)."""
        prim = stage.GetPrimAtPath(f"/World/envs/env_0/{bone_name}")
        if not prim or not prim.IsValid():
            return None
        from pxr import UsdGeom as _Ug
        try:
            bbox_cache = _Ug.BBoxCache(0.0, [_Ug.Tokens.default_], useExtentsHint=True)
            world_bbox = bbox_cache.ComputeWorldBound(prim)
            rng = world_bbox.ComputeAlignedRange()
            if rng.IsEmpty():
                return None
            lo = rng.GetMin(); hi = rng.GetMax()
            return np.array([(lo[0] + hi[0]) * 0.5,
                             (lo[1] + hi[1]) * 0.5,
                             (lo[2] + hi[2]) * 0.5])
        except Exception:
            return None

    def _nearest_point_on_bone(bone_name: str, probe_world_pos_np):
        """Distance from probe to the nearest surface point of the bone.
        Order of preference:
          1. Min distance to live ROS point cloud (actual bone surface).
          2. Distance to mesh AABB center in world frame.
          3. Distance to xform origin (worst — same at knee for both bones).
        Returns a float distance or None."""
        pts = _get_pc_points(bone_name)
        if pts is not None and pts.shape[0] > 0:
            diff = pts - probe_world_pos_np  # (N, 3)
            return float(np.sqrt((diff * diff).sum(axis=1).min()))
        bbc = _bone_world_aabb_center(bone_name)
        if bbc is not None:
            return float(np.linalg.norm(probe_world_pos_np - bbc))
        bp, _ = _read_bone_world_pose(bone_name)
        if bp is None:
            return None
        return float(np.linalg.norm(probe_world_pos_np - bp))

    def _attach_probe_to_nearest_bone(probe_world_pos_np):
        """Called right after probe gets locked — pick nearer bone, record offset."""
        nonlocal _auto_bone_name, _auto_bone_off, _auto_bone_qoff
        q_probe = _euler_xyz_to_quat(probe_euler_deg)

        distances = {}
        for name in ("femur", "tibia"):
            d = _nearest_point_on_bone(name, probe_world_pos_np)
            if d is not None:
                distances[name] = d

        if not distances:
            print("[PLAN] ✗ No bones available for attachment")
            return

        best_name = min(distances, key=distances.get)
        best_bp, best_bq = _read_bone_world_pose(best_name)
        if best_bp is None:
            return

        # Transform probe into bone-local frame: offset = R_bone^T @ (p_probe - p_bone)
        off_world = probe_world_pos_np - best_bp
        _auto_bone_off  = _qrot(_qconj(best_bq), off_world)
        _auto_bone_qoff = _qmul(_qconj(best_bq), q_probe)
        _auto_bone_name = best_name

        dist_str = " | ".join(f"{n}={d*1000:.1f}mm" for n, d in distances.items())
        print(f"[PLAN] ► Probe attached to {best_name}  ({dist_str})")

    def _detach_probe_from_bone():
        nonlocal _auto_bone_name, _auto_bone_off, _auto_bone_qoff
        if _auto_bone_name is not None:
            print(f"[PLAN] ◄ Probe detached from {_auto_bone_name}")
        _auto_bone_name = None
        _auto_bone_off  = None
        _auto_bone_qoff = None

    def _get_pc_points(which: str):
        """Get raw point cloud array for a bone."""
        if ros_node is None:
            return None
        if which == "femur":
            return ros_node.pc_points
        if which == "tibia":
            return ros_node.tibia_pc_points
        return None

    def _pc_centroid(which: str):
        """Get centroid as 3-element list, or None."""
        pts = _get_pc_points(which)
        if pts is None or pts.shape[0] == 0:
            return None
        c = pts.mean(axis=0)
        return [float(c[0]), float(c[1]), float(c[2])]

    def _pc_pose(which: str):
        """Get centroid + PCA axes (3x3 rotation matrix) from point cloud.
        Returns (centroid_3, axes_3x3) or (None, None).
        Columns of axes are the principal directions; sign-consistent via SVD."""
        pts = _get_pc_points(which)
        if pts is None or pts.shape[0] < 10:
            return None, None
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        axes = Vt.T  # columns = principal directions
        # Ensure right-handed frame
        if np.linalg.det(axes) < 0:
            axes[:, 2] *= -1
        return centroid.copy(), axes

    def _axes_to_euler_deg(axes):
        """Convert 3x3 rotation matrix to Euler angles (XYZ) in degrees."""
        R = axes
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        if sy > 1e-6:
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(-R[2, 0], sy)
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.arctan2(-R[2, 0], sy)
            rz = 0.0
        return np.degrees([rx, ry, rz])

    def _sign_consistent_axes(axes_new, axes_ref):
        """Flip PCA axis signs to be consistent with a reference frame.
        PCA axes can arbitrarily flip sign between frames; this fixes it."""
        for i in range(3):
            if np.dot(axes_new[:, i], axes_ref[:, i]) < 0:
                axes_new[:, i] *= -1
        if np.linalg.det(axes_new) < 0:
            axes_new[:, 2] *= -1
        return axes_new

    _input = carb.input.acquire_input_interface()
    _keyboard = omni.appwindow.get_default_app_window().get_keyboard()

    def _on_keyboard_event(event):
        nonlocal probe_pos, probe_euler_deg, line_depth, line_pos, line_rot
        nonlocal bone_opacity
        nonlocal lock_to_bone, lock_bone, lock_offset_local, lock_euler_local, prev_centroid, prev_axes

        if event.type == carb.input.KeyboardEventType.KEY_PRESS or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
            step = 0.01
            deg_step = 2.0
            if event.input == carb.input.KeyboardInput.UP:
                probe_pos[0] += step
            if event.input == carb.input.KeyboardInput.DOWN:
                probe_pos[0] -= step
            if event.input == carb.input.KeyboardInput.LEFT:
                probe_pos[1] += step
            if event.input == carb.input.KeyboardInput.RIGHT:
                probe_pos[1] -= step
            if event.input == carb.input.KeyboardInput.PAGE_UP:
                probe_pos[2] += step
            if event.input == carb.input.KeyboardInput.PAGE_DOWN:
                probe_pos[2] -= step
            if event.input == carb.input.KeyboardInput.Y:
                probe_euler_deg[0] += deg_step
            if event.input == carb.input.KeyboardInput.H:
                probe_euler_deg[0] -= deg_step
            if event.input == carb.input.KeyboardInput.N:
                probe_euler_deg[1] += deg_step
            if event.input == carb.input.KeyboardInput.M:
                probe_euler_deg[1] -= deg_step
            if event.input == carb.input.KeyboardInput.LEFT_BRACKET:
                line_depth = max(0.01, line_depth - step)
                print(f"[PLAN] Line length: {line_depth * 100:.1f} cm")
            if event.input == carb.input.KeyboardInput.RIGHT_BRACKET:
                line_depth += step
                print(f"[PLAN] Line length: {line_depth * 100:.1f} cm")
            if event.input == carb.input.KeyboardInput.COMMA:
                bone_opacity = max(0.0, bone_opacity - 0.05)
            if event.input == carb.input.KeyboardInput.PERIOD:
                bone_opacity = min(1.0, bone_opacity + 0.05)
            if event.input == carb.input.KeyboardInput.P:
                line_pos = list(probe_pos)
                line_rot = list(probe_euler_deg)
                print("[PLAN] Saved guide line (keyboard).")
            if event.input == carb.input.KeyboardInput.K:
                lock_to_bone = not lock_to_bone
                if lock_to_bone:
                    centroid, axes = _pc_pose(lock_bone)
                    if centroid is not None and axes is not None:
                        pos_np = np.array(probe_pos)
                        offset_world = pos_np - centroid
                        lock_offset_local = axes.T @ offset_world
                        bone_euler = _axes_to_euler_deg(axes)
                        lock_euler_local = np.array(probe_euler_deg) - bone_euler
                        prev_centroid = centroid
                        prev_axes = axes.copy()
                        print(f"[PLAN] 6-DOF locked to {lock_bone}")
                    else:
                        lock_to_bone = False
                        print(f"[PLAN] Cannot lock — no {lock_bone} point cloud data yet")
                else:
                    lock_offset_local = None
                    lock_euler_local = None
                    prev_centroid = None
                    prev_axes = None
                    print("[PLAN] Bone lock OFF — probe free")
            if event.input == carb.input.KeyboardInput.ONE:
                lock_bone = "femur"
                lock_offset_local = None
                prev_centroid = None
                prev_axes = None
                if lock_to_bone:
                    lock_to_bone = False
                print("[PLAN] Lock target: femur (press K to lock)")
            if event.input == carb.input.KeyboardInput.TWO:
                lock_bone = "tibia"
                lock_offset_local = None
                prev_centroid = None
                prev_axes = None
                if lock_to_bone:
                    lock_to_bone = False
                print("[PLAN] Lock target: tibia (press K to lock)")

    if args_cli.keyboard_fallback:
        _input.subscribe_to_keyboard_events(_keyboard, _on_keyboard_event)
        print("[INFO] Keyboard fallback ON.")

    def get_matrix(pos, euler_deg):
        m = Gf.Matrix4d()
        rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), euler_deg[2]) * Gf.Rotation(Gf.Vec3d(0, 1, 0), euler_deg[1]) * Gf.Rotation(
            Gf.Vec3d(1, 0, 0), euler_deg[0]
        )
        m.SetRotateOnly(rot)
        m.SetTranslateOnly(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        return m

    env.reset()

    # ── Seed arm from the first /lbr/joint_states message (ROS-matched init) ──
    # Eliminates the home-pose "slam" at startup: the arm is placed at whatever
    # the bag / FRI driver is currently publishing, same as what RViz would show.
    if ros_node is not None and rclpy_mod is not None:
        print("[INFO] Waiting for first /lbr/joint_states to seed arm init pose …")
        _seed_deadline = time.time() + 10.0   # max 10 s wait
        while (ros_node.joint_positions is None
                and simulation_app.is_running()
                and time.time() < _seed_deadline):
            rclpy_mod.spin_once(ros_node, timeout_sec=0.1)
        if ros_node.joint_positions is not None:
            _ros_names   = ros_node.joint_names
            _isaac_names = [env.robot.data.joint_names[i] for i in joint_ids]
            try:
                _reorder  = [_ros_names.index(n) for n in _isaac_names]
                _init_arm = ros_node.joint_positions[_reorder].to(env.device)
                _full_pos = env.robot.data.joint_pos.clone()
                _full_vel = torch.zeros_like(env.robot.data.joint_vel)
                for _k, _j in enumerate(joint_ids):
                    _full_pos[0, _j] = _init_arm[_k]
                env.robot.write_joint_state_to_sim(_full_pos, _full_vel)
                env.robot.set_joint_position_target(_full_pos)
                print(f"[INFO] Arm seeded from ROS: "
                      + ", ".join(f"{n}={v:+.3f}" for n, v in zip(_isaac_names, _init_arm.cpu().tolist())))
            except ValueError as _vmap:
                print(f"[WARN] Could not map ROS joint names to Isaac: {_vmap}. Using default init pose.")
        else:
            print("[WARN] No /lbr/joint_states received in 10 s — using default init pose.")

        # Confirm the seed landed in sim state before the main loop starts.
        _post_seed = env.robot.data.joint_pos[0, joint_ids].detach().cpu().tolist()
        print("[SEED-CHECK] sim joint_pos =", ["%+.3f" % v for v in _post_seed])

    stage = omni.usd.get_context().get_stage()
    probe_prim, line_prim, stylus_prim = _setup_surgical_plan_prims(stage)
    print(f"[INFO] Surgical pointer: {probe_prim_path} valid={probe_prim.IsValid()}  guide: {line_prim_path} valid={line_prim.IsValid()}")
    print(f"[INFO] StylusTracker: {stylus_prim_path} valid={stylus_prim.IsValid()}")

    # ── Bone ICP tracker ────────────────────────────────────────────────
    # Bone USD meshes are spawned from env_cfg (visible from start, in USDZ).
    # When ROS point cloud data arrives, we extract mesh vertices from the
    # bone prim, run ICP (mesh verts → point cloud), and apply the resulting
    # transform to move the bone to match the tracked position.
    _bone_ref_pcd: dict = {}   # bone_name → open3d.PointCloud (first PC = reference)
    _bone_icp_count: dict = {"femur": 0, "tibia": 0}
    _bone_mesh_verts: dict = {}  # bone_name → Nx3 numpy (mesh vertices in initial world frame)
    _BONE_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".bone_cache")

    # ── Apple Vision Pro AR streaming (avp_stream) ────────────────────────
    avp_streamer = None
    if args_cli.avp:
        try:
            from avp_stream import VisionProStreamer
            # Clear stale USDZ cache to guarantee fresh export with bone meshes
            import shutil as _shutil_avp
            _uc = "/tmp/isaac_usdz_cache"
            if os.path.isdir(_uc):
                _shutil_avp.rmtree(_uc)
                print("[AVP] Cleared stale USDZ cache → fresh export with bone meshes.")
            avp_streamer = VisionProStreamer(ip=args_cli.avp, origin="sim")
            avp_streamer.configure_isaac(
                scene=env.scene,
                relative_to=[0.0, 0.0, args_cli.avp_z, args_cli.avp_yaw],
                include_ground=False,
                env_indices=[0],
                # force_reload=True sets metadata.forceReload so the AVP side
                # ignores its on-device USDZ cache (cache_key is hashed from
                # scene dataclass only; it can't detect edits to med7.urdf).
                force_reload=True,
            )
            avp_streamer.start_webrtc()
            _avp_ip_display = args_cli.avp
            print("=" * 68)
            print("[AVP] Apple Vision Pro AR streaming ready.")
            print(f"[AVP]   Workstation IP : {_avp_ip_display}")
            print(f"[AVP]   Robot base z   : {args_cli.avp_z:.3f} m above real floor")
            print("[AVP]   If arm floats above table → decrease --avp-z (e.g. --avp-z 0.75)")
            print("[AVP]   If arm sinks into table  → increase --avp-z (e.g. --avp-z 0.90)")
            print("[AVP]   Streamed       : robot arm + surgical pointer + bone meshes")
            print("[AVP]   Ground/bed     : excluded (real environment used)")
            print("[AVP]   Hand tracking  : Vision Pro ARKit (sim frame)")
            print("[AVP]   Bones          : Poisson mesh (native USD Mesh, no re-export needed)")
            print("[AVP] ► On Vision Pro: open 'Tracking Streamer', enter workstation IP.")
            print("[AVP]   If stuck on 'Waiting', check: same Wi-Fi, port 8765 open.")
            print("=" * 68)
        except ImportError:
            print(
                "[AVP] ERROR: avp_stream not installed.\n"
                "       Run:  pip install --upgrade avp_stream\n"
                "       Then retry with --avp."
            )
            avp_streamer = None
        except Exception as e:
            print(f"[AVP] ERROR during setup: {e}")
            avp_streamer = None

    # ── UDP RTT probe (latency measurement to Vision Pro) ──────────────
    rtt_probe = None
    rtt_pub = None
    if args_cli.avp is not None:
        try:
            rtt_probe = UdpRttProbe(args_cli.avp)
            print(f"[RTT] UDP probe started → {args_cli.avp}:9998")
        except Exception as e:
            print(f"[RTT] Failed to start probe: {e}")
    if rtt_probe is not None and ros_node is not None:
        from std_msgs.msg import Float32
        rtt_pub = ros_node.create_publisher(Float32, "/avp/rtt_ms", 10)
        print("[RTT] Publishing RTT on /avp/rtt_ms (std_msgs/Float32)")

    def on_xr_reset():
        nonlocal probe_prim, line_prim, stylus_prim
        env.reset()
        st2 = omni.usd.get_context().get_stage()
        probe_prim, line_prim, stylus_prim = _setup_surgical_plan_prims(st2)
        print("[INFO] XR Reset: env + surgical prims re-created.")

    if teleop_interface is not None:
        teleop_interface.add_callback("RESET", on_xr_reset)

    print("-" * 72)
    print("LEFT single tap: lock probe → lock line length")
    print("LEFT double tap: cycle bone lock (femur → tibia → off)")
    print("LEFT hold 3 s:   unlock all")
    print("RIGHT wrist: move/rotate probe | RIGHT pinch spread: line length")
    print("Keyboard: K = bone lock, 1 = femur, 2 = tibia")
    print("-" * 72)

    step_count = 0
    _last_pc_update = 0.0
    _PC_UPDATE_INTERVAL = 0.05    # 20 Hz — matches /tf publish rate

    # --- Rendered-bone pose recorder (enabled via --log-render-bones) ---
    _render_log_dur   = float(args_cli.log_render_bones)
    _render_log_rows: list = []
    _render_log_start = None
    _render_log_done  = False
    _render_log_out_path = str(
        pathlib.Path(__file__).resolve().parent / "rendered_bones.csv"
    )
    if _render_log_dur > 0:
        print(f"[LOG-RENDER] Recording femur/tibia USD poses for "
              f"{_render_log_dur:.1f}s → {_render_log_out_path}")

    _avp_femur_seen = False
    _avp_tibia_seen = False

    # AVP hand-data throttle: poll get_latest() at ~2 Hz until Vision Pro connects,
    # then switch to every frame.  Avoids flooding the terminal with "No data" spam.
    _avp_connected = False          # flips True once we get real joint data
    _avp_last_poll = 0.0            # wall-clock time of last get_latest() call
    _AVP_POLL_SLOW = 0.5            # seconds between polls while waiting (2 Hz)
    _avp_cached_rh: dict = {}       # last known right-hand dict
    _avp_cached_lh: dict = {}       # last known left-hand dict
    try:
        while simulation_app.is_running():
            step_count += 1
            now = time.time()

            # Per-frame diagnostics for --log-render-bones. Reset each iteration;
            # populated by the ROS bone-write blocks below.
            _diag_flag_fem  = False
            _diag_flag_tib  = False
            _diag_wrote_fem = False
            _diag_wrote_tib = False
            _diag_wrote_fem_vec = [float("nan")] * 7
            _diag_wrote_tib_vec = [float("nan")] * 7

            # Per-TF probing: send a probe only when we actually write a bone
            # TF (below, inside each write block). Here we just drain the
            # rolling-10-ack stats the receiver thread has aggregated and
            # publish them live.
            if rtt_probe is not None and rtt_pub is not None:
                _stats = rtt_probe.poll_batch_avg()
                if _stats is not None:
                    from std_msgs.msg import Float32 as _F32
                    rtt_pub.publish(_F32(data=float(_stats["avg_ms"])))
                    print(f"[RTT-AVG{_stats['n']}] "
                          f"avg={_stats['avg_ms']:.1f}ms  "
                          f"min={_stats['min_ms']:.1f}  max={_stats['max_ms']:.1f}  "
                          f"jitter={_stats['jitter_ms']:.1f}  lost={_stats['lost']}  "
                          f"size={_stats['payload_bytes']}B")

            if ros_node is not None and rclpy_mod is not None:
                # Drain all pending callbacks (spin_once handles only one).
                # Bumped from 10→200 so /tf bursts (robot links + bones + etc.)
                # drain completely each iteration instead of back-pressuring the
                # new deep-queue /tf subscription (depth=500).  timeout_sec=0.0
                # makes this cheap when the queue is empty — the loop exits on
                # the first idle spin.
                for _ in range(200):
                    rclpy_mod.spin_once(ros_node, timeout_sec=0.0)
                ros_node.lookup_bone_tfs()

                pc_due = (now - _last_pc_update) >= _PC_UPDATE_INTERVAL
                if pc_due and (ros_node.pc_updated or ros_node.tibia_pc_updated):
                    if ros_node.pc_updated and ros_node.pc_points is not None:
                        ros_node.pc_updated = False
                        if not _avp_femur_seen:
                            _avp_femur_seen = True
                    if ros_node.tibia_pc_updated and ros_node.tibia_pc_points is not None:
                        ros_node.tibia_pc_updated = False
                        if not _avp_tibia_seen:
                            _avp_tibia_seen = True
                    _last_pc_update = now

                # Capture flag state BEFORE the write consumes it (for CSV diagnostics).
                _diag_flag_fem = bool(
                    ros_node.femur_origin_updated and ros_node.femur_origin_pos is not None
                )
                _diag_flag_tib = bool(
                    ros_node.tibia_origin_updated and ros_node.tibia_origin_pos is not None
                )

                # WRITE ORDER TEST (2026-04-22): tibia FIRST, femur SECOND.
                # Tests whether intra-frame Fabric write-propagation ordering is the
                # cause (first write may not flush to the AVP XformCache read in time).
                # Revert by swapping these two blocks back (femur first, tibia second).
                if ros_node.tibia_origin_updated and ros_node.tibia_origin_pos is not None:
                    _tib_prim = stage.GetPrimAtPath("/World/envs/env_0/tibia")
                    if _tib_prim and _tib_prim.IsValid():
                        from pxr import Gf as _Gf
                        _q = ros_node.tibia_origin_quat
                        _p = ros_node.tibia_origin_pos
                        _rot = _Gf.Rotation(_Gf.Quaternion(float(_q[0]),
                                _Gf.Vec3d(float(_q[1]), float(_q[2]), float(_q[3]))))
                        _mat = _Gf.Matrix4d()
                        _mat.SetRotateOnly(_rot)
                        _mat.SetTranslateOnly(_Gf.Vec3d(float(_p[0]), float(_p[1]), float(_p[2])))
                        UsdGeom.Xformable(_tib_prim).MakeMatrixXform().Set(_mat)
                        _diag_wrote_tib = True
                        _diag_wrote_tib_vec = [
                            float(_p[0]), float(_p[1]), float(_p[2]),
                            float(_q[0]), float(_q[1]), float(_q[2]), float(_q[3]),
                        ]
                        if rtt_probe is not None:
                            rtt_probe.send_probe()
                    ros_node.tibia_origin_updated = False

                # Apply bone origin transforms (6-DOF from /tracked/*_origin)
                if ros_node.femur_origin_updated and ros_node.femur_origin_pos is not None:
                    _fem_prim = stage.GetPrimAtPath("/World/envs/env_0/femur")
                    if _fem_prim and _fem_prim.IsValid():
                        from pxr import Gf as _Gf
                        _q = ros_node.femur_origin_quat
                        _p = ros_node.femur_origin_pos
                        _rot = _Gf.Rotation(_Gf.Quaternion(float(_q[0]),
                                _Gf.Vec3d(float(_q[1]), float(_q[2]), float(_q[3]))))
                        _mat = _Gf.Matrix4d()
                        _mat.SetRotateOnly(_rot)
                        _mat.SetTranslateOnly(_Gf.Vec3d(float(_p[0]), float(_p[1]), float(_p[2])))
                        UsdGeom.Xformable(_fem_prim).MakeMatrixXform().Set(_mat)
                        _diag_wrote_fem = True
                        _diag_wrote_fem_vec = [
                            float(_p[0]), float(_p[1]), float(_p[2]),
                            float(_q[0]), float(_q[1]), float(_q[2]), float(_q[3]),
                        ]
                        if rtt_probe is not None:
                            rtt_probe.send_probe()
                    ros_node.femur_origin_updated = False

                # Force arm to track ROS joint positions every frame (teleport).
                # This bypasses PD transient / joint-limit stickiness / collision
                # response and pins the articulation to whatever /lbr/joint_states
                # publishes.  The PD target below is also set to the same values,
                # so PD does not fight the teleport between frames.
                if ros_node.robot_state_updated and ros_node.joint_positions is not None:
                    try:
                        env.sync_robot_joint_state(ros_node.joint_names, ros_node.joint_positions)
                    except RuntimeError as e:
                        if "inference" not in str(e).lower():
                            raise
                    ros_node.robot_state_updated = False

            # --- Rendered-bone recorder tick ---
            if _render_log_dur > 0 and not _render_log_done:
                _now_rec = time.time()
                if _render_log_start is None:
                    _render_log_start = _now_rec
                _elapsed = _now_rec - _render_log_start
                if _elapsed < _render_log_dur:
                    _xf_cache_rec = UsdGeom.XformCache()
                    _row = [_elapsed]
                    for _bp_path in ("/World/envs/env_0/femur", "/World/envs/env_0/tibia"):
                        _pp = stage.GetPrimAtPath(_bp_path)
                        if _pp and _pp.IsValid():
                            _m  = _xf_cache_rec.GetLocalToWorldTransform(_pp)
                            _t  = _m.ExtractTranslation()
                            _q  = _m.ExtractRotationQuat()
                            _iq = _q.GetImaginary()
                            _row += [float(_t[0]), float(_t[1]), float(_t[2]),
                                     float(_q.GetReal()),
                                     float(_iq[0]), float(_iq[1]), float(_iq[2])]
                        else:
                            _row += [float("nan")] * 7
                    # Append per-frame diagnostic counters: flag state before
                    # write, whether Set() executed, and the pose that was
                    # written (NaN if not written this frame).
                    _row += [
                        int(_diag_flag_fem), int(_diag_flag_tib),
                        int(_diag_wrote_fem), int(_diag_wrote_tib),
                    ]
                    _row += _diag_wrote_fem_vec
                    _row += _diag_wrote_tib_vec
                    _render_log_rows.append(_row)
                else:
                    try:
                        import csv as _csv
                        with open(_render_log_out_path, "w", newline="") as _fh:
                            _w = _csv.writer(_fh)
                            _w.writerow([
                                "t",
                                "fx", "fy", "fz", "fqw", "fqx", "fqy", "fqz",
                                "tx", "ty", "tz", "tqw", "tqx", "tqy", "tqz",
                                "flag_fem", "flag_tib",
                                "wrote_fem", "wrote_tib",
                                "wrote_fem_x", "wrote_fem_y", "wrote_fem_z",
                                "wrote_fem_qw", "wrote_fem_qx", "wrote_fem_qy", "wrote_fem_qz",
                                "wrote_tib_x", "wrote_tib_y", "wrote_tib_z",
                                "wrote_tib_qw", "wrote_tib_qx", "wrote_tib_qy", "wrote_tib_qz",
                            ])
                            _w.writerows(_render_log_rows)
                        _n_rows = len(_render_log_rows)
                        _fem_flag_frames   = sum(1 for r in _render_log_rows if r[15])
                        _tib_flag_frames   = sum(1 for r in _render_log_rows if r[16])
                        _fem_write_frames  = sum(1 for r in _render_log_rows if r[17])
                        _tib_write_frames  = sum(1 for r in _render_log_rows if r[18])
                        print(f"[LOG-RENDER] Wrote {_n_rows} rows → {_render_log_out_path}")
                        print(f"[LOG-RENDER] femur flag frames: {_fem_flag_frames}/{_n_rows}  "
                              f"femur write frames: {_fem_write_frames}/{_n_rows}")
                        print(f"[LOG-RENDER] tibia flag frames: {_tib_flag_frames}/{_n_rows}  "
                              f"tibia write frames: {_tib_write_frames}/{_n_rows}")
                    except Exception as _e:
                        print(f"[LOG-RENDER] CSV write failed: {_e}")
                    _render_log_done = True

            if teleop_interface is not None:
                delta_pose = teleop_interface.advance()
            else:
                delta_pose = torch.zeros(12, device=env.device)

            with torch.inference_mode():
                d = delta_pose.to(env.device).flatten()
                if d.numel() < 12:
                    d_r = d[:6] if d.numel() >= 6 else torch.zeros(6, device=env.device)
                    d_l = torch.zeros(6, device=env.device)
                else:
                    d_r = d[:6]
                    d_l = d[6:12]

                # ── Hand tracking source: AVP or OpenXR ───────────────────
                if avp_streamer is not None:
                    # Apple Vision Pro — throttle get_latest() until connected.
                    # avp_stream prints "No data received yet" on every call when
                    # Vision Pro isn't connected; calling it every frame (60 Hz)
                    # floods the terminal.  Poll at 2 Hz until we get real data,
                    # then switch to every frame.
                    _poll_interval = 0.0 if _avp_connected else _AVP_POLL_SLOW
                    if now - _avp_last_poll >= _poll_interval:
                        _avp_last_poll = now
                        try:
                            _avp_td = avp_streamer.get_latest()
                            _new_rh = _avp_hand_to_openxr_dict(getattr(_avp_td, "right", None))
                            _new_lh = _avp_hand_to_openxr_dict(getattr(_avp_td, "left", None))
                            # Consider connected once we receive any joint data
                            if _new_rh or _new_lh:
                                if not _avp_connected:
                                    print("[AVP] Vision Pro connected — hand tracking active.")
                                _avp_connected = True
                                _avp_cached_rh = _new_rh
                                _avp_cached_lh = _new_lh
                        except Exception:
                            pass
                    rh = _avp_cached_rh
                    lh = _avp_cached_lh
                else:
                    # OpenXR (CloudXR) — existing path
                    raw = {}
                    if teleop_interface is not None:
                        try:
                            raw = teleop_interface._get_raw_data()
                        except Exception:
                            pass
                    rh = raw.get(OpenXRDevice.TrackingTarget.HAND_RIGHT, {})
                    lh = raw.get(OpenXRDevice.TrackingTarget.HAND_LEFT, {})

                # Hand tracking disabled — Muse stylus drives the probe instead.
                rh = {}
                lh = {}

                pd_r = _pinch_dist(rh)
                pd_l = _pinch_dist(lh)
                pinch_l = pd_l is not None and pd_l < 0.042

                _update_hand_stick_figure(stage, "/World/envs/env_0/SurgicalPlan/HandStick/Right", rh, (0.2, 0.92, 1.0))
                _update_hand_stick_figure(stage, "/World/envs/env_0/SurgicalPlan/HandStick/Left", lh, (1.0, 0.55, 0.15))

                # ── Logitech Muse stylus → probe tip + buttons ──────────────────
                # When the AVP has a paired Muse stylus, use its 6-DOF pose to drive
                # the probe tip and its primary/secondary buttons to lock/unlock.
                # Primary press → LOCK probe.  Secondary press → UNLOCK probe.
                _stylus_active = False
                if avp_streamer is not None and hasattr(avp_streamer, "get_stylus"):
                    try:
                        _stylus = avp_streamer.get_stylus()
                    except Exception as _sty_ex:
                        _stylus = None
                        if step_count % 180 == 0:
                            print(f"[STYLUS] get_stylus() exception: {_sty_ex}")

                    def _sget(obj, *keys, default=None):
                        """Robust accessor for dict-style or object-style stylus data."""
                        if obj is None:
                            return default
                        for k in keys:
                            if isinstance(obj, dict):
                                if k in obj:
                                    return obj[k]
                            else:
                                v = getattr(obj, k, None)
                                if v is not None:
                                    return v
                        return default

                    if _stylus is not None:
                        # One-shot debug print to see the actual field layout
                        if not _stylus_debug_printed:
                            try:
                                if isinstance(_stylus, dict):
                                    print(f"[STYLUS] get_stylus() keys: {list(_stylus.keys())}")
                                else:
                                    _attrs = [a for a in dir(_stylus) if not a.startswith("_")]
                                    print(f"[STYLUS] get_stylus() attrs: {_attrs}")
                            except Exception:
                                pass
                            _stylus_debug_printed = True

                        # avp_stream.StreamerBridge.get_stylus() returns a dict
                        # whose bit-decoded `primary_pressed` / `secondary_pressed`
                        # booleans are corrupted by a client-side bug in the m33
                        # timestamp term (it sweeps [0,10) instead of [0,1), so
                        # the low bits toggle constantly).
                        #
                        # `primary_pressure` / `secondary_pressure` (matrix m31 /
                        # m32) however are clean digital 0.0 / 1.0 signals — they
                        # are effectively the real button flags. We thresholding
                        # those and ignore `*_pressed` and m33 entirely.
                        _s_pose      = _sget(_stylus, "pose", "Pose", "transform")
                        _s_tracked   = True   # dict is only returned when tracked
                        _s_primary   = float(_sget(_stylus, "primary_pressure",   default=0.0)) > 0.5
                        _s_secondary = float(_sget(_stylus, "secondary_pressure", default=0.0)) > 0.5

                        if _s_pose is not None:
                            _s_pose = np.asarray(_s_pose, dtype=np.float64)

                        if _s_pose is not None and _s_pose.shape == (4, 4) and _s_tracked:
                            _stylus_active = True

                            # ── EKF smoothing: feed raw pose, use filtered output. ──
                            _s_pose_raw = _s_pose
                            _s_pose_filt, _ekf_ok = stylus_ekf.process(_s_pose_raw, True, now)
                            if _ekf_ok:
                                _s_pose = np.asarray(_s_pose_filt, dtype=np.float64)
                            _stylus_pose_world = _s_pose

                            # Update StylusTracker Xform (visual) from marker pose
                            if stylus_prim is not None and stylus_prim.IsValid():
                                from pxr import Gf as _GfS
                                from scipy.spatial.transform import Rotation as _RotS
                                _q_xyzw = _RotS.from_matrix(_s_pose[:3, :3]).as_quat()
                                _gf_rot_s = _GfS.Rotation(
                                    _GfS.Quaternion(float(_q_xyzw[3]),
                                        _GfS.Vec3d(float(_q_xyzw[0]),
                                                   float(_q_xyzw[1]),
                                                   float(_q_xyzw[2]))))
                                _gf_mat_s = _GfS.Matrix4d()
                                _gf_mat_s.SetRotateOnly(_gf_rot_s)
                                _gf_mat_s.SetTranslateOnly(_GfS.Vec3d(
                                    float(_s_pose[0, 3]),
                                    float(_s_pose[1, 3]),
                                    float(_s_pose[2, 3])))
                                UsdGeom.Xformable(stylus_prim).MakeMatrixXform().Set(_gf_mat_s)
                                UsdGeom.Imageable(stylus_prim).MakeVisible()

                            # Drive probe tip + orientation from stylus (only while unlocked)
                            if not probe_pos_locked:
                                _tip_local = np.array([0.0, 0.0, -STYLUS_TIP_OFFSET, 1.0])
                                _tip_world = _s_pose @ _tip_local
                                probe_pos[0] = float(_tip_world[0])
                                probe_pos[1] = float(_tip_world[1])
                                probe_pos[2] = float(_tip_world[2])
                                from scipy.spatial.transform import Rotation as _RotE
                                _eul = _RotE.from_matrix(_s_pose[:3, :3]).as_euler("xyz", degrees=True)
                                probe_euler_deg[0] = float(_eul[0])
                                probe_euler_deg[1] = float(_eul[1])
                                probe_euler_deg[2] = float(_eul[2])

                            # Hold-to-trigger: track when each button first went True.
                            # Reset on any release. Fires once per hold after
                            # STYLUS_BTN_HOLD_S continuous seconds.
                            if _s_primary:
                                if _prim_true_since is None:
                                    _prim_true_since = now
                            else:
                                _prim_true_since = None
                                _prim_triggered  = False

                            if _s_secondary:
                                if _sec_true_since is None:
                                    _sec_true_since = now
                            else:
                                _sec_true_since = None
                                _sec_triggered  = False

                            _prim_held_s = (now - _prim_true_since) if _prim_true_since else 0.0
                            _sec_held_s  = (now - _sec_true_since)  if _sec_true_since  else 0.0

                            # Log every transition so we can see hold progress.
                            if _s_primary != prev_primary_btn or _s_secondary != prev_secondary_btn:
                                print(f"[STYLUS] raw primary={_s_primary} "
                                      f"secondary={_s_secondary} "
                                      f"held=({_prim_held_s:.2f}s,{_sec_held_s:.2f}s) "
                                      f"locked={probe_pos_locked}")

                            # Primary held ≥ 2 s → LOCK (once per hold).
                            if (_s_primary and not _prim_triggered
                                    and _prim_held_s >= STYLUS_BTN_HOLD_S
                                    and not probe_pos_locked):
                                probe_pos_locked = True
                                _locked_pos = np.array(probe_pos, dtype=np.float64)
                                _prim_triggered = True
                                _attach_probe_to_nearest_bone(_locked_pos)
                                print(f"[STYLUS] ✓ Probe LOCKED "
                                      f"(primary held {_prim_held_s:.2f}s)")

                            # Secondary held ≥ 2 s → UNLOCK (once per hold).
                            if (_s_secondary and not _sec_triggered
                                    and _sec_held_s >= STYLUS_BTN_HOLD_S
                                    and probe_pos_locked):
                                probe_pos_locked = False
                                line_len_locked  = False
                                line_pos         = None
                                line_rot         = None
                                _dwell_start     = now
                                _dwell_ref_pos   = np.array(probe_pos, dtype=np.float64)
                                _sec_triggered   = True
                                _detach_probe_from_bone()
                                print(f"[STYLUS] ✓ Probe UNLOCKED "
                                      f"(secondary held {_sec_held_s:.2f}s)")

                            prev_primary_btn   = _s_primary
                            prev_secondary_btn = _s_secondary

                if (not _stylus_active
                        and stylus_prim is not None and stylus_prim.IsValid()):
                    UsdGeom.Imageable(stylus_prim).MakeInvisible()

                # ── Dwell-based lock / unlock (no gestures needed) ───────────────
                # Skipped when the Muse stylus is active — the primary/secondary
                # buttons are the lock mechanism in that case.
                if _stylus_active:
                    # Keep dwell reference in sync so it doesn't trigger spuriously
                    # right after a stylus-driven unlock.
                    _dwell_ref_pos = np.array(probe_pos, dtype=np.float64)
                    _dwell_start   = now
                    _dwell_printed = -1.0
                else:
                    # LOCK  : hold probe still (< DWELL_LOCK_MM mm) for DWELL_LOCK_S seconds
                    # UNLOCK: return to locked position (< DWELL_UNLOCK_NEAR_MM mm) and
                    #         hold still again for DWELL_LOCK_S seconds
                    DWELL_LOCK_S        = 2.0    # seconds of stillness to trigger
                    DWELL_LOCK_MM       = 5.0    # mm — "still enough" threshold
                    DWELL_UNLOCK_NEAR_MM = 30.0  # mm — how close to locked point to start unlock dwell

                    _cur_pos = np.array(probe_pos, dtype=np.float64)

                    if not probe_pos_locked:
                        # ── Moving freely — watch for stillness to LOCK ──────────
                        _moved = np.linalg.norm(_cur_pos - _dwell_ref_pos) * 1000.0  # mm
                        if _moved > DWELL_LOCK_MM:
                            _dwell_ref_pos = _cur_pos.copy()
                            _dwell_start   = now
                            _dwell_printed = -1.0
                        else:
                            elapsed = now - _dwell_start
                            bucket = int(elapsed / 0.5) * 0.5
                            if bucket > _dwell_printed and elapsed < DWELL_LOCK_S:
                                _dwell_printed = bucket
                                remaining = DWELL_LOCK_S - elapsed
                                print(f"[PLAN] Dwell locking in {remaining:.1f}s …")
                            if elapsed >= DWELL_LOCK_S:
                                probe_pos_locked = True
                                _locked_pos      = _cur_pos.copy()
                                _unlock_dwell_start = now + 1e9
                                _dwell_printed   = -1.0
                                _attach_probe_to_nearest_bone(_locked_pos)
                                print("[PLAN] ✓ Probe LOCKED (2 s still).")
                    else:
                        # ── Locked — watch for return-and-hold to UNLOCK ─────────
                        _dist_to_lock = np.linalg.norm(_cur_pos - _locked_pos) * 1000.0  # mm
                        if _dist_to_lock < DWELL_UNLOCK_NEAR_MM:
                            if _unlock_dwell_start > now:
                                _unlock_dwell_start = now
                                _unlock_ref_pos     = _cur_pos.copy()
                                _dwell_printed      = -1.0
                            _moved_since_near = np.linalg.norm(_cur_pos - _unlock_ref_pos) * 1000.0
                            if _moved_since_near > DWELL_LOCK_MM:
                                _unlock_dwell_start = now
                                _unlock_ref_pos     = _cur_pos.copy()
                                _dwell_printed      = -1.0
                            else:
                                elapsed = now - _unlock_dwell_start
                                bucket  = int(elapsed / 0.5) * 0.5
                                if bucket > _dwell_printed and elapsed < DWELL_LOCK_S:
                                    _dwell_printed = bucket
                                    remaining = DWELL_LOCK_S - elapsed
                                    print(f"[PLAN] Dwell unlocking in {remaining:.1f}s …")
                                if elapsed >= DWELL_LOCK_S:
                                    probe_pos_locked    = False
                                    line_len_locked     = False
                                    line_pos            = None
                                    line_rot            = None
                                    _dwell_start        = now
                                    _dwell_ref_pos      = _cur_pos.copy()
                                    _unlock_dwell_start = now + 1e9
                                    _dwell_printed      = -1.0
                                    _detach_probe_from_bone()
                                    print("[PLAN] ✓ Probe UNLOCKED (held near locked point 2 s).")
                        else:
                            _unlock_dwell_start = now + 1e9
                            _dwell_printed      = -1.0

                can_move_probe = not probe_pos_locked

                # Bone-follow: if probe is locked and auto-attached to a bone, recompute
                # probe world pose from current bone pose + stored local offset.
                if probe_pos_locked and _auto_bone_name is not None and _auto_bone_off is not None:
                    _bp_w, _bq_w = _read_bone_world_pose(_auto_bone_name)
                    if _bp_w is not None:
                        _new_p = _bp_w + _qrot(_bq_w, _auto_bone_off)
                        _new_q = _qmul(_bq_w, _auto_bone_qoff)
                        probe_pos[0] = float(_new_p[0])
                        probe_pos[1] = float(_new_p[1])
                        probe_pos[2] = float(_new_p[2])
                        _e = _quat_to_euler_xyz_deg(_new_q)
                        probe_euler_deg[0] = _e[0]
                        probe_euler_deg[1] = _e[1]
                        probe_euler_deg[2] = _e[2]

                idx_tip     = _joint_xyz(rh, "index_tip")
                idx_distal  = _joint_xyz(rh, "index_distal")

                # Position: 2 cm beyond the index tip along the finger direction.
                # Skipped when the Muse stylus is active — stylus already wrote
                # probe_pos above and must not be overwritten.
                if can_move_probe and idx_tip is not None and not _stylus_active:
                    if idx_distal is not None:
                        finger_dir = idx_tip - idx_distal
                        n = float(np.linalg.norm(finger_dir))
                        finger_dir = finger_dir / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])
                    else:
                        finger_dir = np.array([0.0, 1.0, 0.0])
                    offset_tip = idx_tip + 0.02 * finger_dir  # 2 cm past tip
                    probe_pos[0] = float(offset_tip[0])
                    probe_pos[1] = float(offset_tip[1])
                    probe_pos[2] = float(offset_tip[2])

                # --- Angle tracking commented out: probe stays at fixed orientation ---
                # import math as _m
                # can_move_angle = not probe_angle_locked
                # idx_base = _joint_xyz(rh, "index_proximal")
                # if can_move_angle and idx_tip is not None and idx_base is not None:
                #     d = idx_tip - idx_base
                #     length = float(np.linalg.norm(d))
                #     if length > 1e-6:
                #         d = d / length
                #         dx, dy, dz = float(d[0]), float(d[1]), float(d[2])
                #         h = _m.sqrt(dx * dx + dy * dy)
                #         probe_euler_deg[0] = _m.degrees(_m.atan2(dz, h)) if h > 1e-6 else (90.0 if dz > 0 else -90.0)
                #         probe_euler_deg[1] = probe_roll_deg
                #         probe_euler_deg[2] = _m.degrees(_m.atan2(-dx, dy)) if h > 1e-6 else 0.0
                #
                # # Wrist twist → roll
                # wt_r = rh.get("wrist")
                # if wt_r is not None and len(wt_r) >= 7:
                #     qw, qx, qy, qz = float(wt_r[3]), float(wt_r[4]), float(wt_r[5]), float(wt_r[6])
                #     twist = _m.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
                #     twist_deg = _m.degrees(twist)
                #     if prev_wrist_twist is not None and can_move_angle:
                #         delta = twist_deg - prev_wrist_twist
                #         if delta > 180.0:
                #             delta -= 360.0
                #         elif delta < -180.0:
                #             delta += 360.0
                #         probe_roll_deg += delta * 0.5 * sens
                #     prev_wrist_twist = twist_deg
                # else:
                #     prev_wrist_twist = None

                if teleoperation_active and probe_pos_locked and not line_len_locked and pd_r is not None:
                    line_depth = float(np.clip(0.02 + pd_r * 2.5, 0.01, 0.45))

                spread = _spread_thumb_pinky(lh)
                if teleoperation_active and spread is not None:
                    tgt = float(np.clip((spread - 0.05) / 0.14, 0.08, 1.0))
                    bone_opacity = 0.85 * bone_opacity + 0.15 * tgt

                wt = rh.get("wrist")
                if wt is not None and len(wt) >= 7:
                    hp = torch.zeros((env.num_envs, 7), device=env.device)
                    for i in range(7):
                        hp[0, i] = float(wt[i])
                    env.scene["hand_proxy"].write_root_pose_to_sim(hp)

                # Lock-to-bone: full 6-DOF tracking via PCA
                if lock_to_bone and lock_offset_local is not None:
                    centroid, axes = _pc_pose(lock_bone)
                    if centroid is not None and axes is not None:
                        if prev_axes is not None:
                            axes = _sign_consistent_axes(axes, prev_axes)
                        # Reconstruct probe pose in world frame
                        new_pos = centroid + axes @ lock_offset_local
                        bone_euler = _axes_to_euler_deg(axes)
                        new_euler = bone_euler + lock_euler_local
                        probe_pos[0] = float(new_pos[0])
                        probe_pos[1] = float(new_pos[1])
                        probe_pos[2] = float(new_pos[2])
                        probe_euler_deg[0] = float(new_euler[0])
                        probe_euler_deg[1] = float(new_euler[1])
                        probe_euler_deg[2] = float(new_euler[2])
                        prev_centroid = centroid
                        prev_axes = axes.copy()

                # Move probe via RigidObject API so avp_stream tracks it
                import math as _m_probe
                _pr = _m_probe.radians(probe_euler_deg[0]) / 2
                _pp = _m_probe.radians(probe_euler_deg[1]) / 2
                _py = _m_probe.radians(probe_euler_deg[2]) / 2
                _cx, _sx = _m_probe.cos(_pr), _m_probe.sin(_pr)
                _cy, _sy = _m_probe.cos(_pp), _m_probe.sin(_pp)
                _cz, _sz = _m_probe.cos(_py), _m_probe.sin(_py)
                _qw = _cx * _cy * _cz + _sx * _sy * _sz
                _qx = _sx * _cy * _cz - _cx * _sy * _sz
                _qy = _cx * _sy * _cz + _sx * _cy * _sz
                _qz = _cx * _cy * _sz - _sx * _sy * _cz
                _probe_pose = torch.zeros((env.num_envs, 7), device=env.device)
                _probe_pose[0, 0] = float(probe_pos[0])
                _probe_pose[0, 1] = float(probe_pos[1])
                _probe_pose[0, 2] = float(probe_pos[2])
                _probe_pose[0, 3] = _qw
                _probe_pose[0, 4] = _qx
                _probe_pose[0, 5] = _qy
                _probe_pose[0, 6] = _qz
                env.scene["probe"].write_root_pose_to_sim(_probe_pose)
                if line_prim is not None and line_prim.IsValid():
                    if probe_pos_locked:
                        # Show guide line only after probe is locked
                        target_pos = line_pos if line_pos is not None else probe_pos
                        target_rot = line_rot if line_rot is not None else probe_euler_deg
                        UsdGeom.Xformable(line_prim).MakeMatrixXform().Set(get_matrix(target_pos, target_rot))
                        attr_height = line_prim.GetAttribute("height")
                        if attr_height.IsValid() and abs(attr_height.Get() - line_depth) > 0.001:
                            attr_height.Set(line_depth)
                        UsdGeom.Imageable(line_prim).MakeVisible()
                    else:
                        UsdGeom.Imageable(line_prim).MakeInvisible()

                if probe_pos_locked and probe_pose_pub is not None and step_count % 12 == 0:
                    from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Point, Quaternion

                    stamp = ros_node.get_clock().now().to_msg()

                    # ── Probe tip orientation ──────────────────────────────────────
                    # Constraints:
                    #   1. local +Z points straight DOWN in world frame.
                    #   2. additional 90° clockwise yaw when viewed top-down
                    #      (i.e. -90° about world +Z axis).
                    # Compose R = R_z(-90°) · R_x(180°). Resulting quaternion
                    # (x, y, z, w) = (√2/2, -√2/2, 0, 0).
                    _s2 = float(np.sqrt(2.0) / 2.0)
                    qx, qy, qz, qw = _s2, -_s2, 0.0, 0.0
                    _approach_vec = np.array([0.0, 0.0, -1.0])

                    probe_msg = PoseStamped()
                    probe_msg.header.frame_id = "lbr_link_0"
                    probe_msg.header.stamp = stamp
                    probe_msg.pose.position  = Point(
                        x=float(probe_pos[0]), y=float(probe_pos[1]), z=float(probe_pos[2]))
                    probe_msg.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
                    probe_pose_pub.publish(probe_msg)

                    # ── Guide-line: top = probe_pos + approach × line_depth ────────
                    lp = list(probe_pos)
                    line_top = [
                        lp[0] + float(_approach_vec[0]) * line_depth,
                        lp[1] + float(_approach_vec[1]) * line_depth,
                        lp[2] + float(_approach_vec[2]) * line_depth,
                    ]
                    orient = Quaternion(x=qx, y=qy, z=qz, w=qw)
                    line_msg = PoseArray()
                    line_msg.header.frame_id = "lbr_link_0"
                    line_msg.header.stamp = stamp
                    line_msg.poses = [
                        Pose(position=Point(
                            x=float(line_top[0]), y=float(line_top[1]), z=float(line_top[2])),
                            orientation=orient),
                        Pose(position=Point(
                            x=float(lp[0]), y=float(lp[1]), z=float(lp[2])),
                            orientation=orient),
                    ]
                    guide_line_pub.publish(line_msg)

                if ros_node is not None and ros_node.joint_positions is not None:
                    # Reorder ROS joints to match Isaac Sim joint order
                    ros_names = ros_node.joint_names
                    isaac_names = [env.robot.data.joint_names[i] for i in joint_ids]
                    reorder = [ros_names.index(n) for n in isaac_names]
                    joint_pos_des = ros_node.joint_positions[reorder].unsqueeze(0)
                else:
                    joint_pos_des = env.robot.data.joint_pos[:, joint_ids].clone()

                # ── DIAG: compare ROS-commanded vs Isaac-observed joint angles ──
                if ros_node is not None and ros_node.joint_positions is not None and step_count % 30 == 0:
                    _cmd = joint_pos_des[0].detach().cpu().numpy()
                    _cur = env.robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()
                    _names_short = [n.replace("lbr_", "") for n in isaac_names]
                    print("[DIAG] joint :", "  ".join(f"{n:>3}" for n in _names_short))
                    print("[DIAG] CMD   :", "  ".join(f"{x:+.3f}" for x in _cmd))
                    print("[DIAG] SIM   :", "  ".join(f"{x:+.3f}" for x in _cur))
                    print("[DIAG] DIFF  :", "  ".join(f"{(c-s):+.3f}" for c, s in zip(_cmd, _cur)))

                try:
                    env.step(joint_pos_des)
                except Exception as e:
                    err = str(e).lower()
                    if any(s in err for s in ("dof veloc", "dof_vel", "failed to get dof", "from backend", "physx")):
                        print("[INFO] Simulation step stopped. Exiting teleop loop.")
                        break

                # ── Stream pose updates to Vision Pro AR (every frame) ──
                if avp_streamer is not None:
                    try:
                        avp_streamer.update_sim()
                        # Inject non-physics prims (Probe, GuideLine, PointClouds)
                        # into the pose stream.  update_sim() only covers
                        # articulation / rigid-object bodies from Isaac Lab.
                        if avp_streamer._pose_stream_running:
                            # Debug: print known pose keys once to verify probe key matches
                            if step_count == 602:
                                try:
                                    _known = list((avp_streamer._current_poses or {}).get("poses", {}).keys())
                                    print(f"[AVP-DEBUG] Pose stream keys ({len(_known)}): {_known[:20]}")
                                except Exception:
                                    pass
                            _extra_poses: dict = {}
                            _xform_cache = UsdGeom.XformCache()
                            _stage_for_pc = omni.usd.get_context().get_stage()
                            _inject_prims = [
                                (line_prim,   "env_0/SurgicalPlan/GuideLine"),
                                (stylus_prim, "env_0/SurgicalPlan/StylusTracker"),
                            ]
                            for _bone_name in ("femur", "tibia"):
                                _bp = _stage_for_pc.GetPrimAtPath(
                                    f"/World/envs/env_0/{_bone_name}")
                                if _bp and _bp.IsValid():
                                    _inject_prims.append(
                                        (_bp, f"env_0/{_bone_name}"))

                            for _prim, _swift_key in _inject_prims:
                                if _prim is None or not _prim.IsValid():
                                    continue
                                try:
                                    _mat = _xform_cache.GetLocalToWorldTransform(_prim)
                                    _t   = _mat.ExtractTranslation()
                                    _r   = _mat.ExtractRotationQuat()
                                    _img = _r.GetImaginary()
                                    _extra_poses[_swift_key] = {
                                        "xpos":  [_t[0], _t[1], _t[2]],
                                        "xquat": [_r.GetReal(), _img[0], _img[1], _img[2]],
                                    }
                                except Exception:
                                    pass
                            if _extra_poses:
                                with avp_streamer._pose_stream_lock:
                                    if (avp_streamer._current_poses
                                            and "poses" in avp_streamer._current_poses):
                                        avp_streamer._current_poses["poses"].update(_extra_poses)
                    except Exception:
                        pass
            if step_count % 500 == 0:
                rh_wrist = rh.get("wrist") if rh else None
                print(
                    f"[STATUS] step={step_count} active={teleoperation_active} "
                    f"locked={probe_pos_locked} probe={[round(p,3) for p in probe_pos]} "
                    f"rh_track={'yes' if rh_wrist is not None else 'no'}"
                )

    finally:
        env.close()
        if rtt_probe is not None:
            rtt_probe.close()
        if avp_streamer is not None:
            try:
                avp_streamer.cleanup()
                print("[AVP] Streamer stopped.")
            except Exception:
                pass
        if ros_node is not None and rclpy_mod is not None:
            ros_node.destroy_node()
            rclpy_mod.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
