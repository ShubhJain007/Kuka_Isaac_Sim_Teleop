"""
Lightweight Kuka Med7 surgical visualization.

Mirrors robot arm from ROS joint states, tracks bone poses from /tf,
Muse stylus drives the surgical probe, streams to Apple Vision Pro.

No physics. No RL. No hand tracking. Just rendering.

Usage:
    python lite/run.py                                    # scene viewer only
    python lite/run.py --ros                              # + bone/robot from ROS
    python lite/run.py --ros --avp 192.168.1.42           # + Vision Pro AR
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

parser = argparse.ArgumentParser(description="Kuka Med7 surgical visualization (lite)")
parser.add_argument("--ros", action="store_true", help="Subscribe to /tf and /lbr/joint_states")
parser.add_argument("--avp", type=str, default=None, metavar="IP", help="Vision Pro IP for AR streaming")
parser.add_argument("--avp-z", type=float, default=0.825, metavar="M", help="Robot base height above floor")
parser.add_argument("--avp-yaw", type=float, default=0.0, metavar="DEG", help="Yaw offset vs Vision Pro")

if "--ros" in sys.argv:
    from ros_bridge import setup_ros2_libs
    setup_ros2_libs()

from isaacsim import SimulationApp

args = parser.parse_args()
simulation_app = SimulationApp({"headless": False})

import omni.usd
from pxr import UsdGeom

from scene import Med7Scene, SceneConfig
from probe import ProbeState


def main():
    stage = omni.usd.get_context().get_stage()
    scene = Med7Scene(SceneConfig())
    scene.setup(stage)

    # ── ROS ──────────────────────────────────────────────────────────────
    ros = None
    probe_pub = None
    guide_pub = None
    if args.ros:
        from ros_bridge import RosBridge
        ros = RosBridge()
        from geometry_msgs.msg import PoseStamped, PoseArray
        probe_pub = ros.create_publisher(PoseStamped, "/surgical_plan/probe_pose", 10)
        guide_pub = ros.create_publisher(PoseArray, "/surgical_plan/guide_line", 10)
        print("[ROS] Listening: /tf, /lbr/joint_states")

    # ── Stylus ───────────────────────────────────────────────────────────
    stylus_input = None
    if args.avp:
        try:
            from stylus import StylusInput
            stylus_input = StylusInput()
        except ImportError as e:
            print(f"[WARN] Stylus unavailable: {e}")

    # ── Probe ────────────────────────────────────────────────────────────
    probe = ProbeState()

    # ── AVP streaming ────────────────────────────────────────────────────
    avp = None
    if args.avp:
        try:
            from avp_stream import VisionProStreamer
            import shutil
            cache = "/tmp/isaac_usdz_cache"
            if os.path.isdir(cache):
                shutil.rmtree(cache)
            avp = VisionProStreamer(ip=args.avp, origin="sim")
            avp.configure_isaac(
                stage=stage,
                relative_to=[0.0, 0.0, args.avp_z, args.avp_yaw],
                include_ground=False,
                force_reload=True,
            )
            avp.start_webrtc()
            print(f"[AVP] Ready → {args.avp} (z={args.avp_z}m)")
        except ImportError:
            print("[AVP] pip install avp_stream")
        except Exception as e:
            print(f"[AVP] Error: {e}")

    print("=" * 50)
    print("Kuka Med7 Lite — rendering only, no physics")
    if stylus_input:
        print("  Primary hold 2s → LOCK | Secondary hold 2s → UNLOCK")
    print("=" * 50)

    # ── Main loop ────────────────────────────────────────────────────────
    step = 0
    while simulation_app.is_running():
        step += 1
        now = time.time()

        bone_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        # 1. ROS updates
        if ros is not None:
            ros.spin_once()

            if ros.femur_updated and ros.femur_pos is not None:
                scene.set_bone_pose("femur", ros.femur_pos, ros.femur_quat)
                bone_poses["femur"] = (ros.femur_pos, ros.femur_quat)
                ros.femur_updated = False

            if ros.tibia_updated and ros.tibia_pos is not None:
                scene.set_bone_pose("tibia", ros.tibia_pos, ros.tibia_quat)
                bone_poses["tibia"] = (ros.tibia_pos, ros.tibia_quat)
                ros.tibia_updated = False

            if ros.joint_updated and ros.joint_positions is not None:
                joint_dict = dict(zip(ros.joint_names, ros.joint_positions))
                scene.set_joint_positions(joint_dict)
                ros.joint_updated = False
        else:
            for bname in ("femur", "tibia"):
                bp, bq = _read_bone_pose(stage, bname)
                if bp is not None:
                    bone_poses[bname] = (bp, bq)

        # 2. Stylus → probe
        if avp is not None and stylus_input is not None:
            try:
                raw = avp.get_stylus()
            except Exception:
                raw = None

            stylus_input.update(raw, now)

            if stylus_input.active:
                scene.set_stylus_pose(stylus_input.pose_matrix, visible=True)
                if not probe.locked:
                    probe.set_pose(stylus_input.tip_pos, euler_deg=stylus_input.euler_deg)
                if stylus_input.lock_triggered and not probe.locked:
                    probe.lock(bone_poses)
                if stylus_input.unlock_triggered and probe.locked:
                    probe.unlock()
            else:
                scene.set_stylus_pose(np.eye(4), visible=False)

        # 3. Bone follow
        probe.follow_bone(bone_poses)

        # 4. Update visuals
        scene.set_probe_pose(probe.pos, probe.quat_wxyz)
        if probe.locked:
            lp = probe.line_pos if probe.line_pos is not None else probe.pos
            lq = probe.line_quat if probe.line_quat is not None else probe.quat_wxyz
            scene.set_guide_line(lp, lq, probe.line_depth, visible=True)
        else:
            scene.set_guide_line(probe.pos, probe.quat_wxyz, probe.line_depth, visible=False)

        # 5. ROS publish (~10 Hz)
        if probe.locked and probe_pub is not None and step % 6 == 0:
            _publish_plan(ros, probe_pub, guide_pub, probe)

        # 6. AVP stream
        if avp is not None and step % 2 == 0:
            try:
                avp.update_sim()
                _inject_poses(avp, stage)
            except Exception:
                pass

        # 7. Status
        if step % 500 == 0:
            print(f"[{step}] locked={probe.locked} pos={np.round(probe.pos, 3).tolist()}")

        simulation_app.update()

    # Cleanup
    if ros is not None:
        ros.destroy()
    if avp is not None:
        try:
            avp.cleanup()
        except Exception:
            pass


def _read_bone_pose(stage, bone_name: str):
    prim = stage.GetPrimAtPath(f"/World/Bones/{bone_name}")
    if not prim or not prim.IsValid():
        return None, None
    mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
    pos = np.array(mat.ExtractTranslation())
    rot = mat.ExtractRotation().GetQuaternion()
    im = rot.GetImaginary()
    return pos, np.array([float(rot.GetReal()), float(im[0]), float(im[1]), float(im[2])])


def _publish_plan(ros, probe_pub, guide_pub, probe: ProbeState):
    from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Point, Quaternion
    stamp = ros.get_clock().now().to_msg()
    s2 = float(np.sqrt(2.0) / 2.0)
    qx, qy, qz, qw = s2, -s2, 0.0, 0.0

    msg = PoseStamped()
    msg.header.frame_id = "lbr_link_0"
    msg.header.stamp = stamp
    msg.pose.position = Point(x=float(probe.pos[0]), y=float(probe.pos[1]), z=float(probe.pos[2]))
    msg.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
    probe_pub.publish(msg)

    top = probe.pos + np.array([0, 0, -1]) * probe.line_depth
    orient = Quaternion(x=qx, y=qy, z=qz, w=qw)
    line_msg = PoseArray()
    line_msg.header.frame_id = "lbr_link_0"
    line_msg.header.stamp = stamp
    line_msg.poses = [
        Pose(position=Point(x=float(top[0]), y=float(top[1]), z=float(top[2])), orientation=orient),
        Pose(position=Point(x=float(probe.pos[0]), y=float(probe.pos[1]), z=float(probe.pos[2])), orientation=orient),
    ]
    guide_pub.publish(line_msg)


def _inject_poses(avp, stage):
    if not getattr(avp, "_pose_stream_running", False):
        return
    xform_cache = UsdGeom.XformCache()
    extra = {}
    for path, key in [
        ("/World/SurgicalPlan/GuideLine", "GuideLine"),
        ("/World/SurgicalPlan/StylusTracker", "StylusTracker"),
        ("/World/Bones/femur", "femur"),
        ("/World/Bones/tibia", "tibia"),
    ]:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        try:
            mat = xform_cache.GetLocalToWorldTransform(prim)
            t = mat.ExtractTranslation()
            r = mat.ExtractRotationQuat()
            im = r.GetImaginary()
            extra[key] = {"xpos": [t[0], t[1], t[2]], "xquat": [r.GetReal(), im[0], im[1], im[2]]}
        except Exception:
            pass
    if extra and hasattr(avp, "_pose_stream_lock"):
        with avp._pose_stream_lock:
            if avp._current_poses and "poses" in avp._current_poses:
                avp._current_poses["poses"].update(extra)


if __name__ == "__main__":
    main()
    simulation_app.close()
