"""
Minimal ROS 2 bridge: subscribes to bone transforms and robot joint states.

No EKF, no point clouds, no ICP — bones already arrive as filtered poses
from /tf. Joint states arrive from /lbr/joint_states.

Replaces: med7_ros_bones.py + med7_ros_pointcloud.py (~800 lines → ~150).
"""

from __future__ import annotations

import os
import sys
import numpy as np


def setup_ros2_libs() -> bool:
    """Configure Python path to use Isaac Sim's bundled ROS 2 libraries.

    Must be called BEFORE any rclpy import. May re-exec the process
    if LD_LIBRARY_PATH needs updating.
    """
    sys.path = [p for p in sys.path
                if "/opt/ros/" not in p
                and "site-packages/ros" not in p
                and "dist-packages/rclpy" not in p]

    try:
        import isaacsim
        isaacsim_path = os.path.dirname(isaacsim.__file__)
    except ImportError:
        print("[ROS] Cannot find isaacsim — ROS 2 bridge unavailable.")
        return False

    bridge_base = os.path.join(isaacsim_path, "exts", "isaacsim.ros2.bridge", "humble")
    lib_path = os.path.join(bridge_base, "lib")
    rclpy_path = os.path.join(bridge_base, "rclpy")

    if not os.path.isdir(lib_path):
        print(f"[ROS] Isaac Sim ROS2 bridge not found at {bridge_base}")
        return False

    sys.path.insert(0, rclpy_path)

    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_path not in ld:
        os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{ld}"
        os.execv(sys.executable, [sys.executable] + sys.argv)

    return True


class RosBridge:
    """Lightweight ROS 2 subscriber for bone poses and robot joint states."""

    def __init__(self):
        import rclpy
        from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

        try:
            rclpy.init()
        except RuntimeError as e:
            if "already" not in str(e).lower():
                raise

        from rclpy.node import Node
        self._node = rclpy.create_node("med7_lite_bridge")
        self._rclpy = rclpy

        sensor_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        from tf2_msgs.msg import TFMessage
        from sensor_msgs.msg import JointState

        self._node.create_subscription(TFMessage, "/tf", self._tf_cb, sensor_qos)
        self._node.create_subscription(JointState, "/lbr/joint_states", self._joint_cb, 10)

        # Bone state
        self.femur_pos: np.ndarray | None = None
        self.femur_quat: np.ndarray | None = None
        self.femur_updated = False

        self.tibia_pos: np.ndarray | None = None
        self.tibia_quat: np.ndarray | None = None
        self.tibia_updated = False

        # Robot state
        self.joint_names: list[str] = []
        self.joint_positions: np.ndarray | None = None
        self.joint_updated = False

        self._tf_frame_map = {
            "tracked/femur_origin": "femur",
            "tracked/tibia_origin": "tibia",
        }
        self._joint_logged = False
        self._tf_logged = False

    def spin_once(self):
        """Process pending ROS callbacks (non-blocking)."""
        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def _tf_cb(self, msg):
        for tf in msg.transforms:
            if not self._tf_logged:
                print(f"[ROS] TF frame received: child={tf.child_frame_id} parent={tf.header.frame_id}")
            bone = self._tf_frame_map.get(tf.child_frame_id)
            if bone is None:
                continue
            if not self._tf_logged:
                print(f"[ROS] Matched bone: {bone}")
                self._tf_logged = True

            t = tf.transform.translation
            r = tf.transform.rotation
            pos = np.array([t.x, t.y, t.z], dtype=np.float64)
            quat = np.array([r.w, r.x, r.y, r.z], dtype=np.float64)

            if bone == "femur":
                self.femur_pos = pos
                self.femur_quat = quat
                self.femur_updated = True
            else:
                self.tibia_pos = pos
                self.tibia_quat = quat
                self.tibia_updated = True

    def _joint_cb(self, msg):
        self.joint_names = list(msg.name)
        self.joint_positions = np.array(msg.position, dtype=np.float64)
        self.joint_updated = True
        if not self._joint_logged:
            print(f"[ROS] First joint_states: names={self.joint_names} pos={self.joint_positions.tolist()}")
            self._joint_logged = True

    def get_clock(self):
        return self._node.get_clock()

    def create_publisher(self, msg_type, topic, qos):
        return self._node.create_publisher(msg_type, topic, qos)

    def destroy(self):
        self._node.destroy_node()
        self._rclpy.shutdown()
