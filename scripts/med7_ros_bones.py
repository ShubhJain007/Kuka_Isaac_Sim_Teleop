# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Optional ROS 2 bone pose subscription shared by teleop scripts."""

from __future__ import annotations

import os
import sys

import torch


def setup_ros2_libs() -> bool:
    """Set LD_LIBRARY_PATH for bundled ROS 2 bridge; re-exec if needed.

    Removes system ROS *Python* paths to avoid version conflicts (ROS Humble = 3.10,
    Isaac Sim = 3.11), but KEEPS system ROS *library* paths so the DDS middleware
    (rmw_fastrtps / rmw_cyclonedds) can still discover publishers on the local machine.
    """
    try:
        import isaacsim

        isaacsim_path = os.path.dirname(isaacsim.__file__)
        ros2_bridge_root = os.path.join(isaacsim_path, "exts/isaacsim.ros2.bridge/humble")
        lib_path = os.path.join(ros2_bridge_root, "lib")
        python_path = os.path.join(ros2_bridge_root, "rclpy")

        # Remove system ROS from *Python* path only (keeps C++ libs for DDS discovery)
        sys.path = [p for p in sys.path
                     if "dist-packages/rclpy" not in p
                     and "site-packages/rclpy" not in p
                     and "/opt/ros/" not in p]

        # Prepend Isaac Sim's ROS2 bridge to sys.path (highest priority)
        if python_path not in sys.path:
            sys.path.insert(0, python_path)

        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if lib_path not in current_ld_path:
            # PREPEND Isaac Sim libs but KEEP system ROS libs for DDS/middleware
            os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{current_ld_path}"
            os.execv(sys.executable, [sys.executable] + sys.argv)

        return True
    except ImportError:
        return False


def make_bone_tracker_node(device: torch.device):
    """Build a ROS 2 node that subscribes to femur/tibia PoseStamped topics."""
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped

    class BoneTrackerNode(Node):
        def __init__(self):
            super().__init__("bone_tracker_node")
            self.create_subscription(PoseStamped, "/bone_pose_femur", self.femur_callback, 10)
            self.create_subscription(PoseStamped, "/bone_pose_tibia", self.tibia_callback, 10)
            self.femur_pos = None
            self.femur_quat = None
            self.tibia_pos = None
            self.tibia_quat = None
            self.femur_updated = False
            self.tibia_updated = False

        def _pose_tensors(self, msg):
            pos = torch.tensor(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
                device=device,
                dtype=torch.float32,
            ).unsqueeze(0)
            quat = torch.tensor(
                [
                    msg.pose.orientation.w,
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                ],
                device=device,
                dtype=torch.float32,
            ).unsqueeze(0)
            return pos, quat

        def femur_callback(self, msg):
            self.femur_pos, self.femur_quat = self._pose_tensors(msg)
            self.femur_updated = True

        def tibia_callback(self, msg):
            self.tibia_pos, self.tibia_quat = self._pose_tensors(msg)
            self.tibia_updated = True

    return BoneTrackerNode()
