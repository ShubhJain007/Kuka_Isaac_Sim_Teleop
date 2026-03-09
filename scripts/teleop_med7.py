# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run keyboard teleoperation for the Kuka Med 7 robot.
"""

import sys
import os

def setup_ros2_libs():
    """Finds and sets the LD_LIBRARY_PATH for the bundled ROS 2 bridge to avoid ImportErrors."""
    try:
        import isaacsim
        isaacsim_path = os.path.dirname(isaacsim.__file__)
        ros2_bridge_root = os.path.join(isaacsim_path, "exts/isaacsim.ros2.bridge/humble")
        lib_path = os.path.join(ros2_bridge_root, "lib")
        python_path = os.path.join(ros2_bridge_root, "rclpy")

        # If the lib path is not in LD_LIBRARY_PATH, we must re-execute the script
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if lib_path not in current_ld_path:
            # print(f"[INFO] Automatically setting LD_LIBRARY_PATH to include bundled ROS 2 libraries...")
            os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{current_ld_path}"
            os.execv(sys.executable, [sys.executable] + sys.argv)
        
        # Add the Python path to sys.path
        if python_path not in sys.path:
            sys.path.append(python_path)
        return True
    except ImportError:
        return False

# Self-fix: Ensure the environment is set correctly before proceeding
setup_ros2_libs()

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Kuka Med 7.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import rclpy
from rclpy.node import Node

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
import Kuka_Med_7  # noqa: F401
from Kuka_Med_7.tasks.med7.med7_env_cfg import Med7EnvCfg


class BoneTrackerNode(Node):
    """ROS 2 node to subscribe to bone pose tracking data."""
    def __init__(self):
        super().__init__("bone_tracker_node")
        # Import PoseStamped here to ensure it's loaded from the bridge path
        from geometry_msgs.msg import PoseStamped
        
        # Femur subscription
        self.femur_sub = self.create_subscription(
            PoseStamped,
            "/bone_pose_femur",
            self.femur_callback,
            10
        )
        
        # Tibia subscription
        self.tibia_sub = self.create_subscription(
            PoseStamped,
            "/bone_pose_tibia",
            self.tibia_callback,
            10
        )
        
        self.femur_pos = None
        self.femur_quat = None
        self.tibia_pos = None
        self.tibia_quat = None
        
        self.femur_updated = False
        self.tibia_updated = False
        self._femur_msg_count = 0
        self._tibia_msg_count = 0

    def femur_callback(self, msg):
        """Callback to store the latest received femur pose."""
        # ROS Pose uses (x, y, z, w), Isaac Lab uses (w, x, y, z)
        self.femur_pos = torch.tensor(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], 
            device="cuda:0", dtype=torch.float32
        ).unsqueeze(0)
        self.femur_quat = torch.tensor(
            [msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z], 
            device="cuda:0", dtype=torch.float32
        ).unsqueeze(0)
        self.femur_updated = True
        self._femur_msg_count += 1
        if self._femur_msg_count == 1:
            print("[ROS] First femur pose received!")

    def tibia_callback(self, msg):
        """Callback to store the latest received tibia pose."""
        self.tibia_pos = torch.tensor(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], 
            device="cuda:0", dtype=torch.float32
        ).unsqueeze(0)
        self.tibia_quat = torch.tensor(
            [msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z], 
            device="cuda:0", dtype=torch.float32
        ).unsqueeze(0)
        self.tibia_updated = True
        self._tibia_msg_count += 1
        if self._tibia_msg_count == 1:
            print("[ROS] First tibia pose received!")


def main():
    """Main function."""
    # Initialize ROS 2
    rclpy.init()
    ros_node = BoneTrackerNode()

    # create environment configuration
    env_cfg = Med7EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    print(f"[DEBUG] env_cfg.scene: {env_cfg.scene}")
    
    # create environment
    print("[DEBUG] Calling gym.make...")
    env = gym.make("Isaac-Med7-v0", cfg=env_cfg)
    env = env.unwrapped

    # create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=env.num_envs, device=env.device)

    # create keyboard device
    teleop_interface = Se3Keyboard(
        Se3KeyboardCfg(pos_sensitivity=0.1 * args_cli.sensitivity, rot_sensitivity=0.1 * args_cli.sensitivity)
    )
    teleop_interface.reset()

    # specify robot-specific parameters
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["lbr_A.*"], body_names=["lbr_link_7"])
    robot_entity_cfg.resolve(env.scene)
    
    # obtain the frame index of the end-effector
    if env.robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # reset environment
    obs, _ = env.reset()
    
    print("[INFO]: Setup complete. Use W/S, A/D, Q/E to move EE, and Z/X, T/G, C/V to rotate.")
    print("[INFO]: Press 'L' to reset the environment.")
    print("[INFO]: Listening for bone pose on /bone_pose (geometry_msgs/PoseStamped)")

    # simulate physics
    while simulation_app.is_running():
        # Spin ROS to process callbacks (small timeout to actually check the queue)
        rclpy.spin_once(ros_node, timeout_sec=0.001)

        # run everything in inference mode
        with torch.inference_mode():
            # Update femur pose if new data arrived
            if ros_node.femur_updated:
                env.update_femur_pose(ros_node.femur_pos, ros_node.femur_quat)
                ros_node.femur_updated = False
            # Update tibia pose if new data arrived
            if ros_node.tibia_updated:
                env.update_tibia_pose(ros_node.tibia_pos, ros_node.tibia_quat)
                ros_node.tibia_updated = False

            # get device command
            delta_pose = teleop_interface.advance()
            # The keyboard device returns [x, y, z, rx, ry, rz, gripper]
            # convert to command for diff-ik (pose only)
            ik_command = delta_pose[:6].repeat(env.num_envs, 1)

            # obtain quantities from simulation
            jacobian = env.robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = env.robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            root_pose_w = env.robot.data.root_pose_w
            joint_pos = env.robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            
            # set command to controller
            diff_ik_controller.set_command(ik_command, ee_pos_b, ee_quat_b)
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            
            # step
            obs, rew, terminated, truncated, info = env.step(joint_pos_des)
            
    # close the environment
    env.close()
    ros_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
