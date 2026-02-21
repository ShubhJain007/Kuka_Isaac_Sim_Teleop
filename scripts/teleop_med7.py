# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run keyboard teleoperation for the Kuka Med 7 robot.
"""

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

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
import Kuka_Med_7  # noqa: F401
from Kuka_Med_7.tasks.med7.med7_env_cfg import Med7EnvCfg


def main():
    """Main function."""
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

    # simulate physics
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
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


if __name__ == "__main__":
    main()
    simulation_app.close()
