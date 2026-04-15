# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run keyboard teleoperation for the Kuka Med 7 robot.
"""

import pathlib
import sys

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from med7_ros_bones import make_bone_tracker_node, setup_ros2_libs

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Kuka Med 7.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--no-ros",
    action="store_true",
    help="Do not use ROS 2 bone topics (avoids rclpy / bridge setup if sim fails to start).",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if not args_cli.no_ros:
    setup_ros2_libs()

# launch omniverse app
# SimulationApp initialization handles many paths.
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
    import carb
    import omni.usd
    import math
    from pxr import UsdGeom, Gf, Usd, UsdShade, Sdf

    # create environment configuration
    env_cfg = Med7EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # create environment
    env = gym.make("Isaac-Med7-v0", cfg=env_cfg)
    env = env.unwrapped

    # USD stage after env spawn (reliable for /World/Bones and surgical prims)
    stage = omni.usd.get_context().get_stage()

    ros_node = None
    rclpy = None
    if not args_cli.no_ros:
        import rclpy as _rclpy

        rclpy = _rclpy
        try:
            rclpy.init()
        except RuntimeError as e:
            if "already" not in str(e).lower():
                raise
        ros_node = make_bone_tracker_node(env.device)

    # create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=env.num_envs, device=env.device)

    # create keyboard device
    teleop_interface = Se3Keyboard(
        Se3KeyboardCfg(pos_sensitivity=0.1 * args_cli.sensitivity, rot_sensitivity=0.1 * args_cli.sensitivity)
    )
    teleop_interface.reset()

    # ── Surgical Plan Visualization ──────────────────────────────────────────
    # 0. Make bones translucent
    def _set_translucency(prim, opacity=0.5):
        # 1. Apply Display Opacity (basic)
        if prim.IsA(UsdGeom.Gprim):
            UsdGeom.Gprim(prim).GetDisplayOpacityAttr().Set([opacity])
        
        # 2. Apply Shader Opacity (for materials)
        mat_binding = UsdShade.MaterialBindingAPI(prim)
        if mat_binding:
            material = mat_binding.GetDirectBinding().GetMaterial()
            if material:
                # Look for the surface shader (typically UsdPreviewSurface)
                shader, _, _ = material.ComputeSurfaceSource()
                if shader:
                    # Try to set "opacity" input
                    op_input = shader.GetInput("opacity")
                    if op_input:
                        op_input.Set(opacity)
                    else:
                        # Create if it doesn't exist (less likely for standard converted STLs)
                        shader.CreateInput("opacity", UsdShade.SdfValueTypeNames.Float).Set(opacity)

        # Recurse to children
        for child in prim.GetChildren():
            _set_translucency(child, opacity)

    for bone_path in ["/World/Bones/femur", "/World/Bones/tibia"]:
        bone_root = stage.GetPrimAtPath(bone_path)
        if bone_root.IsValid():
            _set_translucency(bone_root)
    
    # 1. Virtual Probe (Surgical Pointer) — parent scope must exist for USD Define()
    if not stage.GetPrimAtPath("/World/SurgicalPlan").IsValid():
        UsdGeom.Scope.Define(stage, "/World/SurgicalPlan")
    probe_prim_path = "/World/SurgicalPlan/Probe"
    probe_prim = stage.GetPrimAtPath(probe_prim_path)
    if not probe_prim.IsValid():
        probe_sphere = UsdGeom.Sphere.Define(stage, probe_prim_path)
        probe_sphere.GetRadiusAttr().Set(0.015)
        probe_sphere.GetDisplayColorAttr().Set([(0.0, 1.0, 1.0)]) # Cyan
        probe_prim = probe_sphere.GetPrim()
    
    # 2. Guide Line (Cylinder)
    line_prim_path = "/World/SurgicalPlan/GuideLine"
    line_prim = stage.GetPrimAtPath(line_prim_path)
    if not line_prim.IsValid():
        line_cyl = UsdGeom.Cylinder.Define(stage, line_prim_path)
        line_cyl.GetRadiusAttr().Set(0.002)
        line_cyl.GetHeightAttr().Set(0.2)
        line_cyl.GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)]) # Red
        line_prim = line_cyl.GetPrim()

    # State variables
    probe_pos = [0.90, 0.0, 0.75]
    probe_euler = [0.0, 0.0, 0.0] # degrees (X, Y, Z)
    line_depth = 0.20 # 20cm total length
    line_pos = None
    line_rot = None
    
    bone_opacity = 0.5
    bone_prev_opacity = 1.0
    bone_opacity_needs_update = True
    
    # Keyboard listener for extra keys
    import omni.appwindow
    _input = carb.input.acquire_input_interface()
    _keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    
    def _on_keyboard_event(event):
        nonlocal probe_pos, probe_euler, line_depth, line_pos, line_rot
        nonlocal bone_opacity, bone_prev_opacity, bone_opacity_needs_update
        
        if event.type == carb.input.KeyboardEventType.KEY_PRESS or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
            step = 0.01 # 1cm / 1%
            deg_step = 2.0 # 2 degrees
            
            # Arrow Keys: X and Y Pos
            if event.input == carb.input.KeyboardInput.UP:    probe_pos[0] += step
            if event.input == carb.input.KeyboardInput.DOWN:  probe_pos[0] -= step
            if event.input == carb.input.KeyboardInput.LEFT:  probe_pos[1] += step
            if event.input == carb.input.KeyboardInput.RIGHT: probe_pos[1] -= step
            # PageUp/Down: Z Pos
            if event.input == carb.input.KeyboardInput.PAGE_UP:   probe_pos[2] += step
            if event.input == carb.input.KeyboardInput.PAGE_DOWN: probe_pos[2] -= step
            
            # Rotation: Y/H (X-axis), N/M (Y-axis)
            if event.input == carb.input.KeyboardInput.Y: probe_euler[0] += deg_step
            if event.input == carb.input.KeyboardInput.H: probe_euler[0] -= deg_step
            if event.input == carb.input.KeyboardInput.N: probe_euler[1] += deg_step
            if event.input == carb.input.KeyboardInput.M: probe_euler[1] -= deg_step
            
            # Depth: [ and ]
            if event.input == carb.input.KeyboardInput.LEFT_BRACKET: 
                line_depth = max(0.01, line_depth - step)
                print(f"[SURGERY] Planning Length: {line_depth*100:.1f} cm")
            if event.input == carb.input.KeyboardInput.RIGHT_BRACKET:
                line_depth += step
                print(f"[SURGERY] Planning Length: {line_depth*100:.1f} cm")

            # Opacity: , (Comma) and . (Period)
            if event.input == carb.input.KeyboardInput.COMMA:
                bone_opacity = max(0.0, bone_opacity - 0.05)
                bone_opacity_needs_update = True
                print(f"[SURGERY] Bone Opacity: {bone_opacity*100:.0f}%")
            if event.input == carb.input.KeyboardInput.PERIOD:
                bone_opacity = min(1.0, bone_opacity + 0.05)
                bone_opacity_needs_update = True
                print(f"[SURGERY] Bone Opacity: {bone_opacity*100:.0f}%")
            # Toggle Opacity: O
            if event.input == carb.input.KeyboardInput.O:
                if bone_opacity > 0.99:
                    bone_opacity = bone_prev_opacity if bone_prev_opacity < 0.99 else 0.5
                else:
                    bone_prev_opacity = bone_opacity
                    bone_opacity = 1.0
                bone_opacity_needs_update = True
                print(f"[SURGERY] Bone Opacity: {bone_opacity*100:.0f}%")
                
            # Drop Line: P (Saves the pose)
            if event.input == carb.input.KeyboardInput.P:
                line_pos = list(probe_pos)
                line_rot = list(probe_euler)
                print(f"[SURGERY] Saved Guide Line at {line_pos}, Rot {line_rot}")

    _input.subscribe_to_keyboard_events(_keyboard, _on_keyboard_event)

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
    print("[INFO]: ── SURGICAL PLAN ──────────────────────────────────────")
    print("[INFO]: Use ARROW KEYS & PgUp/PgDn to move the Virtual Probe.")
    print("[INFO]: Use Y/H and N/M to adjust the drilling angle.")
    print("[INFO]: Press 'P' to drop/update the surgical guide line.")
    print("[INFO]: Use '[' and ']' to adjust total guide length.")
    print("[INFO]: Use ',' and '.' to adjust bone opacity; 'O' to toggle.")
    print("[INFO]: ──────────────────────────────────────────────────────")
    if args_cli.no_ros:
        print("[INFO]: ROS bone topics disabled (--no-ros).")
    else:
        print("[INFO]: Listening on /bone_pose_femur and /bone_pose_tibia (geometry_msgs/PoseStamped).")

    # simulate physics
    while simulation_app.is_running():
        if ros_node is not None and rclpy is not None:
            rclpy.spin_once(ros_node, timeout_sec=0.001)

        # run everything in inference mode
        with torch.inference_mode():
            if ros_node is not None:
                if ros_node.femur_updated:
                    env.update_femur_pose(ros_node.femur_pos, ros_node.femur_quat)
                    ros_node.femur_updated = False
                if ros_node.tibia_updated:
                    env.update_tibia_pose(ros_node.tibia_pos, ros_node.tibia_quat)
                    ros_node.tibia_updated = False

            # get device command
            delta_pose = teleop_interface.advance()
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
            
            # ── Update Surgical Visuals ──────────────────────────────────────
            # Update Bone Translucency if requested
            if bone_opacity_needs_update:
                for bone_path in ["/World/Bones/femur", "/World/Bones/tibia"]:
                    bone_root = stage.GetPrimAtPath(bone_path)
                    if bone_root.IsValid():
                        _set_translucency(bone_root, bone_opacity)
                bone_opacity_needs_update = False

            # Helper to create xform matrix from pos/euler
            def get_matrix(pos, euler):
                m = Gf.Matrix4d()
                # Apply rotations in Z, Y, X order (common for USD)
                rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), euler[2]) * \
                      Gf.Rotation(Gf.Vec3d(0, 1, 0), euler[1]) * \
                      Gf.Rotation(Gf.Vec3d(1, 0, 0), euler[0])
                m.SetRotateOnly(rot)
                m.SetTranslateOnly(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
                return m

            # Update Probe
            UsdGeom.Xformable(probe_prim).MakeMatrixXform().Set(get_matrix(probe_pos, probe_euler))
            
            # Update Guide Line (always visible, attached to probe or saved pose)
            target_pos = line_pos if line_pos is not None else probe_pos
            target_rot = line_rot if line_rot is not None else probe_euler
            
            # Place cylinder symmetrically at target_pos
            UsdGeom.Xformable(line_prim).MakeMatrixXform().Set(get_matrix(target_pos, target_rot))
            
            # Update height if changed
            attr_height = line_prim.GetAttribute("height")
            if abs(attr_height.Get() - line_depth) > 0.001:
                attr_height.Set(line_depth)
            
            # step
            obs, rew, terminated, truncated, info = env.step(joint_pos_des)
            
    env.close()
    if ros_node is not None and rclpy is not None:
        ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
