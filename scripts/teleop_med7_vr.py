
import argparse
import os
import time
import torch
import gymnasium as gym
import numpy as np
import cv2
import threading
from flask import Flask, Response

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Apple Vision Pro teleoperation for Kuka Med 7 via Web Streaming.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument("--teleop_device", type=str, default="handtracking", help="Device for interacting with environment ('handtracking', 'keyboard')")
parser.add_argument("--port", type=int, default=5000, help="Port for the Flask streaming server (default: 5000).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.utils.math as math_utils
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, OpenXRDevice
from isaaclab.devices.openxr.retargeters.manipulator import Se3RelRetargeter, Se3RelRetargeterCfg
from isaaclab.managers import SceneEntityCfg

import Kuka_Med_7  # noqa: F401
from Kuka_Med_7.tasks.med7.med7_env_cfg import Med7EnvCfg

# Global buffers for streaming
frame_buffer = {
    "wrist": None,
    "room": None
}
frame_lock = threading.Lock()

# --- Flask Server Logic ---
server = Flask(__name__)

def generate_mjpeg(camera_key):
    while True:
        with frame_lock:
            if frame_buffer[camera_key] is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img = frame_buffer[camera_key]
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03) # ~30 FPS

@server.route('/wrist')
def wrist_stream():
    return Response(generate_mjpeg('wrist'), mimetype='multipart/x-mixed-replace; boundary=frame')

@server.route('/room')
def room_stream():
    return Response(generate_mjpeg('room'), mimetype='multipart/x-mixed-replace; boundary=frame')

@server.route('/')
def home():
    return """
    <html>
        <head><title>Med7 VR Streaming</title></head>
        <body style="background: #111; color: #eee; font-family: sans-serif; text-align: center;">
            <h1>Kuka Med 7 VR Streams</h1>
            <div style="display: flex; justify-content: center; gap: 20px;">
                <div><h3>Wrist Camera</h3><img src="/wrist" width="448"></div>
                <div><h3>Room Camera</h3><img src="/room" width="448"></div>
            </div>
            <p>Open these URLs in Vision Pro Safari to pin them as floating windows.</p>
        </body>
    </html>
    """

def run_server():
    server.run(host='0.0.0.0', port=args_cli.port, threaded=True, use_reloader=False)

def main():
    # Start streaming server in background
    stream_thread = threading.Thread(target=run_server, daemon=True)
    stream_thread.start()
    print(f"[INFO]: Web streaming server started at http://0.0.0.0:{args_cli.port}")

    # create environment configuration
    env_cfg = Med7EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # create environment
    env = gym.make("Isaac-Med7-v0", cfg=env_cfg)
    env = env.unwrapped

    # Teleoperation state
    teleoperation_active = True if args_cli.teleop_device == "keyboard" else False

    def start_teleop(): nonlocal teleoperation_active; teleoperation_active = True; print("[INFO]: Teleop Started")
    def stop_teleop(): nonlocal teleoperation_active; teleoperation_active = False; print("[INFO]: Teleop Stopped")

    # Set up device
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
    elif args_cli.teleop_device.lower() == "handtracking":
        retargeter = Se3RelRetargeter(
            Se3RelRetargeterCfg(
                bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
                zero_out_xy_rotation=False,
                use_wrist_position=True,
                use_wrist_rotation=True,
                delta_rot_scale_factor=5.0,
                delta_pos_scale_factor=5.0,
            )
        )
        teleop_interface = OpenXRDevice(env_cfg.xr, retargeters=[retargeter])
        teleop_interface.add_callback("START", start_teleop)
        teleop_interface.add_callback("STOP", stop_teleop)
    
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface.add_callback("L", env.reset)
    elif args_cli.teleop_device.lower() == "handtracking":
        teleop_interface.add_callback("RESET", env.reset)
    teleop_interface.reset()

    # Robot entity for EE pose
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["lbr_A.*"], body_names=["lbr_link_7"])
    robot_entity_cfg.resolve(env.scene)

    # obtain the frame index of the end-effector
    if env.robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # create controller
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=env.num_envs, device=env.device)

    print("-" * 80)
    print("[IMPORTANT] VR Teleoperation Launch Instructions:")
    print("1. Ensure you are running with the CloudXR experience file:")
    print("   --experience apps/isaaclab.python.headless.cloudxr.kit")
    print("2. Set the environment variable: export EXTERNAL_RENDERER=cloudxr")
    print("3. Perform a PINCH gesture (Index + Thumb) to start control.")
    print("-" * 80)
    print("[INFO]: Setup complete. Waiting for VR connection...")

    # Action transformation for hand tracking (Wrist -> World)
    transform_matrix = torch.tensor(
        [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
        dtype=torch.float32, device=env.device
    )

    # Force enable OpenXR hand tracking settings in Carbonite
    import carb
    settings = carb.settings.get_settings()
    settings.set_bool("/xr/openxr/components/omni.kit.xr.openxr.ext.hand_tracking/enabled", True)
    settings.set_bool("/xr/openxr/components/isaacsim.xr.openxr.hand_tracking/enabled", True)
    # Also ensure AR profile is enabled
    settings.set_bool("/persistent/xr/profile/ar/enabled", True)

    import traceback
    
    # Check for CloudXR requirement
    if os.environ.get("EXTERNAL_RENDERER") != "cloudxr":
        print("\n" + "!" * 80)
        print("[WARNING] EXTERNAL_RENDERER is NOT set to 'cloudxr'.")
        print("          XR Hand tracking may not work correctly without this.")
        print("          Run: export EXTERNAL_RENDERER=cloudxr")
        print("!" * 80 + "\n")

    try:
        # loop
        step_count = 0
        while simulation_app.is_running():
            step_count += 1
            # 1. Get device command (Always poll to maintain connection and update visuals)
            delta_pose = teleop_interface.advance()
            
            if step_count % 300 == 0:
                print(f"[DEBUG] Raw tracking device output: {delta_pose}")

            try:
                with torch.inference_mode():
                    # Convert to torch
                    delta_pose_torch = delta_pose.to(env.device).repeat(env.num_envs, 1)

                    if args_cli.teleop_device == "handtracking":
                        # Diagnostic: Print non-zero deltas to confirm tracking is alive
                        if torch.any(delta_pose_torch != 0):
                            print(f"[DEBUG] TRACKING RECEIVED! Delta: {delta_pose_torch[0, :3].tolist()}")
                        else:
                            # Extra diagnostic: Check if we are getting raw data at all
                            raw_data = teleop_interface._get_raw_data()
                            right_hand = raw_data.get(OpenXRDevice.TrackingTarget.HAND_RIGHT, {})
                            if right_hand:
                                wrist_pose = right_hand.get("wrist")
                                # If wrist is zero [0,0,0], check if it's because it's never updated
                                if wrist_pose is not None and np.linalg.norm(wrist_pose[:3]) < 1e-5:
                                    # Peek into the actual device to see why it's not updating
                                    right_dev = teleop_interface._xr_core.get_input_device("/user/hand/right") if teleop_interface._xr_core else None
                                    if right_dev:
                                        all_poses = right_dev.get_all_virtual_world_poses()
                                        print(f"[DEBUG] Device found. Detected joints in dict: {list(all_poses.keys())}")
                                        if "wrist" in all_poses:
                                            validity = all_poses["wrist"].validity_flags
                                            print(f"[DEBUG] Wrist 'validity_flags': {validity}")
                                    else:
                                        print("[DEBUG] Right hand device handle is NULL.")
                                elif wrist_pose is not None:
                                    print(f"[DEBUG] Raw Wrist Pose: {wrist_pose[:3]}")
                            else:
                                print("[DEBUG] No hand data received from OpenXR.")
                        
                        # Apply coordinate transform
                        delta_pos_4d = torch.cat([delta_pose_torch[:, :3], torch.ones((delta_pose_torch.shape[0], 1), device=env.device)], dim=1)
                        delta_pos = torch.matmul(delta_pos_4d, transform_matrix)[:, :3]
                        
                        delta_rot_quat = math_utils.quat_from_euler_xyz(delta_pose_torch[:, 3], delta_pose_torch[:, 4], delta_pose_torch[:, 5])
                        delta_rot_mat = math_utils.matrix_from_quat(delta_rot_quat)
                        delta_rot_mat = torch.matmul(transform_matrix[:3, :3], torch.matmul(delta_rot_mat, transform_matrix[:3, :3].T))
                        delta_rot_quat = math_utils.quat_from_matrix(delta_rot_mat)
                    else:
                        delta_pos = delta_pose_torch[:, :3]
                        delta_rot_quat = math_utils.quat_from_euler_xyz(delta_pose_torch[:, 3], delta_pose_torch[:, 4], delta_pose_torch[:, 5])

                    # Get current EE pose
                    ee_pose_w = env.scene["robot"].data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
                    ee_pos_w, ee_rot_w = ee_pose_w[:, :3], ee_pose_w[:, 3:7]

                    # Compute target
                    target_pos_w, target_rot_w = math_utils.combine_frame_transforms(ee_pos_w, ee_rot_w, delta_pos, delta_rot_quat)
                    
                    # Diagnostic Prints
                    if step_count % 300 == 0:
                        print(f"[DEBUG] Heartbeat (Step {step_count}). Tracking: {'LIVE' if torch.any(delta_pose_torch != 0) else 'IDLE'}")
                        print(f"[DEBUG] Targeted EE Position (Env 0): {target_pos_w[0].tolist()}")

                    # 2. Update Hand Proxy visualization (Always)
                    # Create pose tensor [num_envs, 7] (pos + quat)
                    hand_pose = torch.zeros((env.num_envs, 7), device=env.device)
                    hand_pose[:, :3] = target_pos_w
                    hand_pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1) # Identity rotation
                    env.scene["hand_proxy"].write_root_pose_to_sim(hand_pose)
            except Exception as e:
                if "invalidated" in str(e).lower() or "backend" in str(e).lower():
                    print(f"[WARNING]: Simulation backend view invalidated during pose update. Stopping loop.")
                    break
                else:
                    raise e

            # 3. Handle Simulation Step / Render
            if teleoperation_active:
                with torch.inference_mode():
                    # obtain quantities from simulation
                    jacobian = env.robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
                    ee_pose_w = env.robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
                    root_pose_w = env.robot.data.root_pose_w
                    joint_pos = env.robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

                    # compute frame in root frame
                    ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
                        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
                    )
                    
                    # Compute target in root frame
                    target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
                        root_pose_w[:, 0:3], root_pose_w[:, 3:7], target_pos_w, target_rot_w
                    )

                    # Compute IK command (relative pose command)
                    # The controller expects [dx, dy, dz, drx, dry, drz]
                    # We compute the error between target and current EE in the root frame
                    pos_error_b, rot_error_quat_b = math_utils.compute_pose_error(
                        ee_pos_b, ee_quat_b, target_pos_b, target_quat_b, rot_error_type="quat"
                    )
                    ik_command = torch.cat([pos_error_b, math_utils.axis_angle_from_quat(rot_error_quat_b)], dim=1)

                    # set command to controller
                    diff_ik_controller.set_command(ik_command, ee_pos_b, ee_quat_b)
                    
                    # compute the joint commands
                    joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
                    
                    # Step environment (performs physics and rendering)
                    obs, rew, terminated, truncated, info = env.step(joint_pos_des)
            else:
                # Pulse the simulation app to satisfy XR frame submission and update graphics
                simulation_app.update()

            # 4. Update shared frame buffers for Flask streaming (ALWAYS)
            # When using simulation_app.update() instead of env.step(), we need to manually update sensors
            if not teleoperation_active:
                try:
                    # Surgically update only cameras to avoid triggering Articulation view invalidation errors
                    if "wrist_camera" in env.scene.sensors:
                        env.scene.sensors["wrist_camera"].update(dt=0.01)
                    if "room_camera" in env.scene.sensors:
                        env.scene.sensors["room_camera"].update(dt=0.01)
                except Exception as e:
                    if "invalidated" in str(e).lower() or "backend" in str(e).lower():
                        print(f"[WARNING]: Simulation backend view invalidated (VR Stop?). Stopping loop.")
                        break
                    else:
                        raise e

            with frame_lock:
                try:
                    if "wrist_camera" in env.scene.sensors:
                        wrist_data = env.scene.sensors["wrist_camera"].data.output["rgb"]
                        frame_buffer["wrist"] = wrist_data[0, ...].cpu().numpy()
                    if "room_camera" in env.scene.sensors:
                        room_data = env.scene.sensors["room_camera"].data.output["rgb"]
                        frame_buffer["room"] = room_data[0, ...].cpu().numpy()
                except Exception as e:
                     print(f"[DEBUG]: Failed to fetch camera data (Session closing?): {e}")
                     break
    except Exception:
        print("[ERROR]: An error occurred during the simulation loop:")
        traceback.print_exc()
    finally:
        env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
