import sys
import os
import time
import math

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
            os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{current_ld_path}"
            os.execv(sys.executable, [sys.executable] + sys.argv)
        
        # Add the Python path to sys.path
        if python_path not in sys.path:
            sys.path.append(python_path)
        return True
    except ImportError:
        return False

setup_ros2_libs()

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class MockBonePublisher(Node):
    def __init__(self):
        super().__init__('mock_bone_publisher')
        self.femur_pub = self.create_publisher(PoseStamped, '/bone_pose_femur', 10)
        self.tibia_pub = self.create_publisher(PoseStamped, '/bone_pose_tibia', 10)
        timer_period = 0.02  # 50Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.start_time = time.time()

    def timer_callback(self):
        t = time.time() - self.start_time
        
        # Publish Femur
        femur_msg = PoseStamped()
        femur_msg.header.stamp = self.get_clock().now().to_msg()
        femur_msg.header.frame_id = 'world'
        femur_msg.pose.position.x = 1.3 + 0.05 * math.cos(t)
        femur_msg.pose.position.y = 0.0 + 0.05 * math.sin(t)
        femur_msg.pose.position.z = 0.6
        femur_msg.pose.orientation.x = 0.7071067811865476
        femur_msg.pose.orientation.y = 0.0
        femur_msg.pose.orientation.z = 0.0
        femur_msg.pose.orientation.w = 0.7071067811865476
        self.femur_pub.publish(femur_msg)

        # Publish Tibia (offset from femur)
        tibia_msg = PoseStamped()
        tibia_msg.header.stamp = self.get_clock().now().to_msg()
        tibia_msg.header.frame_id = 'world'
        tibia_msg.pose.position.x = 1.5 + 0.05 * math.cos(t + 0.5)
        tibia_msg.pose.position.y = 0.0 + 0.05 * math.sin(t + 0.5)
        tibia_msg.pose.position.z = 0.6
        tibia_msg.pose.orientation.x = 0.7071067811865476
        tibia_msg.pose.orientation.y = 0.0
        tibia_msg.pose.orientation.z = 0.0
        tibia_msg.pose.orientation.w = 0.7071067811865476
        self.tibia_pub.publish(tibia_msg)

def main(args=None):
    # This mock script doesn't have SimulationApp, so we need to mock 'carb' if rclpy imports it
    # Actually, rclpy/__init__.py imports carb, so we might need a fake carb module if it's not present.
    try:
        import carb
    except ImportError:
        import types
        sys.modules['carb'] = types.ModuleType('carb')
        sys.modules['carb'].log_info = lambda x: print(f"[MOCK CARB] {x}")
        sys.modules['carb'].log_warn = lambda x: print(f"[MOCK CARB] {x}")
        sys.modules['carb'].log_error = lambda x: print(f"[MOCK CARB] {x}")

    rclpy.init(args=args)
    mock_pub = MockBonePublisher()
    print("Mock bone publisher started. Publishing to /bone_pose...")
    try:
        rclpy.spin(mock_pub)
    except KeyboardInterrupt:
        pass
    mock_pub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
