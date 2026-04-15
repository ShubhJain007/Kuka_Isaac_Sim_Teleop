# ROS Integration: Point Cloud & Robot State Sync

This document describes the ROS2 integration for visualizing tracked bone point clouds and synchronizing robot state.

## Features

### ✅ Point Cloud Visualization
Renders ROS PointCloud2 data as USD Points prims in Isaac Sim for real-time bone tracking visualization.

### ⚠️ Robot State Monitoring
Subscribes to `/lbr/joint_states` for monitoring (active sync disabled by default to avoid conflicts with VR teleoperation).

### 📋 Robot Description
Monitors `/lbr/robot_description` for URDF updates.

## Files Created/Modified

### New Files
- `scripts/med7_ros_pointcloud.py` - ROS2 node for point cloud and joint state subscriptions
- `scripts/usd_pointcloud_viz.py` - USD Points helper for rendering point clouds
- `scripts/ROS_INTEGRATION.md` - This file

### Modified Files
- `source/Kuka_Med_7/Kuka_Med_7/tasks/med7/med7_env.py`
  - Removed USD bone mesh spawning
  - Added `update_pointcloud()` method for visualizing ROS point clouds
  - Added `sync_robot_joint_state()` method for syncing ROS joint states
- `scripts/teleop_med7_vr.py`
  - Updated to use new ROS integration node
  - Removed bone lock feature (K, 1, 2 keys)
  - Simplified bone opacity controls (`,` and `.` still work for future use)

## Usage

### Running with ROS Integration

**Recommended method** (handles environment automatically):
```bash
cd /home/kneepolean/Isaac_Lab_projects/Kuka_Med_7
./scripts/run_teleop_ros.sh
```

**Direct method** (if wrapper doesn't work):
```bash
cd /home/kneepolean/Isaac_Lab_projects/Kuka_Med_7

# Unset ROS environment variables first
unset ROS_DISTRO AMENT_PREFIX_PATH ROS_PYTHON_VERSION

python scripts/teleop_med7_vr.py --ros
```

### Prerequisites

1. **ROS Humble** must be running with active publishers:
   - `/tracked/femur` (sensor_msgs/PointCloud2)
   - `/lbr/joint_states` (sensor_msgs/JointState)
   - `/lbr/robot_description` (std_msgs/String)

2. **Environment setup**: The script automatically configures Isaac Sim's bundled ROS2 bridge libraries.

### Expected Behavior

- **Point clouds** appear at `/World/PointClouds/femur` as USD Points
- **Robot joints** sync from ROS to Isaac Sim articulation
- Console shows: `[INFO] ROS: /tracked/femur (PointCloud2), /lbr/robot_description, /lbr/joint_states`

## ROS Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/tracked/femur` | `sensor_msgs/PointCloud2` | Femur point cloud with optional RGB |
| `/lbr/joint_states` | `sensor_msgs/JointState` | Robot joint positions and velocities |
| `/lbr/robot_description` | `std_msgs/String` | Robot URDF description |

## Technical Details

### Point Cloud Rendering

Point clouds are converted from ROS PointCloud2 messages to USD Points prims:
- Extracts XYZ coordinates
- Parses RGB colors (if available) from packed float
- Renders with 3mm point size by default
- Fallback bone color: `(0.9, 0.7, 0.5)` if no RGB data

### Robot Joint Sync

**Current Status**: Disabled by default

The integration subscribes to `/lbr/joint_states` but doesn't actively sync the robot state during teleoperation to avoid conflicts with VR control. The ROS joint states are received and can be logged for monitoring purposes.

**To Enable** (experimental): Uncomment the sync block in `teleop_med7_vr.py` around line 515. Note:
- This will override VR teleoperation commands
- May cause conflicts between ROS state and VR commands
- Best used for visualization-only scenarios (no active VR teleoperation)

**Better Alternative**: For true robot-ROS sync, consider:
- Using the ROS robot state as the source of truth
- Disabling VR teleoperation commands (`teleoperation_active = False`)
- Only using VR for visualization and planning

### Performance

- ROS spin: `0.001s` timeout per frame
- Point cloud updates: Only when `pc_updated` flag is set
- Joint state sync: Only when `robot_state_updated` flag is set

## Legacy Features Removed

- USD bone meshes (`/World/Bones/femur`, `/World/Bones/tibia`) - no longer spawned
- Bone lock (K key, 1/2 keys) - removed (point clouds don't have pose tracking)
- Bone pose ROS topics (`/bone_pose_femur`, `/bone_pose_tibia`) - replaced by point clouds

## Troubleshooting

### ModuleNotFoundError: No module named 'rclpy._rclpy_pybind11'

**Symptom**: Python version mismatch error when importing rclpy
```
The C extension '...cpython-311-x86_64-linux-gnu.so' isn't present
```

**Cause**: ROS Humble is built for Python 3.10, but Isaac Sim uses Python 3.11. System ROS paths conflict with Isaac Sim's bundled ROS2 bridge.

**Fix**: The script now automatically:
1. Removes `/opt/ros/` from `sys.path` and `LD_LIBRARY_PATH`
2. Uses Isaac Sim's bundled ROS2 bridge at `isaacsim.ros2.bridge/humble`
3. Re-executes the script with correct paths if needed

If the error persists:
```bash
# Unset any ROS environment variables before running Isaac Sim
unset ROS_DISTRO
unset AMENT_PREFIX_PATH
unset ROS_PYTHON_VERSION

# Then run the script
python scripts/teleop_med7_vr.py --ros
```

### No point clouds visible
- Check ROS topic: `ros2 topic echo /tracked/femur --no-arr`
- Verify Isaac Sim ROS2 bridge: `ros2 node list | grep med7`

### Robot not syncing
- Check joint states: `ros2 topic echo /lbr/joint_states`
- Review console for joint name mapping warnings

### ROS initialization errors
- Ensure no other `rclpy.init()` calls in external scripts
- Check `LD_LIBRARY_PATH` includes Isaac Sim's ROS2 libs

## Future Enhancements

- Add tibia point cloud support (subscribe to `/tracked/tibia`)
- Implement point cloud opacity controls (left thumb-pinky spread)
- Add joint velocity sync from ROS
- Support multiple point cloud rendering modes (sphere vs. disk)
