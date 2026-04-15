#!/bin/bash
# Wrapper script to run teleop_med7_vr.py with ROS integration
# Ensures Isaac Sim's bundled ROS2 is used instead of system ROS Humble

# Unset ROS environment variables to prevent Python version conflicts,
# but KEEP ROS_DOMAIN_ID so we talk to the same DDS network as the publisher.
SAVED_ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
SAVED_RMW="${RMW_IMPLEMENTATION:-}"
SAVED_CYCLONE="${CYCLONEDDS_URI:-}"
SAVED_FASTRTPS="${FASTRTPS_DEFAULT_PROFILES_FILE:-}"

unset ROS_DISTRO
unset AMENT_PREFIX_PATH
unset ROS_PYTHON_VERSION
unset ROS_VERSION
unset ROS_LOCALHOST_ONLY

# Restore networking-critical vars
export ROS_DOMAIN_ID="$SAVED_ROS_DOMAIN_ID"
[ -n "$SAVED_RMW" ] && export RMW_IMPLEMENTATION="$SAVED_RMW"
[ -n "$SAVED_CYCLONE" ] && export CYCLONEDDS_URI="$SAVED_CYCLONE"
[ -n "$SAVED_FASTRTPS" ] && export FASTRTPS_DEFAULT_PROFILES_FILE="$SAVED_FASTRTPS"

# Remove system ROS *Python* paths only; KEEP library paths for DDS discovery
if [ -n "$PYTHONPATH" ]; then
    export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "/opt/ros/" | tr '\n' ':' | sed 's/:$//')
fi
# NOTE: LD_LIBRARY_PATH is intentionally NOT stripped — system ROS libs are
# needed for DDS middleware (rmw_fastrtps / rmw_cyclonedds) to discover publishers.

cd "$(dirname "$0")/.." || exit 1

# Source CloudXR environment (auto-detects Docker mount path for IPC socket)
if [ -f scripts/cloudxr_env.sh ]; then
    source scripts/cloudxr_env.sh
fi

echo ""
echo "=== Starting VR Teleoperation with ROS Integration ==="
echo "Using Isaac Sim's bundled ROS2 bridge (Python 3.11 compatible)"
echo ""

# Run the script with all arguments passed through
python scripts/teleop_med7_vr.py --ros --enable_cameras "$@"
