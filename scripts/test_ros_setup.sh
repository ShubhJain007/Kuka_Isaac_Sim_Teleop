#!/bin/bash
# Test script to verify ROS integration setup

echo "=== Isaac Sim ROS2 Integration Test ==="
echo ""
echo "Checking Python version..."
python --version

echo ""
echo "Checking for system ROS in PYTHONPATH..."
if echo "$PYTHONPATH" | grep -q "/opt/ros"; then
    echo "WARNING: System ROS found in PYTHONPATH:"
    echo "$PYTHONPATH" | tr ':' '\n' | grep "/opt/ros"
else
    echo "OK: No system ROS in PYTHONPATH"
fi

echo ""
echo "Checking for system ROS in LD_LIBRARY_PATH..."
if echo "$LD_LIBRARY_PATH" | grep -q "/opt/ros"; then
    echo "WARNING: System ROS found in LD_LIBRARY_PATH:"
    echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep "/opt/ros"
else
    echo "OK: No system ROS in LD_LIBRARY_PATH"
fi

echo ""
echo "Testing rclpy import..."
python -c "
import sys
sys.path = [p for p in sys.path if '/opt/ros/' not in p]
try:
    import isaacsim
    print(f'Isaac Sim found at: {isaacsim.__file__}')
except ImportError:
    print('ERROR: isaacsim not importable - make sure Isaac Sim env is active')
    exit(1)
" || exit 1

echo ""
echo "=== Test Complete ==="
