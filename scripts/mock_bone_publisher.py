"""
mock_bone_publisher.py
======================
Publishes bone origin transforms (PoseStamped, frame_id = lbr_link_0) on:
  /tracked/femur_origin
  /tracked/tibia_origin

Poses start from the values defined in med7_env_cfg.py.
Tibia is FIXED.  Femur SWINGS about X-axis around the knee joint.

Also publishes legacy /bone_pose_femur and /bone_pose_tibia for backward
compatibility.
"""

import sys, os, time, math


# ── ROS 2 bootstrap ─────────────────────────────────────────────────────────

def _setup_ros2():
    try:
        import isaacsim
        root = os.path.join(os.path.dirname(isaacsim.__file__),
                            "exts/isaacsim.ros2.bridge/humble")
        lib = os.path.join(root, "lib")
        pyp = os.path.join(root, "rclpy")
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        if lib not in cur:
            os.environ["LD_LIBRARY_PATH"] = f"{lib}:{cur}"
            os.execv(sys.executable, [sys.executable] + sys.argv)
        if pyp not in sys.path:
            sys.path.append(pyp)
    except ImportError:
        pass

_setup_ros2()

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


# ============================================================
#  ★  GROUND-TRUTH BONE POSES  —  copied from med7_env_cfg.py
#
#  If you change the bones in med7_env_cfg.py, copy the new
#  pos / rot values here to keep them in sync.
#
#  rot convention: (w, x, y, z)
# ============================================================

# Femur initial pose  (from med7_env_cfg.py)
FEMUR_INIT_POS = (0.904,  0.133,  0.7442)
FEMUR_INIT_ROT = (1.0, 0.0, 0.0, 0.0)        # identity  (360° about Y = no rotation)

# Tibia initial pose  (from med7_env_cfg.py line ~229-230)
TIBIA_INIT_POS = (0.9892, -0.0833, 0.7126)          # (x, y, z) metres
TIBIA_INIT_ROT = (0.7071, -0.7071, 0.0, 0.0)        # (w, x, y, z)

# ── Femur animation ─────────────────────────────────────────────────────────
# Swing range: 0° to −20° (DOWN only, then returns to base).
# Uses (1−cos)/2 envelope so the motion starts and ends smoothly at 0°.
# −0° = base position.  −20° = fully down (toward tibia).
KNEE_JOINT = (
    (FEMUR_INIT_POS[0] + TIBIA_INIT_POS[0]) / 2.0,
    (FEMUR_INIT_POS[1] + TIBIA_INIT_POS[1]) / 2.0,
    (FEMUR_INIT_POS[2] + TIBIA_INIT_POS[2]) / 2.0,
)
FEMUR_SWING_MIN_DEG = -45.0   # most-down position  (toward tibia)
FEMUR_SWING_MAX_DEG = -20.0   # least-down position  (closest to base)
FEMUR_SWING_FREQ_HZ =  0.15   # Hz  — one full oscillation every ~6.7 s


# ── Quaternion helpers  (w, x, y, z) ────────────────────────────────────────

def _normalise(q):
    w, x, y, z = q
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (w/n, x/n, y/n, z/n)


def _quat_axis_angle(ax, ay, az, rad):
    s = math.sin(rad / 2)
    return math.cos(rad / 2), ax*s, ay*s, az*s


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2)


def _rotate_vec(v, q):
    """Rotate vector v by unit quaternion q (w,x,y,z)."""
    w, qx, qy, qz = q
    vx, vy, vz    = v
    tx = 2*(qy*vz - qz*vy)
    ty = 2*(qz*vx - qx*vz)
    tz = 2*(qx*vy - qy*vx)
    return (vx + w*tx + qy*tz - qz*ty,
            vy + w*ty + qz*tx - qx*tz,
            vz + w*tz + qx*ty - qy*tx)


def _rotate_pose_about_world_axis(pos, rot, pivot, axis_quat):
    """
    Rotate a pose (pos, rot) about a world-space pivot point.

    axis_quat : quaternion representing the incremental rotation to apply.
    pivot     : (x, y, z) world point to rotate around.

    Returns (new_pos, new_rot).
    """
    # Translate pos to pivot frame
    dx = pos[0] - pivot[0]
    dy = pos[1] - pivot[1]
    dz = pos[2] - pivot[2]
    # Rotate the offset vector
    rdx, rdy, rdz = _rotate_vec((dx, dy, dz), axis_quat)
    # Translate back
    new_pos = (pivot[0] + rdx, pivot[1] + rdy, pivot[2] + rdz)
    # Compose orientation: apply axis_quat on top of current rot (world-frame rotation)
    new_rot = _quat_mul(axis_quat, rot)
    return new_pos, _normalise(new_rot)


# ── Publisher node ───────────────────────────────────────────────────────────

class MockBonePublisher(Node):

    def __init__(self):
        super().__init__('mock_bone_publisher')

        # Origin topics (used by simulation to move bone USD prims)
        self.femur_origin_pub = self.create_publisher(PoseStamped, '/tracked/femur_origin', 10)
        self.tibia_origin_pub = self.create_publisher(PoseStamped, '/tracked/tibia_origin', 10)

        # Legacy topics (kept for backward compatibility)
        self.femur_pub = self.create_publisher(PoseStamped, '/bone_pose_femur', 10)
        self.tibia_pub = self.create_publisher(PoseStamped, '/bone_pose_tibia', 10)

        self.timer = self.create_timer(0.02, self._cb)   # 50 Hz
        self.t0    = time.time()

        self._femur_base_pos = FEMUR_INIT_POS
        self._femur_base_rot = _normalise(FEMUR_INIT_ROT)
        self._tibia_pos      = TIBIA_INIT_POS
        self._tibia_rot      = _normalise(TIBIA_INIT_ROT)
        self._knee           = KNEE_JOINT

        self.get_logger().info(
            f"\n{'─'*62}\n"
            f"  Mock Bone Origin Publisher  (frame = lbr_link_0)\n"
            f"  Femur init pos : {FEMUR_INIT_POS}\n"
            f"  Femur init rot : {FEMUR_INIT_ROT}\n"
            f"  Tibia init pos : {TIBIA_INIT_POS}\n"
            f"  Tibia init rot : {TIBIA_INIT_ROT}\n"
            f"  Knee pivot     : {tuple(round(v,4) for v in self._knee)}\n"
            f"  Femur swing    : {FEMUR_SWING_MIN_DEG}° to {FEMUR_SWING_MAX_DEG}°"
            f" @ {FEMUR_SWING_FREQ_HZ} Hz about X-axis\n"
            f"  Origin topics  : /tracked/femur_origin  /tracked/tibia_origin\n"
            f"  Legacy topics  : /bone_pose_femur  /bone_pose_tibia\n"
            f"{'─'*62}"
        )

    # ── Timer ────────────────────────────────────────────────────────────────

    def _cb(self):
        t   = time.time() - self.t0
        now = self.get_clock().now().to_msg()

        # Femur: oscillate about X-axis around knee pivot
        mid_deg   = (FEMUR_SWING_MIN_DEG + FEMUR_SWING_MAX_DEG) / 2.0
        amp_deg   = (FEMUR_SWING_MAX_DEG - FEMUR_SWING_MIN_DEG) / 2.0
        swing_deg = mid_deg + amp_deg * math.sin(2 * math.pi * FEMUR_SWING_FREQ_HZ * t)
        swing_rad = math.radians(swing_deg)
        swing_q   = _quat_axis_angle(1.0, 0.0, 0.0, swing_rad)

        femur_pos, femur_rot = _rotate_pose_about_world_axis(
            self._femur_base_pos, self._femur_base_rot, self._knee, swing_q)

        # Publish on all topics (frame_id = lbr_link_0 for origin topics)
        self._pub(self.femur_origin_pub, now, femur_pos, femur_rot, frame='lbr_link_0')
        self._pub(self.tibia_origin_pub, now, self._tibia_pos, self._tibia_rot, frame='lbr_link_0')
        self._pub(self.femur_pub,        now, femur_pos, femur_rot, frame='world')
        self._pub(self.tibia_pub,        now, self._tibia_pos, self._tibia_rot, frame='world')

        if int(t * 10) % 40 == 0:
            self.get_logger().info(
                f"t={t:.1f}s  swing={swing_deg:+.1f}°  "
                f"femur=({femur_pos[0]:.3f},{femur_pos[1]:.3f},{femur_pos[2]:.3f})")

    @staticmethod
    def _pub(publisher, stamp, pos, q, frame='lbr_link_0'):
        msg = PoseStamped()
        msg.header.stamp       = stamp
        msg.header.frame_id    = frame
        msg.pose.position.x    = pos[0]
        msg.pose.position.y    = pos[1]
        msg.pose.position.z    = pos[2]
        msg.pose.orientation.w = q[0]
        msg.pose.orientation.x = q[1]
        msg.pose.orientation.y = q[2]
        msg.pose.orientation.z = q[3]
        publisher.publish(msg)


# ── Entry point ──────────────────────────────────────────────────────────────

def main(args=None):
    try:
        import carb
    except ImportError:
        import types
        m = types.ModuleType('carb')
        m.log_info = m.log_warn = m.log_error = lambda x: print(f'[carb] {x}')
        sys.modules['carb'] = m

    rclpy.init(args=args)
    node = MockBonePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
