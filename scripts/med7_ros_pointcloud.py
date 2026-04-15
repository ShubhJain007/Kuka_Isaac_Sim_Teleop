# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""ROS 2 integration: point cloud visualization + robot state sync.

IMPORTANT: This module should only be imported AFTER setup_ros2_libs() has been called
to ensure Isaac Sim's bundled ROS2 is used instead of system ROS (Python version conflicts).
"""

from __future__ import annotations

import numpy as np
import torch

from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

latch_qos = QoSProfile(
      depth=1,
      durability=DurabilityPolicy.TRANSIENT_LOCAL,
      reliability=ReliabilityPolicy.RELIABLE,
  )

sensor_qos = QoSProfile(
      depth=1,
      history=HistoryPolicy.KEEP_LAST,
      reliability=ReliabilityPolicy.BEST_EFFORT,
  )



# ── Manual PointCloud2 parser (no sensor_msgs_py dependency) ─────────────

# ROS PointField type → numpy dtype
_PFTYPE_TO_NP = {
    1: np.uint8,    # UINT8
    2: np.int8,     # INT8
    3: np.uint16,   # UINT16
    4: np.int16,    # INT16
    5: np.uint32,   # UINT32
    6: np.int32,    # INT32
    7: np.float32,  # FLOAT32
    8: np.float64,  # FLOAT64
}


def _parse_pointcloud2(msg) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Parse a PointCloud2 message into numpy arrays (vectorized, no Python loops).

    Returns:
        (points_Nx3, colors_Nx3) — colors may be None if no rgb/rgba field.
    """
    field_map = {}
    for f in msg.fields:
        field_map[f.name] = (f.offset, f.datatype, f.count)

    if "x" not in field_map or "y" not in field_map or "z" not in field_map:
        return None, None

    rgb_field = "rgb" if "rgb" in field_map else ("rgba" if "rgba" in field_map else None)
    point_step = msg.point_step
    n_points = msg.width * msg.height
    if n_points == 0:
        return None, None

    raw = bytes(msg.data)
    expected = n_points * point_step
    if len(raw) < expected:
        n_points = len(raw) // point_step
        if n_points == 0:
            return None, None

    x_off, x_dt, _ = field_map["x"]
    y_off, y_dt, _ = field_map["y"]
    z_off, z_dt, _ = field_map["z"]

    dt = np.dtype({
        "names": ["x", "y", "z"],
        "formats": [
            _PFTYPE_TO_NP.get(x_dt, np.float32),
            _PFTYPE_TO_NP.get(y_dt, np.float32),
            _PFTYPE_TO_NP.get(z_dt, np.float32),
        ],
        "offsets": [x_off, y_off, z_off],
        "itemsize": point_step,
    })
    structured = np.frombuffer(raw, dtype=dt, count=n_points)
    x = structured["x"].astype(np.float32)
    y = structured["y"].astype(np.float32)
    z = structured["z"].astype(np.float32)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[valid], y[valid], z[valid]
    if x.shape[0] == 0:
        return None, None

    points = np.column_stack((x, y, z))

    colors = None
    if rgb_field:
        rgb_off, rgb_dt, _ = field_map[rgb_field]
        raw_dt = np.dtype({"names": ["rgb"], "formats": [np.uint32],
                           "offsets": [rgb_off], "itemsize": point_step})
        rgb_raw = np.frombuffer(raw, dtype=raw_dt, count=n_points)["rgb"][valid]
        r = ((rgb_raw >> 16) & 0xFF).astype(np.float32) / 255.0
        g = ((rgb_raw >> 8) & 0xFF).astype(np.float32) / 255.0
        b = (rgb_raw & 0xFF).astype(np.float32) / 255.0
        colors = np.column_stack((r, g, b))
    else:
        colors = np.full((points.shape[0], 3), [0.9, 0.7, 0.5], dtype=np.float32)

    return points, colors


# ── ROS 2 Node ───────────────────────────────────────────────────────────

def make_ros_integration_node(device: torch.device):
    """Build ROS 2 node that subscribes to point clouds, robot description, and joint states."""
    import rclpy as rclpy  # noqa: PLC0414 — needed for create_node clock probe
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2, JointState, Image
    from std_msgs.msg import String

    from geometry_msgs.msg import PoseStamped as _PoseStamped
    from tf2_msgs.msg import TFMessage as _TFMessage  # direct /tf sub, no tf2 buffer

    class RosIntegrationNode(Node):
        def __init__(self):
            super().__init__("med7_ros_integration")

            self.create_subscription(PointCloud2, "/tracked/femur", self.femur_pc_callback, sensor_qos)
            self.create_subscription(PointCloud2, "/tracked/tibia", self.tibia_pc_callback, sensor_qos)
            # Also accept direct PoseStamped on origin topics (mock publisher / live system)
            self.create_subscription(_PoseStamped, "/tracked/femur_origin", self.femur_origin_callback, sensor_qos)
            self.create_subscription(_PoseStamped, "/tracked/tibia_origin", self.tibia_origin_callback, sensor_qos)
            self.create_subscription(String, "/lbr/robot_description", self.robot_desc_callback, latch_qos)
            self.create_subscription(JointState, "/lbr/joint_states", self.joint_state_callback, 10)
            self.create_subscription(Image, "/camera_arm/camera/color/image_rect_raw", self.arm_camera_callback, 10)

            # Direct /tf subscription — bypasses tf2 buffer entirely.
            # tf2's internal buffer retains old data and rejects replayed
            # (older-timestamped) transforms on bag restart, causing bones
            # to freeze forever.  Raw subscription has no such issue.
            self.create_subscription(_TFMessage, "/tf", self._raw_tf_callback, sensor_qos)
            self._tf_child_femur = "tracked/femur_origin"
            self._tf_child_tibia = "tracked/tibia_origin"

            # Femur point cloud
            self.pc_points: np.ndarray | None = None
            self.pc_colors: np.ndarray | None = None
            self.pc_updated = False

            # Tibia point cloud
            self.tibia_pc_points: np.ndarray | None = None
            self.tibia_pc_colors: np.ndarray | None = None
            self.tibia_pc_updated = False

            # Bone origin transforms (6-DOF pose wrt lbr_link_0)
            self.femur_origin_pos: np.ndarray | None = None   # [x, y, z]
            self.femur_origin_quat: np.ndarray | None = None  # [w, x, y, z]
            self.femur_origin_updated = False
            self.tibia_origin_pos: np.ndarray | None = None
            self.tibia_origin_quat: np.ndarray | None = None
            self.tibia_origin_updated = False
            self._femur_origin_count = 0
            self._tibia_origin_count = 0

            self.robot_description: str | None = None
            self.joint_names: list[str] = []
            self.joint_positions: torch.Tensor | None = None
            self.joint_velocities: torch.Tensor | None = None
            self.robot_state_updated = False

            # Arm camera image (for MJPEG stream)
            self.arm_camera_frame: np.ndarray | None = None
            self.arm_camera_updated = False
            self._cam_msg_count = 0

            self.device = device
            self._femur_msg_count = 0
            self._tibia_msg_count = 0
            self._js_msg_count = 0

        def _handle_pc(self, topic: str, msg: PointCloud2, count: int):
            """Shared point cloud parsing + debug logging."""
            try:
                pts, cols = _parse_pointcloud2(msg)
                if pts is not None and pts.shape[0] > 0:
                    if count <= 3 or count % 100 == 0:
                        print(f"[ROS] {topic} msg #{count}: {pts.shape[0]} pts, "
                              f"range x=[{pts[:,0].min():.3f}, {pts[:,0].max():.3f}] "
                              f"y=[{pts[:,1].min():.3f}, {pts[:,1].max():.3f}] "
                              f"z=[{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")
                    return pts, cols
                else:
                    if count <= 5:
                        fields = [f.name for f in msg.fields]
                        print(f"[ROS] {topic} msg #{count}: 0 valid pts "
                              f"(w={msg.width} h={msg.height} step={msg.point_step} fields={fields})")
            except Exception as e:
                print(f"[ROS] {topic} parse error (msg #{count}): {e}")
            return None, None

        def femur_pc_callback(self, msg: PointCloud2):
            self._femur_msg_count += 1
            pts, cols = self._handle_pc("/tracked/femur", msg, self._femur_msg_count)
            if pts is not None:
                self.pc_points = pts
                self.pc_colors = cols
                self.pc_updated = True

        def tibia_pc_callback(self, msg: PointCloud2):
            self._tibia_msg_count += 1
            pts, cols = self._handle_pc("/tracked/tibia", msg, self._tibia_msg_count)
            if pts is not None:
                self.tibia_pc_points = pts
                self.tibia_pc_colors = cols
                self.tibia_pc_updated = True

        # ── EKF for bone pose tracking ────────────────────────────────────
        # State: [x, y, z, vx, vy, vz]  (constant-velocity model)
        # Orientation: separate SLERP-rate predictor (angular velocity)
        # Outputs at a constant 30 Hz regardless of measurement rate.

        class _BoneEKF:
            """Constant-velocity Extended Kalman Filter for a single bone.

            Position uses a 6-state linear KF (pos + vel).
            Orientation uses quaternion angular-velocity prediction with
            SLERP-based correction on measurement.
            """

            def __init__(self, process_noise_pos=0.05, process_noise_vel=2.0,
                         meas_noise_pos=0.002, process_noise_quat=0.1,
                         meas_noise_quat=0.01):
                # Position KF state: [x, y, z, vx, vy, vz]
                self.x = np.zeros(6, dtype=np.float64)
                self.P = np.eye(6, dtype=np.float64) * 1.0

                # Tuning parameters
                self._q_pos  = process_noise_pos
                self._q_vel  = process_noise_vel
                self._r_pos  = meas_noise_pos

                # Measurement matrix: we observe [x, y, z] only
                self.H = np.zeros((3, 6), dtype=np.float64)
                self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0
                self.R = np.eye(3, dtype=np.float64) * meas_noise_pos

                # Orientation state
                self.quat = np.array([1.0, 0.0, 0.0, 0.0])  # [w,x,y,z]
                self.ang_vel = np.zeros(3, dtype=np.float64)  # rad/s in body frame
                self._q_quat  = process_noise_quat
                self._r_quat  = meas_noise_quat

                self.last_time = None
                self.initialized = False

            def predict(self, dt: float):
                """Propagate state forward by dt seconds."""
                if dt <= 0:
                    return

                # --- Position KF predict ---
                F = np.eye(6, dtype=np.float64)
                F[0, 3] = F[1, 4] = F[2, 5] = dt
                self.x = F @ self.x

                Q = np.zeros((6, 6), dtype=np.float64)
                dt2 = dt * dt
                dt3 = dt2 * dt
                q_p, q_v = self._q_pos, self._q_vel
                for i in range(3):
                    Q[i, i]     = q_p * dt3 / 3.0 + q_v * dt
                    Q[i, i+3]   = q_p * dt2 / 2.0
                    Q[i+3, i]   = q_p * dt2 / 2.0
                    Q[i+3, i+3] = q_p * dt + q_v * dt
                self.P = F @ self.P @ F.T + Q

                # --- Orientation predict (integrate angular velocity) ---
                angle = float(np.linalg.norm(self.ang_vel)) * dt
                if angle > 1e-8:
                    axis = self.ang_vel / (angle / dt)
                    axis = axis / np.linalg.norm(axis)
                    from scipy.spatial.transform import Rotation as _R
                    dq = _R.from_rotvec(axis * angle).as_quat()  # [x,y,z,w]
                    # Convert to [w,x,y,z] and apply
                    dq_wxyz = np.array([dq[3], dq[0], dq[1], dq[2]])
                    self.quat = self._quat_mul(self.quat, dq_wxyz)
                    self.quat /= np.linalg.norm(self.quat)

            def update(self, meas_pos: np.ndarray, meas_quat: np.ndarray, t: float):
                """Incorporate a new measurement."""
                if not self.initialized:
                    self.x[:3] = meas_pos
                    self.x[3:] = 0.0
                    self.quat  = meas_quat.copy()
                    self.ang_vel = np.zeros(3)
                    self.last_time = t
                    self.initialized = True
                    return

                dt = t - self.last_time
                if dt < 1e-6:
                    return

                # Predict to measurement time
                self.predict(dt)

                # --- Position KF update ---
                z = meas_pos
                y = z - self.H @ self.x
                S = self.H @ self.P @ self.H.T + self.R
                K = self.P @ self.H.T @ np.linalg.inv(S)
                self.x = self.x + K @ y
                I_KH = np.eye(6) - K @ self.H
                self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

                # --- Orientation update (SLERP correction + angular velocity estimate) ---
                from scipy.spatial.transform import Rotation as _R
                r_pred = _R.from_quat([self.quat[1], self.quat[2], self.quat[3], self.quat[0]])
                r_meas = _R.from_quat([meas_quat[1], meas_quat[2], meas_quat[3], meas_quat[0]])

                # Correction: blend predicted toward measurement
                alpha_q = min(self._r_quat / (self._r_quat + self._q_quat), 0.95)
                from scipy.spatial.transform import Slerp as _Slerp
                slerp = _Slerp([0.0, 1.0], _R.concatenate([r_pred, r_meas]))
                r_corrected = slerp([1.0 - alpha_q])[0]
                qi = r_corrected.as_quat()  # [x,y,z,w]
                self.quat = np.array([qi[3], qi[0], qi[1], qi[2]])

                # Estimate angular velocity from measurement delta
                r_old = _R.from_quat([self.quat[1], self.quat[2], self.quat[3], self.quat[0]])
                delta_r = r_old.inv() * r_meas
                rotvec = delta_r.as_rotvec()
                new_ang_vel = rotvec / dt if dt > 1e-6 else np.zeros(3)
                # Smooth angular velocity estimate (EMA)
                self.ang_vel = 0.6 * self.ang_vel + 0.4 * new_ang_vel

                self.last_time = t

            def get_state(self):
                """Return current (pos, quat) estimate."""
                return self.x[:3].copy(), self.quat.copy()

            @staticmethod
            def _quat_mul(q1, q2):
                """Hamilton product: q1 * q2, both in [w,x,y,z] convention."""
                w1, x1, y1, z1 = q1
                w2, x2, y2, z2 = q2
                return np.array([
                    w1*w2 - x1*x2 - y1*y2 - z1*z2,
                    w1*x2 + x1*w2 + y1*z2 - z1*y2,
                    w1*y2 - x1*z2 + y1*w2 + z1*x2,
                    w1*z2 + x1*y2 - y1*x2 + z1*w2,
                ])

        def _init_ekfs(self):
            """Lazily create EKF instances for each bone."""
            if not hasattr(self, "_ekf_femur"):
                self._ekf_femur = self._BoneEKF()
                self._ekf_tibia = self._BoneEKF()
                self._ekf_last_predict = None

        def _raw_tf_callback(self, msg: _TFMessage):
            """Direct /tf subscription — feeds measurements into EKF.

            Every incoming TF message is accepted unconditionally, which means
            bag restarts work without any special detection or buffer flushing.
            """
            import time as _wtime
            now_wall = _wtime.time()
            self._init_ekfs()

            for tf_stamped in msg.transforms:
                child = tf_stamped.child_frame_id
                if child == self._tf_child_femur:
                    ekf = self._ekf_femur
                    bone = "femur"
                elif child == self._tf_child_tibia:
                    ekf = self._ekf_tibia
                    bone = "tibia"
                else:
                    continue

                t = tf_stamped.transform.translation
                r = tf_stamped.transform.rotation
                meas_pos  = np.array([t.x, t.y, t.z], dtype=np.float64)
                meas_quat = np.array([r.w, r.x, r.y, r.z], dtype=np.float64)

                # Detect bag restart: large position jump → reinitialize
                if ekf.initialized:
                    jump = float(np.linalg.norm(meas_pos - ekf.x[:3]))
                    if jump > 0.5:  # 50 cm jump = bag restarted
                        ekf.initialized = False
                        print(f"[EKF] {bone} position jumped {jump:.2f}m — reinitializing")

                ekf.update(meas_pos, meas_quat, now_wall)

                cnt_attr = f"_{bone}_origin_count"
                cnt = getattr(self, cnt_attr, 0) + 1
                setattr(self, cnt_attr, cnt)
                if cnt <= 3 or cnt % 200 == 0:
                    print(f"[EKF/TF] {bone} meas #{cnt}: "
                          f"pos=({meas_pos[0]:.3f},{meas_pos[1]:.3f},{meas_pos[2]:.3f})")

        def lookup_bone_tfs(self):
            """Predict EKF state forward to current wall time.

            Called from the main sim loop at ~60-120 Hz.  Between TF
            measurements (~20 Hz), the EKF's constant-velocity model
            and angular-velocity predictor fill in the gaps, producing
            smooth 30+ Hz output.
            """
            import time as _wtime
            self._init_ekfs()

            now_wall = _wtime.time()
            last_pred = self._ekf_last_predict or now_wall
            dt = now_wall - last_pred
            self._ekf_last_predict = now_wall

            for bone, ekf in [("femur", self._ekf_femur), ("tibia", self._ekf_tibia)]:
                if not ekf.initialized:
                    continue

                ekf.predict(dt)
                pos, quat = ekf.get_state()

                if bone == "femur":
                    self.femur_origin_pos     = pos
                    self.femur_origin_quat    = quat
                    self.femur_origin_updated = True
                else:
                    self.tibia_origin_pos     = pos
                    self.tibia_origin_quat    = quat
                    self.tibia_origin_updated = True

        def femur_origin_callback(self, msg):
            self._femur_origin_count += 1
            p = msg.pose.position
            q = msg.pose.orientation
            self.femur_origin_pos = np.array([p.x, p.y, p.z], dtype=np.float64)
            self.femur_origin_quat = np.array([q.w, q.x, q.y, q.z], dtype=np.float64)
            self.femur_origin_updated = True
            if self._femur_origin_count <= 3 or self._femur_origin_count % 200 == 0:
                print(f"[ROS] /tracked/femur_origin #{self._femur_origin_count}: "
                      f"pos=({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) "
                      f"quat=({q.w:.3f}, {q.x:.3f}, {q.y:.3f}, {q.z:.3f})")

        def tibia_origin_callback(self, msg):
            self._tibia_origin_count += 1
            p = msg.pose.position
            q = msg.pose.orientation
            self.tibia_origin_pos = np.array([p.x, p.y, p.z], dtype=np.float64)
            self.tibia_origin_quat = np.array([q.w, q.x, q.y, q.z], dtype=np.float64)
            self.tibia_origin_updated = True
            if self._tibia_origin_count <= 3 or self._tibia_origin_count % 200 == 0:
                print(f"[ROS] /tracked/tibia_origin #{self._tibia_origin_count}: "
                      f"pos=({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) "
                      f"quat=({q.w:.3f}, {q.x:.3f}, {q.y:.3f}, {q.z:.3f})")

        def robot_desc_callback(self, msg: String):
            self.robot_description = msg.data
            print(f"[ROS] /lbr/robot_description received ({len(msg.data)} bytes)")

        def joint_state_callback(self, msg: JointState):
            self._js_msg_count += 1
            try:
                self.joint_names = list(msg.name)
                self.joint_positions = torch.tensor(
                    list(msg.position), device=self.device, dtype=torch.float32
                )
                if msg.velocity:
                    self.joint_velocities = torch.tensor(
                        list(msg.velocity), device=self.device, dtype=torch.float32
                    )
                self.robot_state_updated = True
                if self._js_msg_count <= 3 or self._js_msg_count % 500 == 0:
                    pos_str = ", ".join(f"{p:.3f}" for p in list(msg.position)[:4])
                    print(f"[ROS] /lbr/joint_states msg #{self._js_msg_count}: "
                          f"joints={self.joint_names[:4]}... pos=[{pos_str}...]")
            except Exception as e:
                print(f"[ROS] Joint state error: {e}")

        def arm_camera_callback(self, msg: Image):
            self._cam_msg_count += 1
            try:
                h, w = msg.height, msg.width
                encoding = msg.encoding.lower()

                if encoding in ("rgb8",):
                    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
                elif encoding in ("bgr8",):
                    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
                    img = raw[:, :, ::-1].copy()  # BGR → RGB
                elif encoding in ("rgba8",):
                    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
                elif encoding in ("bgra8",):
                    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 4)
                    img = raw[:, :, 2::-1].copy()  # BGRA → RGB
                elif encoding in ("mono8",):
                    gray = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)
                    img = np.stack([gray, gray, gray], axis=-1)
                else:
                    if self._cam_msg_count <= 3:
                        print(f"[ROS] /camera_arm unsupported encoding: {encoding}")
                    return

                self.arm_camera_frame = img
                self.arm_camera_updated = True
                if self._cam_msg_count <= 2:
                    print(f"[ROS] /camera_arm msg #{self._cam_msg_count}: {w}x{h} {encoding}")
            except Exception as e:
                if self._cam_msg_count <= 5:
                    print(f"[ROS] /camera_arm error: {e}")

    return RosIntegrationNode()
