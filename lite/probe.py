"""
Surgical probe state machine: lock, unlock, bone attachment.

Manages the probe position/orientation, guide line, and rigid
attachment to the nearest bone when locked.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


class ProbeState:
    """Tracks probe position, locking, and bone attachment."""

    def __init__(self):
        self.pos = np.array([-0.263, 0.181, 0.449])
        self.quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])

        self.locked = False
        self.line_depth = 0.20
        self.line_pos: np.ndarray | None = None
        self.line_quat: np.ndarray | None = None

        # Bone attachment
        self._bone_name: str | None = None
        self._bone_offset: np.ndarray | None = None    # position in bone-local frame
        self._bone_quat_off: np.ndarray | None = None   # orientation in bone-local frame

    def set_pose(self, pos: np.ndarray, euler_deg: np.ndarray | None = None,
                 quat_wxyz: np.ndarray | None = None):
        """Update probe pose (only when unlocked)."""
        if self.locked:
            return
        self.pos = np.asarray(pos, dtype=np.float64)
        if quat_wxyz is not None:
            self.quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
        elif euler_deg is not None:
            self.quat_wxyz = _euler_to_quat(euler_deg)

    def lock(self, bone_poses: dict[str, tuple[np.ndarray, np.ndarray]]):
        """Lock probe at current position and attach to nearest bone.

        Args:
            bone_poses: {"femur": (pos_3, quat_wxyz_4), "tibia": ...}
        """
        self.locked = True

        # Find nearest bone
        best_name, best_dist = None, float("inf")
        for name, (bp, bq) in bone_poses.items():
            if bp is None:
                continue
            d = float(np.linalg.norm(self.pos - bp))
            if d < best_dist:
                best_dist = d
                best_name = name

        if best_name is None:
            print("[PROBE] Locked (no bones available for attachment)")
            return

        bp, bq = bone_poses[best_name]
        off_world = self.pos - bp
        self._bone_offset = _qrot(_qconj(bq), off_world)
        self._bone_quat_off = _qmul(_qconj(bq), self.quat_wxyz)
        self._bone_name = best_name
        print(f"[PROBE] Locked → attached to {best_name} ({best_dist*1000:.1f}mm)")

    def unlock(self):
        """Unlock probe and detach from bone."""
        if self._bone_name:
            print(f"[PROBE] Unlocked (detached from {self._bone_name})")
        else:
            print("[PROBE] Unlocked")
        self.locked = False
        self._bone_name = None
        self._bone_offset = None
        self._bone_quat_off = None
        self.line_pos = None
        self.line_quat = None

    def follow_bone(self, bone_poses: dict[str, tuple[np.ndarray, np.ndarray]]):
        """If attached to a bone, update probe world pose from bone motion."""
        if not self.locked or self._bone_name is None:
            return
        bp, bq = bone_poses.get(self._bone_name, (None, None))
        if bp is None:
            return
        self.pos = bp + _qrot(bq, self._bone_offset)
        self.quat_wxyz = _qmul(bq, self._bone_quat_off)


# ── Quaternion helpers (w,x,y,z convention) ──────────────────────────────

def _qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])

def _qrot(q, v):
    w, x, y, z = q
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    return np.array([
        (1 - 2*(y*y + z*z))*vx + 2*(x*y - z*w)*vy + 2*(x*z + y*w)*vz,
        2*(x*y + z*w)*vx + (1 - 2*(x*x + z*z))*vy + 2*(y*z - x*w)*vz,
        2*(x*z - y*w)*vx + 2*(y*z + x*w)*vy + (1 - 2*(x*x + y*y))*vz,
    ])

def _euler_to_quat(euler_deg):
    r = R.from_euler("xyz", euler_deg, degrees=True)
    q_xyzw = r.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
