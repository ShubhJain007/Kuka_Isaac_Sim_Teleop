"""
Muse stylus input handler with EKF smoothing and hold-to-lock buttons.

Encapsulates all stylus state: pose filtering, button edge detection,
hold timing, and tip position extraction.
"""

from __future__ import annotations

import time
import numpy as np
from pathlib import Path
import sys

# PoseEKF lives in ir_tracking/scripts
_IR_SCRIPTS = Path(__file__).resolve().parent.parent / "ir_tracking" / "scripts"
if str(_IR_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_IR_SCRIPTS))

try:
    from pose_ekf import PoseEKF
except ImportError:
    PoseEKF = None


STYLUS_TIP_OFFSET = 0.045  # metres from marker to physical tip
HOLD_SECONDS = 2.0         # seconds to hold button before trigger


class StylusInput:
    """Processes Muse stylus data into filtered pose + button events."""

    def __init__(self):
        if PoseEKF is None:
            raise ImportError("PoseEKF not found — check ir_tracking/scripts/pose_ekf.py")

        self.ekf = PoseEKF(
            max_misses=30,
            q_pos=1.0e-6,
            q_vel=1.0e-2,
            q_ang=1.0e-3,
            q_angvel=1.0,
            r_pos=2.5e-5,
            r_ang=3.0e-4,
        )

        # Output state
        self.tip_pos: np.ndarray | None = None       # xyz of physical tip
        self.pose_matrix: np.ndarray | None = None    # filtered 4x4
        self.euler_deg: np.ndarray | None = None      # xyz euler in degrees
        self.active = False

        # Button hold tracking
        self._prim_since: float | None = None
        self._sec_since: float | None = None
        self._prim_fired = False
        self._sec_fired = False

        # Events (read and clear each frame)
        self.lock_triggered = False
        self.unlock_triggered = False

    def update(self, stylus_data, now: float | None = None):
        """Process one frame of stylus data from avp_stream.get_stylus().

        Args:
            stylus_data: dict or object with 'pose', 'primary_pressure', 'secondary_pressure'
            now: monotonic time (defaults to time.time())
        """
        self.lock_triggered = False
        self.unlock_triggered = False
        self.active = False

        if stylus_data is None:
            return

        now = now or time.time()

        pose = _get(stylus_data, "pose", "Pose", "transform")
        primary = float(_get(stylus_data, "primary_pressure", default=0.0)) > 0.5
        secondary = float(_get(stylus_data, "secondary_pressure", default=0.0)) > 0.5

        if pose is None:
            return

        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (4, 4):
            return

        # EKF smooth
        filtered, ok = self.ekf.process(pose, True, now)
        if ok:
            pose = np.asarray(filtered, dtype=np.float64)

        self.active = True
        self.pose_matrix = pose

        # Tip position: offset along local -Z
        tip_local = np.array([0.0, 0.0, -STYLUS_TIP_OFFSET, 1.0])
        tip_world = pose @ tip_local
        self.tip_pos = tip_world[:3]

        # Euler angles from rotation matrix
        from scipy.spatial.transform import Rotation as R
        self.euler_deg = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)

        # Button hold logic
        self._process_button(primary, is_primary=True, now=now)
        self._process_button(secondary, is_primary=False, now=now)

    def _process_button(self, pressed: bool, is_primary: bool, now: float):
        if is_primary:
            if pressed:
                if self._prim_since is None:
                    self._prim_since = now
                if not self._prim_fired and (now - self._prim_since) >= HOLD_SECONDS:
                    self.lock_triggered = True
                    self._prim_fired = True
            else:
                self._prim_since = None
                self._prim_fired = False
        else:
            if pressed:
                if self._sec_since is None:
                    self._sec_since = now
                if not self._sec_fired and (now - self._sec_since) >= HOLD_SECONDS:
                    self.unlock_triggered = True
                    self._sec_fired = True
            else:
                self._sec_since = None
                self._sec_fired = False


def _get(obj, *keys, default=None):
    """Access dict or object attribute by multiple possible key names."""
    if obj is None:
        return default
    for k in keys:
        if isinstance(obj, dict):
            if k in obj:
                return obj[k]
        else:
            v = getattr(obj, k, None)
            if v is not None:
                return v
    return default
