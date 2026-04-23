"""Plot femur / tibia TF (position + orientation) over a fixed capture window.

Subscribes to /tf, captures 'tracked/femur_origin' and 'tracked/tibia_origin'
transforms for N seconds, then renders a 3×2 grid of subplots:

    row 0:  X position       |  Roll  (deg)
    row 1:  Y position       |  Pitch (deg)
    row 2:  Z position       |  Yaw   (deg)

Each subplot overlays femur and tibia on a shared time axis. The PNG is
saved next to the script (default: bone_tf_plot.png).

Run with the Isaac Sim bundled Python (rclpy lives inside the
isaacsim.ros2.bridge/humble tree):

    ~/IsaacLab/_isaac_sim/python.sh scripts/plot_bone_tf.py --duration 15

CLI:
    --duration SECS   Seconds to capture (default 15)
    --out PATH        Output image path (default scripts/bone_tf_plot.png)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np


# ---- rclpy bootstrap (MUST run before any rclpy import) --------------
def _setup_ros2_libs() -> bool:
    try:
        import isaacsim
    except ImportError:
        return False
    isaacsim_path = os.path.dirname(isaacsim.__file__)
    ros2_bridge_root = os.path.join(isaacsim_path, "exts/isaacsim.ros2.bridge/humble")
    lib_path = os.path.join(ros2_bridge_root, "lib")
    python_path = os.path.join(ros2_bridge_root, "rclpy")

    sys.path = [p for p in sys.path
                if "dist-packages/rclpy" not in p
                and "site-packages/rclpy" not in p
                and "/opt/ros/" not in p]
    if python_path not in sys.path:
        sys.path.insert(0, python_path)

    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_path not in current_ld:
        os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{current_ld}"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    return True


if not _setup_ros2_libs():
    print("[ERROR] Could not locate Isaac Sim's bundled ROS 2 bridge.")
    print("Run with the Isaac Sim python, e.g.:")
    print("  ~/IsaacLab/_isaac_sim/python.sh scripts/plot_bone_tf.py")
    sys.exit(1)

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from tf2_msgs.msg import TFMessage
except ImportError as e:
    print(f"[ERROR] rclpy still not importable after bootstrap: {e}")
    sys.exit(1)

# Headless-safe backend (so it works even when no X server is wired up)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BONES = {
    "femur": "tracked/femur_origin",
    "tibia": "tracked/tibia_origin",
}

# Color convention — kept consistent across all subplots.
COLORS = {"femur": "tab:red", "tibia": "tab:blue"}


class TfPlotCollector(Node):
    def __init__(self):
        super().__init__("bone_tf_plot_collector")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
        )
        self.create_subscription(TFMessage, "/tf",        self._tf_cb, qos)
        self.create_subscription(TFMessage, "/tf_static", self._tf_cb, qos)
        self._samples: dict[str, list] = defaultdict(list)
        self._child_to_bone = {v: k for k, v in BONES.items()}

    def _tf_cb(self, msg: TFMessage):
        now_wall = time.time()
        for tf in msg.transforms:
            bone = self._child_to_bone.get(tf.child_frame_id)
            if bone is None:
                continue
            t = tf.transform.translation
            r = tf.transform.rotation
            stamp_sec = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
            self._samples[bone].append((
                now_wall, stamp_sec,
                t.x, t.y, t.z,
                r.w, r.x, r.y, r.z,
            ))


def _quat_to_euler_deg(q: np.ndarray) -> np.ndarray:
    """Convert an (N,4) quaternion array (w,x,y,z) to (N,3) Euler XYZ degrees.

    Returns (roll_x, pitch_y, yaw_z) in degrees. Uses the standard ZYX extraction
    (aerospace convention); reasonable for visualising small angles.
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # Roll (X)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # Pitch (Y) — clamp to avoid gimbal lock nan
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # Yaw (Z)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.degrees(np.stack([roll, pitch, yaw], axis=1))


def _extract(rows: list, t0: float):
    if len(rows) == 0:
        return None
    A = np.array(rows, dtype=np.float64)
    t    = A[:, 1] - t0      # seconds since capture start (header stamp)
    pos  = A[:, 2:5] * 1000  # mm for readability
    quat = A[:, 5:9]         # w,x,y,z
    eul  = _quat_to_euler_deg(quat)
    return {"t": t, "pos": pos, "eul": eul}


def _plot_bone(ax, data, label: str, axis_idx: int, kind: str):
    if data is None:
        return
    y = data["pos"][:, axis_idx] if kind == "pos" else data["eul"][:, axis_idx]
    ax.scatter(data["t"], y, label=label, color=COLORS[label], s=6, alpha=0.8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=float, default=15.0,
                    help="Seconds to capture (default 15)")
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "bone_tf_plot.png"),
                    help="Output image path")
    args = ap.parse_args()

    rclpy.init()
    node = TfPlotCollector()
    print(f"[INFO] Capturing TF for {args.duration:.1f}s "
          f"(children: {list(BONES.values())})...")
    t_end = time.time() + args.duration
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.05)

    counts = {b: len(node._samples[b]) for b in BONES}
    print(f"[INFO] Captured: femur={counts['femur']}  tibia={counts['tibia']}")

    # Shared time origin = earliest header stamp across both bones (so plots align).
    earliest = np.inf
    for b in BONES:
        if node._samples[b]:
            earliest = min(earliest, node._samples[b][0][1])
    if np.isinf(earliest):
        print("[ERROR] No TF captured — is the bag playing or the tracker running?")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(2)

    femur = _extract(node._samples["femur"], earliest)
    tibia = _extract(node._samples["tibia"], earliest)

    node.destroy_node()
    rclpy.shutdown()

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    pos_labels = ["X (mm)", "Y (mm)", "Z (mm)"]
    eul_labels = ["Roll (deg)", "Pitch (deg)", "Yaw (deg)"]

    for i in range(3):
        ax_pos = axes[i, 0]
        _plot_bone(ax_pos, femur, "femur", i, "pos")
        _plot_bone(ax_pos, tibia, "tibia", i, "pos")
        ax_pos.set_ylabel(pos_labels[i])
        ax_pos.grid(True, linestyle=":", alpha=0.5)
        if i == 0:
            ax_pos.legend(loc="upper right", fontsize=9)

        ax_eul = axes[i, 1]
        _plot_bone(ax_eul, femur, "femur", i, "eul")
        _plot_bone(ax_eul, tibia, "tibia", i, "eul")
        ax_eul.set_ylabel(eul_labels[i])
        ax_eul.grid(True, linestyle=":", alpha=0.5)
        if i == 0:
            ax_eul.legend(loc="upper right", fontsize=9)

    axes[2, 0].set_xlabel("time (s, from first TF)")
    axes[2, 1].set_xlabel("time (s, from first TF)")
    fig.suptitle(
        f"Femur vs Tibia TF — {args.duration:.0f}s capture  "
        f"(femur={counts['femur']} samples, tibia={counts['tibia']} samples)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"[INFO] Plot saved: {args.out}")


if __name__ == "__main__":
    main()
