"""Compare noise characteristics of femur vs tibia TF transforms.

Subscribes to /tf, captures transforms for child_frame_id
'tracked/femur_origin' and 'tracked/tibia_origin', then
reports side-by-side noise/jitter metrics.

Run with the Isaac Sim bundled Python (rclpy lives inside the
isaacsim.ros2.bridge/humble tree), e.g. via:

    ./scripts/run_teleop_ros.sh                       # sets LD_LIBRARY_PATH
    python scripts/analyze_bone_tf_noise.py           # then run this

CLI:
    --duration SECS     How long to collect (default 10)
    --stationary        Print extra 'position std' metrics assuming the
                        bone is held physically still during capture
                        (values then represent raw sensor noise).

Metrics printed:
  - Sample count, mean publish rate (Hz), dt mean/std/max (ms)
  - Position range (mm) per axis, total displacement (mm)
  - Position 'jerk' = ||d³x/dt³|| — high value = jittery even while moving
  - Consecutive quaternion angular step (deg) — std = rotational jitter
  - High-frequency power fraction (>5 Hz / >10 Hz) on position magnitude
  - (stationary mode) raw position std per axis in mm
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np


# ---- rclpy bootstrap (must run BEFORE any rclpy import) --------------
# Mirror med7_ros_bones.py::setup_ros2_libs: point sys.path at Isaac Sim's
# bundled rclpy and prepend LD_LIBRARY_PATH, re-execing once if needed.
def _setup_ros2_libs() -> bool:
    try:
        import isaacsim  # provided only by the Isaac Sim Python
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
    print("You must run this with the Isaac Sim python, e.g.:")
    print("  ~/IsaacLab/_isaac_sim/python.sh scripts/analyze_bone_tf_noise.py")
    print("or through the project's wrapper:")
    print("  ./scripts/run_teleop_ros.sh   # sets env")
    print("  then run this script with the same python used by that wrapper.")
    sys.exit(1)

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from tf2_msgs.msg import TFMessage
except ImportError as e:
    print(f"[ERROR] rclpy still not importable after bootstrap: {e}")
    sys.exit(1)


BONES = {
    "femur": "tracked/femur_origin",
    "tibia": "tracked/tibia_origin",
}


class TfNoiseCollector(Node):
    def __init__(self):
        super().__init__("bone_tf_noise_collector")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
        )
        self.create_subscription(TFMessage, "/tf", self._tf_cb, qos)
        # Also subscribe to /tf_static in case origins are published statically
        self.create_subscription(TFMessage, "/tf_static", self._tf_cb, qos)

        # Per-bone buffers: list of (wall_time, stamp_sec, x,y,z, qw,qx,qy,qz)
        self._samples: dict[str, list] = defaultdict(list)

        # Child->bone-name reverse map
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


def _analyze(name: str, rows: list, stationary: bool) -> dict:
    if len(rows) < 4:
        return {"bone": name, "ok": False, "n": len(rows)}

    A = np.array(rows, dtype=np.float64)
    wall   = A[:, 0]
    stamp  = A[:, 1]
    pos    = A[:, 2:5]    # (N,3)
    quat   = A[:, 5:9]    # (N,4) w,x,y,z

    # dt based on header stamps (use stamp rather than wall — network-RX
    # jitter can blur wall-time analysis for incoming topics).
    dt = np.diff(stamp)
    dt_valid = dt[dt > 0]
    rate = 1.0 / dt_valid.mean() if len(dt_valid) else float("nan")

    # Position range / displacement
    pmin = pos.min(axis=0)
    pmax = pos.max(axis=0)
    prange_mm = (pmax - pmin) * 1000.0
    disp_mm = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)) * 1000.0

    # Jerk (3rd derivative). Use simple finite differences vs dt
    # to catch jitter even when bone is moving.
    if len(pos) >= 4:
        vel = np.diff(pos, axis=0) / dt[:, None]
        acc = np.diff(vel, axis=0) / dt[1:, None]
        jrk = np.diff(acc, axis=0) / dt[2:, None]
        jerk_rms = float(np.sqrt(np.mean(np.sum(jrk * jrk, axis=1))))
    else:
        jerk_rms = float("nan")

    # Quaternion angular step (deg).
    # angle = 2 * acos(|dot(q1,q2)|)
    q1 = quat[:-1]
    q2 = quat[1:]
    dotp = np.clip(np.abs(np.einsum("ij,ij->i", q1, q2)), -1.0, 1.0)
    ang_deg = np.degrees(2.0 * np.arccos(dotp))
    ang_deg_mean = float(ang_deg.mean())
    ang_deg_std  = float(ang_deg.std())
    ang_deg_max  = float(ang_deg.max())

    # High-frequency power fraction on |position magnitude deviation|.
    # Detrend, then FFT. Compare power above 5/10 Hz to total.
    try:
        p_mag = np.linalg.norm(pos - pos.mean(axis=0), axis=1)
        # Resample to uniform grid using stamp time.
        t0 = stamp[0]
        t_uniform_dt = float(np.median(dt_valid)) if len(dt_valid) else 1.0 / 30.0
        n = len(p_mag)
        T = stamp[-1] - t0
        n_uniform = max(16, int(T / t_uniform_dt))
        t_uniform = np.linspace(0.0, T, n_uniform)
        p_uniform = np.interp(t_uniform, stamp - t0, p_mag)
        fft_mag = np.abs(np.fft.rfft(p_uniform - p_uniform.mean()))
        freqs = np.fft.rfftfreq(n_uniform, d=t_uniform_dt)
        total_power = float(np.sum(fft_mag ** 2))
        if total_power > 1e-12:
            hf5_frac  = float(np.sum(fft_mag[freqs >= 5.0] ** 2) / total_power)
            hf10_frac = float(np.sum(fft_mag[freqs >= 10.0] ** 2) / total_power)
        else:
            hf5_frac = hf10_frac = 0.0
    except Exception:
        hf5_frac = hf10_frac = float("nan")

    out = {
        "bone":       name,
        "ok":         True,
        "n":          len(rows),
        "rate_hz":    rate,
        "dt_mean_ms": float(dt_valid.mean() * 1000.0) if len(dt_valid) else float("nan"),
        "dt_std_ms":  float(dt_valid.std()  * 1000.0) if len(dt_valid) else float("nan"),
        "dt_max_ms":  float(dt_valid.max()  * 1000.0) if len(dt_valid) else float("nan"),
        "range_mm":   prange_mm,
        "disp_mm":    disp_mm,
        "jerk_rms":   jerk_rms,
        "ang_mean":   ang_deg_mean,
        "ang_std":    ang_deg_std,
        "ang_max":    ang_deg_max,
        "hf5_frac":   hf5_frac,
        "hf10_frac":  hf10_frac,
    }
    if stationary:
        out["pos_std_mm"] = (pos.std(axis=0) * 1000.0)
    return out


def _print_row(label: str, femur_val: str, tibia_val: str):
    print(f"  {label:<28} {femur_val:>18}   {tibia_val:>18}")


def _fmt(v, fmt=".3f"):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "n/a"
    return format(v, fmt)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=float, default=10.0,
                    help="Seconds to collect (default 10)")
    ap.add_argument("--stationary", action="store_true",
                    help="Assume bone is held still — print raw position std")
    args = ap.parse_args()

    rclpy.init()
    node = TfNoiseCollector()
    print(f"[INFO] Collecting TF for {args.duration:.1f}s "
          f"(child_frame_ids: {list(BONES.values())})...")
    if args.stationary:
        print("[INFO] Stationary mode — hold both bones physically still")

    t_end = time.time() + args.duration
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.05)

    counts = {b: len(node._samples[b]) for b in BONES}
    print(f"[INFO] Collected: femur={counts['femur']}  tibia={counts['tibia']}")

    femur = _analyze("femur", node._samples["femur"], args.stationary)
    tibia = _analyze("tibia", node._samples["tibia"], args.stationary)

    node.destroy_node()
    rclpy.shutdown()

    print("\n" + "=" * 68)
    print(f"{'METRIC':<30} {'FEMUR':>18}   {'TIBIA':>18}")
    print("-" * 68)

    if not femur["ok"] or not tibia["ok"]:
        print(f"  insufficient samples:  femur n={femur['n']}  tibia n={tibia['n']}")
        return

    _print_row("samples",       str(femur["n"]),                     str(tibia["n"]))
    _print_row("rate (Hz)",     _fmt(femur["rate_hz"],    ".2f"),    _fmt(tibia["rate_hz"],    ".2f"))
    _print_row("dt mean (ms)",  _fmt(femur["dt_mean_ms"], ".2f"),    _fmt(tibia["dt_mean_ms"], ".2f"))
    _print_row("dt std (ms)",   _fmt(femur["dt_std_ms"],  ".2f"),    _fmt(tibia["dt_std_ms"],  ".2f"))
    _print_row("dt max (ms)",   _fmt(femur["dt_max_ms"],  ".2f"),    _fmt(tibia["dt_max_ms"],  ".2f"))
    _print_row("range X/Y/Z (mm)",
               f"{femur['range_mm'][0]:.1f}/{femur['range_mm'][1]:.1f}/{femur['range_mm'][2]:.1f}",
               f"{tibia['range_mm'][0]:.1f}/{tibia['range_mm'][1]:.1f}/{tibia['range_mm'][2]:.1f}")
    _print_row("total path (mm)",
               _fmt(femur["disp_mm"], ".1f"), _fmt(tibia["disp_mm"], ".1f"))
    _print_row("jerk RMS (m/s³)",
               _fmt(femur["jerk_rms"], ".2e"), _fmt(tibia["jerk_rms"], ".2e"))
    _print_row("ang step mean (deg)",
               _fmt(femur["ang_mean"], ".3f"), _fmt(tibia["ang_mean"], ".3f"))
    _print_row("ang step std (deg)",
               _fmt(femur["ang_std"],  ".3f"), _fmt(tibia["ang_std"],  ".3f"))
    _print_row("ang step max (deg)",
               _fmt(femur["ang_max"],  ".3f"), _fmt(tibia["ang_max"],  ".3f"))
    _print_row("HF pwr frac ≥5 Hz",
               _fmt(femur["hf5_frac"],  ".3f"), _fmt(tibia["hf5_frac"],  ".3f"))
    _print_row("HF pwr frac ≥10 Hz",
               _fmt(femur["hf10_frac"], ".3f"), _fmt(tibia["hf10_frac"], ".3f"))
    if args.stationary:
        _print_row("pos std X/Y/Z (mm)",
                   f"{femur['pos_std_mm'][0]:.3f}/{femur['pos_std_mm'][1]:.3f}/{femur['pos_std_mm'][2]:.3f}",
                   f"{tibia['pos_std_mm'][0]:.3f}/{tibia['pos_std_mm'][1]:.3f}/{tibia['pos_std_mm'][2]:.3f}")
    print("=" * 68)

    # Verdict hint
    print("\nInterpretation hints:")
    print("  - Higher 'jerk RMS' on femur = TF itself has 3rd-derivative noise")
    print("    (would cause visible jumping).")
    print("  - Higher 'ang step std' on femur = rotational noise on femur TF.")
    print("  - Higher 'HF pwr frac' on femur = high-frequency chatter source.")
    print("  - If all three are similar between bones, the TF stream is NOT the")
    print("    source; the jitter is downstream (USD / Fabric / render).")


if __name__ == "__main__":
    main()
