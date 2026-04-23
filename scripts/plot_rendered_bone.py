"""Scatter plot of the rendered femur/tibia USD poses captured by teleop.

Reads `scripts/rendered_bones.csv` (produced by running
`teleop_med7_vr.py --log-render-bones 15`) and writes
`scripts/rendered_bone_plot.png` — a 3x2 scatter grid:

    row 0:  X (mm)  |  Roll  (deg)
    row 1:  Y (mm)  |  Pitch (deg)
    row 2:  Z (mm)  |  Yaw   (deg)

No interpolation, no connecting lines — each render frame is one dot.
Femur in red, tibia in blue.

Standalone (no ROS required — just a CSV reader):
    python3 scripts/plot_rendered_bone.py

CLI:
    --csv PATH      Input CSV path (default scripts/rendered_bones.csv)
    --out PATH      Output image path (default scripts/rendered_bone_plot.png)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {"femur": "tab:red", "tibia": "tab:blue"}


def _quat_to_euler_deg(q: np.ndarray) -> np.ndarray:
    """(N,4) w,x,y,z → (N,3) roll/pitch/yaw in degrees. No unwrap."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll  = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw   = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.degrees(np.stack([roll, pitch, yaw], axis=1))


def _load_csv(path: str):
    if not os.path.exists(path):
        print(f"[ERROR] CSV not found: {path}")
        print("Generate it first by running:")
        print("  ~/IsaacLab/_isaac_sim/python.sh scripts/teleop_med7_vr.py "
              "--log-render-bones 15 [...]")
        sys.exit(1)

    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        for r in reader:
            if not r:
                continue
            try:
                rows.append([float(v) for v in r])
            except ValueError:
                continue
    if not rows:
        print(f"[ERROR] CSV empty: {path}")
        sys.exit(1)
    A = np.array(rows, dtype=np.float64)
    return A  # columns: t, fx,fy,fz,fqw,fqx,fqy,fqz, tx,ty,tz,tqw,tqx,tqy,tqz


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "rendered_bones.csv"))
    ap.add_argument("--out",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "rendered_bone_plot.png"))
    args = ap.parse_args()

    A = _load_csv(args.csv)
    t       = A[:, 0]
    fem_pos = A[:, 1:4]  * 1000.0
    fem_q   = A[:, 4:8]
    tib_pos = A[:, 8:11] * 1000.0
    tib_q   = A[:, 11:15]
    fem_eul = _quat_to_euler_deg(fem_q)
    tib_eul = _quat_to_euler_deg(tib_q)

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    pos_labels = ["X (mm)", "Y (mm)", "Z (mm)"]
    eul_labels = ["Roll (deg)", "Pitch (deg)", "Yaw (deg)"]

    for i in range(3):
        ax_p = axes[i, 0]
        ax_p.scatter(t, fem_pos[:, i], s=5, color=COLORS["femur"], alpha=0.85, label="femur")
        ax_p.scatter(t, tib_pos[:, i], s=5, color=COLORS["tibia"], alpha=0.85, label="tibia")
        ax_p.set_ylabel(pos_labels[i])
        ax_p.grid(True, linestyle=":", alpha=0.5)
        if i == 0:
            ax_p.legend(loc="upper right", fontsize=9)

        ax_e = axes[i, 1]
        ax_e.scatter(t, fem_eul[:, i], s=5, color=COLORS["femur"], alpha=0.85, label="femur")
        ax_e.scatter(t, tib_eul[:, i], s=5, color=COLORS["tibia"], alpha=0.85, label="tibia")
        ax_e.set_ylabel(eul_labels[i])
        ax_e.grid(True, linestyle=":", alpha=0.5)
        if i == 0:
            ax_e.legend(loc="upper right", fontsize=9)

    axes[2, 0].set_xlabel("time (s, from recording start)")
    axes[2, 1].set_xlabel("time (s, from recording start)")
    fig.suptitle(
        f"Rendered bone poses (Isaac Sim USD) — {len(t)} frames, "
        f"duration {t[-1] - t[0]:.1f}s",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"[INFO] Plot saved: {args.out}")


if __name__ == "__main__":
    main()
