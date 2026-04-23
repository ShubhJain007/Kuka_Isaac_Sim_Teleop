"""Per-frame write-vs-readback diagnostic plot for femur/tibia USD prims.

Reads `scripts/rendered_bones.csv` (produced by
`teleop_med7_vr.py --log-render-bones 15`, extended schema with
per-frame diagnostic columns) and writes
`scripts/bone_write_diag_plot.png`.

Per bone, four aligned subplots (femur on the left column, tibia on
the right) share the same time axis so you can read top-to-bottom per
frame:

    row 0: flag_<bone>              (0/1 — was TF-arrival flag True this frame?)
    row 1: wrote_<bone>             (0/1 — did MakeMatrixXform().Set() execute?)
    row 2: wrote_<bone>_x (m)       (pose sent into Set(), NaN if not written)
    row 3: readback <bone>_x (m)    (XformCache.GetLocalToWorldTransform after write)

Standalone (no ROS, no Isaac Sim — just a CSV reader):
    python3 scripts/plot_bone_write_diag.py

CLI:
    --csv PATH   Input CSV (default scripts/rendered_bones.csv)
    --out PATH   Output PNG  (default scripts/bone_write_diag_plot.png)
    --axis x|y|z  Which position axis to plot for rows 2/3 (default x)
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


# Column indices in the extended rendered_bones.csv schema
COL = {
    "t":            0,
    "fx":           1, "fy":           2, "fz":           3,
    "fqw":          4, "fqx":          5, "fqy":          6, "fqz":          7,
    "tx":           8, "ty":           9, "tz":          10,
    "tqw":         11, "tqx":         12, "tqy":         13, "tqz":         14,
    "flag_fem":    15, "flag_tib":    16,
    "wrote_fem":   17, "wrote_tib":   18,
    "wrote_fem_x": 19, "wrote_fem_y": 20, "wrote_fem_z": 21,
    "wrote_fem_qw":22, "wrote_fem_qx":23, "wrote_fem_qy":24, "wrote_fem_qz":25,
    "wrote_tib_x": 26, "wrote_tib_y": 27, "wrote_tib_z": 28,
    "wrote_tib_qw":29, "wrote_tib_qx":30, "wrote_tib_qy":31, "wrote_tib_qz":32,
}


def _load(csv_path: str) -> np.ndarray:
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        sys.exit(1)
    rows = []
    with open(csv_path, "r", newline="") as fh:
        rdr = csv.reader(fh)
        header = next(rdr, None)
        for r in rdr:
            if not r:
                continue
            try:
                rows.append([float(v) for v in r])
            except ValueError:
                continue
    if not rows:
        print(f"[ERROR] CSV empty: {csv_path}")
        sys.exit(1)
    A = np.array(rows, dtype=np.float64)
    if A.shape[1] < len(COL):
        print(f"[ERROR] CSV has {A.shape[1]} columns, expected {len(COL)}. "
              "Re-run teleop with the extended --log-render-bones recorder.")
        sys.exit(1)
    return A


def _stairs(ax, t, y, color, label):
    """Plot a 0/1 boolean-ish signal as a step trace for easy scanning."""
    ax.step(t, y, where="post", color=color, linewidth=0.9, alpha=0.8, label=label)
    ax.set_ylim(-0.15, 1.15)
    ax.set_yticks([0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "rendered_bones.csv"))
    ap.add_argument("--out",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "bone_write_diag_plot.png"))
    ap.add_argument("--axis", choices=["x", "y", "z"], default="x",
                    help="Which position axis to plot in rows 2 & 3")
    args = ap.parse_args()

    A = _load(args.csv)
    t = A[:, COL["t"]]

    axis = args.axis
    wrote_fem_col  = COL[f"wrote_fem_{axis}"]
    wrote_tib_col  = COL[f"wrote_tib_{axis}"]
    read_fem_col   = COL[f"f{axis}"]
    read_tib_col   = COL[f"t{axis}"]

    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)

    # Row 0 — TF-arrival flag
    _stairs(axes[0, 0], t, A[:, COL["flag_fem"]], "tab:red",  "flag_fem")
    _stairs(axes[0, 1], t, A[:, COL["flag_tib"]], "tab:blue", "flag_tib")
    axes[0, 0].set_title("FEMUR", fontsize=12)
    axes[0, 1].set_title("TIBIA", fontsize=12)
    axes[0, 0].set_ylabel("flag (TF arrived)")
    axes[0, 1].set_ylabel("flag (TF arrived)")

    # Row 1 — write actually executed
    _stairs(axes[1, 0], t, A[:, COL["wrote_fem"]], "tab:red",  "wrote_fem")
    _stairs(axes[1, 1], t, A[:, COL["wrote_tib"]], "tab:blue", "wrote_tib")
    axes[1, 0].set_ylabel("wrote (Set() ran)")
    axes[1, 1].set_ylabel("wrote (Set() ran)")

    # Row 2 — written pose value (NaN on non-write frames)
    axes[2, 0].scatter(t, A[:, wrote_fem_col], s=6, color="tab:red",  alpha=0.85)
    axes[2, 1].scatter(t, A[:, wrote_tib_col], s=6, color="tab:blue", alpha=0.85)
    axes[2, 0].set_ylabel(f"wrote_{axis} (m)")
    axes[2, 1].set_ylabel(f"wrote_{axis} (m)")
    for a in (axes[2, 0], axes[2, 1]):
        a.grid(True, linestyle=":", alpha=0.5)

    # Row 3 — read-back position every frame
    axes[3, 0].scatter(t, A[:, read_fem_col], s=6, color="tab:red",  alpha=0.85)
    axes[3, 1].scatter(t, A[:, read_tib_col], s=6, color="tab:blue", alpha=0.85)
    axes[3, 0].set_ylabel(f"readback_{axis} (m)")
    axes[3, 1].set_ylabel(f"readback_{axis} (m)")
    for a in (axes[3, 0], axes[3, 1]):
        a.grid(True, linestyle=":", alpha=0.5)

    for row in axes:
        for a in row:
            a.grid(True, linestyle=":", alpha=0.4)

    axes[3, 0].set_xlabel("time (s)")
    axes[3, 1].set_xlabel("time (s)")

    # Summary counts in the title
    n = len(t)
    nf_flag  = int(np.sum(A[:, COL["flag_fem"]]))
    nt_flag  = int(np.sum(A[:, COL["flag_tib"]]))
    nf_write = int(np.sum(A[:, COL["wrote_fem"]]))
    nt_write = int(np.sum(A[:, COL["wrote_tib"]]))
    fig.suptitle(
        f"Bone write-vs-readback diagnostic  "
        f"(axis={axis}, {n} frames, {t[-1]-t[0]:.1f}s)\n"
        f"femur: flag on {nf_flag}/{n} frames, wrote {nf_write}/{n}   |   "
        f"tibia: flag on {nt_flag}/{n} frames, wrote {nt_write}/{n}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"[INFO] Plot saved: {args.out}")


if __name__ == "__main__":
    main()
