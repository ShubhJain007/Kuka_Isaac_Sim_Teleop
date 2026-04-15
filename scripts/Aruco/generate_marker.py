"""
Generate a 5x5 ArUco marker for the surgical stylus pointer.

Usage:
    python generate_marker.py [--id ID] [--size SIZE_CM] [--dict DICT]

Outputs:
    aruco_5x5_id<ID>.png   — ready to print at SIZE_CM × SIZE_CM

Physical print instructions:
    - Print on matte white paper (NOT glossy — reflections kill tracking)
    - Print at exactly the physical size specified (default 4.5 cm × 4.5 cm)
    - Use 600 dpi or higher
    - Cut with ~2 mm extra white border around the edges
    - Mount flat and rigid on the stylus — any flex causes pose jitter

Stylus mounting guide:
    - Mount on a flat face at the TOP of the stylus
    - Tilt ~15° toward the user so AVP cameras see it during natural grip
    - The TIP is the pointer end, offset from marker center by a known
      fixed distance (see TIP_OFFSET_M in teleop_med7_vr.py)

AVP app setup:
    - Open Tracking Streamer on AVP
    - Settings → Marker Detection → Enable → select DICT_5X5_50
    - No code changes needed in the VisionOS app
"""

import argparse
import os
import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("opencv-python not installed. Run: pip install opencv-python")


# ── Supported dictionaries ────────────────────────────────────────────────────
DICT_MAP = {
    "4x4_50":   cv2.aruco.DICT_4X4_50,
    "4x4_100":  cv2.aruco.DICT_4X4_100,
    "5x5_50":   cv2.aruco.DICT_5X5_50,   # recommended for stylus
    "5x5_100":  cv2.aruco.DICT_5X5_100,
    "6x6_50":   cv2.aruco.DICT_6X6_50,
}

# Cells per side for each dictionary (inner bits + 2 border cells)
CELLS_MAP = {
    "4x4_50": 6, "4x4_100": 6,
    "5x5_50": 7, "5x5_100": 7,
    "6x6_50": 8,
}


def generate(marker_id: int = 10,
             dict_name: str = "5x5_50",
             size_cm: float = 3.8,   # inner active pattern = 38 mm
             border_cm: float = 0.6, # white border each side = 6 mm → total outer = 50 mm
             dpi: int = 600,
             out_dir: str | None = None) -> str:
    """
    Generate and save an ArUco marker image.

    Parameters
    ----------
    marker_id : int
        ArUco marker ID (0 to max for chosen dictionary).
    dict_name : str
        Dictionary name key from DICT_MAP (default: '5x5_50').
    size_cm : float
        Physical print size in centimetres (default: 4.5).
    dpi : int
        Print resolution — affects pixel count, NOT the printed size.
    out_dir : str | None
        Output directory. Defaults to the same folder as this script.

    Returns
    -------
    str
        Path to the saved PNG file.
    """
    if dict_name not in DICT_MAP:
        raise ValueError(f"Unknown dict '{dict_name}'. Choose from: {list(DICT_MAP)}")

    dictionary = cv2.aruco.getPredefinedDictionary(DICT_MAP[dict_name])
    cells      = CELLS_MAP[dict_name]

    # Pixel size: 1 inch = 2.54 cm  →  size_px = dpi × size_cm / 2.54
    # Round to nearest multiple of cells so each cell is an integer pixels wide
    raw_px    = int(round(dpi * size_cm / 2.54))
    size_px   = int(round(raw_px / cells) * cells)   # snap to cell grid

    # Generate the marker (includes 1-cell white border inside size_px)
    img = np.zeros((size_px, size_px), dtype=np.uint8)
    cv2.aruco.generateImageMarker(dictionary, marker_id, size_px, img, 1)

    # Add extra outer white padding (1 extra cell on each side) so nothing
    # is clipped when laminating or cutting
    cell_px = size_px // cells
    padded  = cv2.copyMakeBorder(img,
                                  cell_px, cell_px, cell_px, cell_px,
                                  cv2.BORDER_CONSTANT, value=255)

    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))

    filename = f"aruco_{dict_name}_id{marker_id}.png"
    out_path = os.path.join(out_dir, filename)
    cv2.imwrite(out_path, padded)

    total_px = padded.shape[0]
    print(f"[ArUco] Saved: {out_path}")
    print(f"[ArUco] Image size : {total_px} × {total_px} px  @ {dpi} dpi")
    print(f"[ArUco] Print size : {size_cm:.1f} × {size_cm:.1f} cm  "
          f"(total with border: {total_px / dpi * 2.54:.1f} cm)")
    print(f"[ArUco] Dict       : {dict_name}  |  ID: {marker_id}")
    print()
    print("  How to use in teleop_med7_vr.py:")
    print(f"    markers = avp_streamer.get_markers()")
    print(f"    marker  = markers.get({marker_id}, None)")
    print(f"    if marker and marker['is_tracked']:")
    print(f"        pose     = marker['pose']                # 4×4 in sim frame")
    print(f"        offset   = np.array([0, 0, -0.12, 1.0]) # 12 cm tip offset")
    print(f"        tip_pos  = (pose @ offset)[:3]")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate an ArUco marker PNG ready for print & mount on stylus")
    parser.add_argument("--id",   type=int,   default=7,       help="Marker ID (default: 7)")
    parser.add_argument("--size", type=float, default=4.5,     help="Physical size in cm (default: 4.5)")
    parser.add_argument("--dict", type=str,   default="5x5_50",
                        help=f"Dictionary (default: 5x5_50). Options: {list(DICT_MAP)}")
    parser.add_argument("--dpi",  type=int,   default=600,     help="Print DPI (default: 600)")
    args = parser.parse_args()

    generate(
        marker_id=args.id,
        dict_name=args.dict,
        size_cm=args.size,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
