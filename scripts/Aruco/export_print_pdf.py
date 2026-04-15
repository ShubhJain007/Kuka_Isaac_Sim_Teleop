"""
Embed the ArUco marker PNG into a print-ready PDF at the exact physical size.

Usage:
    python export_print_pdf.py

Output:
    aruco_5x5_50_id7_PRINT.pdf  — open in any PDF viewer and print at 100% (no scaling)
"""

import os
from reportlab.lib.units import cm
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

MARKER_PNG   = os.path.join(os.path.dirname(__file__), "aruco_5x5_50_id7.png")
OUTPUT_PDF   = os.path.join(os.path.dirname(__file__), "aruco_5x5_50_id7_PRINT.pdf")

# Physical size of the full image (marker + padding border)
PRINT_SIZE_CM = 5.8   # matches generator output

def make_pdf():
    page_w, page_h = A4
    size = PRINT_SIZE_CM * cm

    c = canvas.Canvas(OUTPUT_PDF, pagesize=A4)

    # Center on A4 page
    x = (page_w - size) / 2
    y = (page_h - size) / 2

    c.drawImage(MARKER_PNG, x, y, width=size, height=size)

    # Label below the marker
    c.setFont("Helvetica", 8)
    label = f"ArUco DICT_5X5_50  ID=7  |  Print size: {PRINT_SIZE_CM} cm × {PRINT_SIZE_CM} cm  |  Print at 100% — NO SCALING"
    c.drawCentredString(page_w / 2, y - 0.5 * cm, label)

    # Crop marks at corners (5 mm lines, 2 mm gap from image edge)
    gap  = 0.2 * cm
    tick = 0.5 * cm
    c.setLineWidth(0.3)
    for cx, cy in [(x, y), (x + size, y), (x, y + size), (x + size, y + size)]:
        # horizontal tick
        dx = -tick if cx == x else tick
        c.line(cx + (gap if cx == x else -gap), cy,
               cx + dx, cy)
        # vertical tick
        dy = -tick if cy == y else tick
        c.line(cx, cy + (gap if cy == y else -gap),
               cx, cy + dy)

    c.save()
    print(f"[PDF] Saved: {OUTPUT_PDF}")
    print(f"[PDF] Print instructions:")
    print(f"       1. Open the PDF in any viewer (Evince, Okular, Adobe, Preview)")
    print(f"       2. Print → Paper size: A4")
    print(f"       3. Scale / Page sizing: 'Actual size' or '100%'  ← IMPORTANT")
    print(f"       4. Do NOT select 'Fit to page' or 'Shrink to printable area'")
    print(f"       5. Use matte paper, 600 dpi or best quality")
    print(f"       6. After printing, verify with a ruler: marker area = 4.5 cm,")
    print(f"          total with border = 5.8 cm")

if __name__ == "__main__":
    make_pdf()
