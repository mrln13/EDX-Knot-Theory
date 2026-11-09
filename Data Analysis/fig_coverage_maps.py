"""Coverage maps for the manuscript (schist Figure 3, panels a-c).

Draws one panel per scan pattern on the BSE mosaic: the tile grid, green dots
for the tiles acquired up to the convergence step of the chosen start, and red
dots for the further tiles up to the pattern's mean tiles-to-convergence
(Table 1). If the run converged after the mean, no red dots are drawn.

Scan orders come from simulate_scan_order_from_cache.py, i.e. the same
generators that produced the orders behind the statistics.

Display convention: the tile indices of the composition cache follow the BSE
mosaic image (row r top to bottom, column c left to right), so tiles are drawn
without flipping. For the schist this was checked against the per-start
convergence steps quoted in the original caption (111 / 145 / 135 tiles for
billiard / row raster / clockwise spiral), which all belong to start r0_c52,
the top-right tile. Use --flip-x only for grids indexed from the top-right.

Example (schist, start r0_c52 = top-right):
  python fig_coverage_maps.py --bse C23S1A_BSE_resized.png --rows 38 --cols 53 \
      --p 38 --q 53 --start 0,52 --out-prefix Fig4 \
      --panel a:billiard:111:126 --panel b:raster-row:145:132 --panel c:spiral-cw:135:139
"""
import argparse
import importlib.util
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("sim", HERE / "simulate_scan_order_from_cache.py")
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)

GREEN = (0, 235, 0)
RED = (235, 0, 0)
GRID = (0, 0, 200)


def scan_order(kind, coords, start, p, q):
    """Return the acquisition order (list of (r, c)) for a pattern variant."""
    if kind == "billiard":
        return sim.order_billiard_methods(coords, p, q, start)
    if kind == "raster-row":
        return sim.order_raster_row(coords, start, snake=True)
    if kind == "raster-column":
        return sim.order_raster_col(coords, start, snake=True)
    if kind == "spiral-cw":
        return sim.order_spiral(coords, start, clockwise=True)
    if kind == "spiral-ccw":
        return sim.order_spiral(coords, start, clockwise=False)
    raise ValueError(kind)


def draw_panel(bse, rows, cols, order, conv, mean, scale, flip_x=False):
    W, H = bse.size
    img = bse.convert("RGB").resize((round(W * scale), round(H * scale)), Image.LANCZOS)
    W, H = img.size
    tw, th = W / cols, H / rows
    d = ImageDraw.Draw(img)
    lw = max(1, round(1.5 * scale))
    for c in range(cols + 1):
        x = round(c * tw); d.line([(x, 0), (x, H)], fill=GRID, width=lw)
    for r in range(rows + 1):
        y = round(r * th); d.line([(0, y), (W, y)], fill=GRID, width=lw)
    rad = 0.3 * min(tw, th)

    def dot(rc, colour):
        r, c = rc
        cx = ((cols - 1 - c) if flip_x else c) * tw + 0.5 * tw
        cy = (r + 0.5) * th
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=colour)

    for rc in order[:conv]:
        dot(rc, GREEN)
    for rc in order[conv:max(conv, mean)]:
        dot(rc, RED)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bse", required=True)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--p", type=int, required=True)
    ap.add_argument("--q", type=int, required=True)
    ap.add_argument("--start", default="0,0", help="start tile 'r,c'")
    ap.add_argument("--flip-x", action="store_true", help="draw column 0 at the right edge")
    ap.add_argument("--panel", action="append", required=True,
                    help="label:pattern:conv_step:mean, pattern in billiard|raster-row|"
                         "raster-column|spiral-cw|spiral-ccw")
    ap.add_argument("--scale", type=float, default=1.6, help="output size relative to the BSE image")
    ap.add_argument("--out-prefix", default="Fig4")
    args = ap.parse_args()

    Image.MAX_IMAGE_PIXELS = None
    bse = Image.open(args.bse)
    coords = [(r, c) for r in range(args.rows) for c in range(args.cols)]
    start = tuple(int(v) for v in args.start.split(","))
    for spec in args.panel:
        label, kind, conv, mean = spec.split(":")
        order = scan_order(kind, coords, start, args.p, args.q)
        img = draw_panel(bse, args.rows, args.cols, order, int(conv), int(mean), args.scale, args.flip_x)
        out = f"{args.out_prefix}_{label}.png"
        img.save(out)
        print(f"[fig] {label}: {kind}, first tiles {order[:4]}, conv {conv}, mean {mean} -> {out}")


if __name__ == "__main__":
    main()
