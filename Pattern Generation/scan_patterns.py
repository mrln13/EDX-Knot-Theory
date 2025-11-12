#!/usr/bin/env python3
"""
Use methods.py end-to-end on either:
  - a real TIFF input (RGB), or
  - a blank RGB canvas (width x height).

You can either:
  A) let the script search tile sizes and (p,q) near (p0,q0), or
  B) specify an explicit grid (--tiles-x, --tiles-y) and (optionally) explicit --p, --q.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import tifffile
import methods as m
import matplotlib.pyplot as plt

def save_visualization_png(canvas, sm, tile_w, tile_h, p, q, out_path, fig_title="", legend_label="Normalized Time Step"):
    """
    Save a PNG with image, title, and colorbar legend (using the ScalarMappable `sm` from methods.place_tiles).
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(canvas, interpolation='none', origin='upper')
    ax.axis('off')

    # Title
    if not fig_title:
        fig_title = f"Canvas: {canvas.shape[1]}x{canvas.shape[0]}, Tile: {tile_w}x{tile_h}, p={p}, q={q}"
    ax.set_title(fig_title)

    # Colorbar legend
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label(legend_label or "Normalized Time Step", rotation=270, labelpad=15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.repeat(img[..., None], 3, axis=2)
    return img


def main():
    ap = argparse.ArgumentParser(description="Billiard-knot scanning using methods.py (supports blank canvas).")

    # Input OR blank canvas
    ap.add_argument("--input", type=str, default="", help="Path to input TIFF (RGB). If omitted, a blank canvas is used.")
    ap.add_argument("--blank-w", type=int, default=0, help="Blank canvas width (pixels) if --input not given.")
    ap.add_argument("--blank-h", type=int, default=0, help="Blank canvas height (pixels) if --input not given.")
    ap.add_argument("--crop-bottom", type=int, default=0, help="Crop N bottom rows (e.g., remove legend).")

    # Grid & search controls
    ap.add_argument("--tiles-x", type=int, default=0, help="Explicit number of tiles along X (optional).")
    ap.add_argument("--tiles-y", type=int, default=0, help="Explicit number of tiles along Y (optional).")
    ap.add_argument("--p", type=int, default=0, help="Explicit p (optional).")
    ap.add_argument("--q", type=int, default=0, help="Explicit q (optional).")

    # If not giving explicit tiles/p,q, we can search around (p0,q0) and tile size bounds
    ap.add_argument("--min-tile", type=int, default=0, help="Min tile px (required if searching).")
    ap.add_argument("--max-tile", type=int, default=0, help="Max tile px (required if searching).")
    ap.add_argument("--p0", type=int, default=0, help="Initial p for search (required if searching).")
    ap.add_argument("--q0", type=int, default=0, help="Initial q for search (required if searching).")
    ap.add_argument("--maximize-reflections", action="store_true", help="Search higher p,q first (default off).")

    # Placement / convergence
    ap.add_argument("--draw-path", action="store_true", help="Draw billiard path on canvas.")
    ap.add_argument("--mark-center", action="store_true", help="Mark centers (white until convergence, then red).")
    ap.add_argument("--limit", type=float, default=1.0, help="Fraction of normalized time to include (0..1].")
    ap.add_argument("--data", action="store_true", help="Treat input as data for composition/convergence. Ignored for blank.")
    ap.add_argument("--roc-threshold", type=float, default=0.001, help="ROC threshold for convergence.")
    ap.add_argument("--min-tiles", type=int, default=50, help="Min tiles before checking convergence.")
    ap.add_argument("--stop-at-convergence", action="store_true", help="Stop once convergence detected.")
    ap.add_argument("--no-show", action="store_true", help="Skip methods.visualize() window.")

    # Output
    ap.add_argument("--out-prefix", type=str, default="", help="Prefix for output files (default: input stem or 'blank').")
    ap.add_argument("--fig-out", type=str, default="", help="Save a PNG figure with title and colorbar legend.")
    ap.add_argument("--fig-title", type=str, default="", help="Optional custom figure title.")
    ap.add_argument("--legend-label", type=str, default="", help="Optional custom colorbar label.")

    args = ap.parse_args()

    # ------------------------------------------------------------
    # Load or build canvas
    # ------------------------------------------------------------
    if args.input:
        full_map = tifffile.imread(args.input)
        full_map = _ensure_rgb(full_map)
        if args.crop_bottom > 0:
            full_map = full_map[:-args.crop_bottom, :, :]
        data_mode = args.data  # only meaningful when we have a real image
        stem = args.out_prefix or Path(args.input).stem
    else:
        if args.blank_w <= 0 or args.blank_h <= 0:
            raise SystemExit("For a blank canvas, provide --blank-w and --blank-h.")
        full_map = np.zeros((args.blank_h, args.blank_w, 3), dtype=np.uint8)  # pure black RGB
        data_mode = False  # ignore --data for blank canvas
        stem = args.out_prefix or "blank"

    H, W = full_map.shape[:2]

    # ------------------------------------------------------------
    # Decide grid and (p,q)
    # ------------------------------------------------------------
    if args.tiles_x > 0 and args.tiles_y > 0:
        # explicit grid
        nx, ny = args.tiles_x, args.tiles_y
        # If p,q given explicitly use them, else fall back to p0/q0 if provided, else error.
        if args.p > 0 and args.q > 0:
            p, q = args.p, args.q
        elif args.p0 > 0 and args.q0 > 0:
            p, q = args.p0, args.q0
        else:
            raise SystemExit("Provide --p and --q (or --p0 and --q0) when using explicit --tiles-x/--tiles-y.")
        # compute coarse tile sizes; adjust_canvas_size will trim to multiples
        tile_w = max(1, W // nx)
        tile_h = max(1, H // ny)
        cropped_map, tile_w, tile_h = m.adjust_canvas_size(full_map, nx, ny)

    else:
        # search mode requires min/max + p0/q0
        if not (args.min_tile and args.max_tile and args.p0 and args.q0):
            raise SystemExit("Search mode requires --min-tile, --max-tile, --p0, and --q0.")
        tile_w, tile_h, nx, ny, p, q = m.find_valid_tile_size(
            full_map.shape[0:2],
            args.min_tile, args.max_tile,
            args.p0, args.q0,
            maximize_reflections=args.maximize_reflections
        )
        cropped_map, tile_w, tile_h = m.adjust_canvas_size(full_map, nx, ny)

    print(f"[grid] {W}x{H} -> tiles_x={nx}, tiles_y={ny}, tile={tile_w}x{tile_h}, p={p}, q={q}")

    # ------------------------------------------------------------
    # Compute time steps, place tiles, visualize
    # ------------------------------------------------------------
    t_vals = m.compute_time_steps(nx, ny, p, q)
    results, sm = m.place_tiles(
        cropped_map, tile_w, tile_h, nx, ny, p, q, t_vals,
        data=data_mode, draw_path=args.draw_path, mark_center=args.mark_center,
        limit=args.limit, rate_of_change_threshold=args.roc_threshold,
        min_tiles=args.min_tiles, stop_at_convergence=args.stop_at_convergence
    )
    # On-screen viz (optional)
    if not args.no_show:
        m.visualize(results[0], sm, tile_w, tile_h, p, q)

    # Save PNG with title + legend if requested
    if args.fig_out:
        save_visualization_png(
            results[0], sm, tile_w, tile_h, p, q,
            f"{stem}_p{p}_q{q}_{args.blank_w}x{args.blank_h}_tsmin{args.min_tile}_tsmax{args.max_tile}_legend.png",
            fig_title=args.fig_title,
            legend_label=args.legend_label or "Normalized Time Step",
        )
        print(f"[saved] {args.fig_out}")

    if not args.no_show:
        m.visualize(results[0], sm, tile_w, tile_h, p, q)

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    out_dir = Path(".")
    tifffile.imwrite(out_dir / f"{stem}_p{p}_q{q}_{args.blank_w}x{args.blank_h}_tsmin{args.min_tile}_tsmax{args.max_tile}_input.tif", cropped_map.astype("uint8"))
    tifffile.imwrite(out_dir / f"{stem}_p{p}_q{q}_{args.blank_w}x{args.blank_h}_tsmin{args.min_tile}_tsmax{args.max_tile}_canvas.tif", results[0].astype("uint8"))
    if len(results) > 1:
        tifffile.imwrite(out_dir / f"{stem}_p{p}_q{q}_{args.blank_w}x{args.blank_h}_tsmin{args.min_tile}_tsmax{args.max_tile}_output.tif", results[1].astype("uint8"))
    print(f"[saved] {len(results)} files" if len(results) > 1 else "")


if __name__ == "__main__":
    main()
