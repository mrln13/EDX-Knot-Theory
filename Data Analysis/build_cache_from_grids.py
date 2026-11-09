#!/usr/bin/env python3
"""
build_cache_from_grids.py — build a composition cache from pre-exported element weight-percent CSV grids.

Takes a directory of CSV files (one per element, each a 2-D pixel grid of wt% values)
and produces a parquet cache compatible with the downstream pipeline
(simulate_scan_order_from_cache.py -> analyze_orders.py -> Calc_Stats.R).

The tile grid is derived from billiard-knot parameters (p, q), not specified manually.

Explicit mode — provide (p, q) directly:
  python build_cache_from_grids.py --dir "D:\\UU\\Kibo" --p 57 --q 40 --out cache.parquet

Search mode — find valid (p, q) within a tile-size range:
  python build_cache_from_grids.py --dir "D:\\UU\\Kibo" --p0 57 --q0 40 \\
         --min-tile 50 --max-tile 200 --maximize-reflections --out cache.parquet
"""

from __future__ import annotations
import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Import billiard-knot helpers from Trajectory Generation
_TRAJ_DIR = str(Path(__file__).resolve().parent.parent / "Trajectory Generation")
if _TRAJ_DIR not in sys.path:
    sys.path.insert(0, _TRAJ_DIR)

from methods import check_full_coverage, find_valid_tile_size


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _pattern_to_regex(pattern: str) -> re.Pattern:
    """Convert a filename pattern with ``{element}`` placeholder to a regex."""
    parts = pattern.split("{element}")
    escaped = [re.escape(p) for p in parts]
    return re.compile(escaped[0] + r"(.+)" + escaped[1])


def discover_element_csvs(directory: Path, pattern: str) -> Dict[str, Path]:
    """Return ``{element_symbol: csv_path}`` for every file matching *pattern*."""
    regex = _pattern_to_regex(pattern)
    found: Dict[str, Path] = {}
    for p in sorted(directory.iterdir()):
        if not p.is_file():
            continue
        m = regex.fullmatch(p.name)
        if m:
            found[m.group(1)] = p
    return found


# ---------------------------------------------------------------------------
# Loading & tiling
# ---------------------------------------------------------------------------

def load_grids(element_paths: Dict[str, Path]) -> Tuple[List[str], np.ndarray]:
    """Load all element CSVs and stack into a 3-D array.

    Returns
    -------
    elements : list[str]
        Sorted element symbols.
    stack : ndarray, shape (n_elements, height, width)
        Weight-percent values.
    """
    elements = sorted(element_paths.keys())
    arrays: List[np.ndarray] = []
    ref_shape = None

    for el in elements:
        path = element_paths[el]
        arr = pd.read_csv(path, header=None).to_numpy(dtype=np.float64)
        if ref_shape is None:
            ref_shape = arr.shape
        elif arr.shape != ref_shape:
            raise ValueError(
                f"Grid shape mismatch: {el} has {arr.shape}, expected {ref_shape}"
            )
        arrays.append(arr)

    return elements, np.stack(arrays, axis=0)


def tile_and_aggregate(
    elements: List[str],
    stack: np.ndarray,
    tile_w: int,
    tile_h: int,
    num_tiles_x: int,
    num_tiles_y: int,
    pixel_size: float,
) -> pd.DataFrame:
    """Divide the pixel grid into tiles and compute mean wt% per tile.

    Parameters
    ----------
    elements : list[str]
        Element symbols, matching axis-0 of *stack*.
    stack : ndarray, shape (n_elements, height, width)
    tile_w, tile_h : int
        Tile dimensions in pixels.
    num_tiles_x, num_tiles_y : int
        Number of tiles along each axis.
    pixel_size : float
        um per pixel (for synthetic stage coordinates).

    Returns
    -------
    DataFrame with columns matching the composition-cache schema.
    """
    # Crop to exact tile grid
    cropped = stack[:, :num_tiles_y * tile_h, :num_tiles_x * tile_w]

    # Reshape and compute per-tile means
    block = cropped.reshape(
        len(elements), num_tiles_y, tile_h, num_tiles_x, tile_w
    )
    means = block.mean(axis=(2, 4))  # (n_elements, num_tiles_y, num_tiles_x)

    rows: List[Dict] = []
    for r in range(num_tiles_y):
        for c in range(num_tiles_x):
            comp = {el: float(means[i, r, c]) for i, el in enumerate(elements)}
            weight = sum(comp.values())
            rows.append({
                "path": f"tile_{r:04d}_{c:04d}",
                "row": r,
                "col": c,
                "stage_x": (c + 0.5) * tile_w * pixel_size,
                "stage_y": (r + 0.5) * tile_h * pixel_size,
                "weight": weight,
                "__valid": 1 if weight > 0 else 0,
                **comp,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["row", "col", "path"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build composition cache from element wt%% CSV grids. "
                    "Tile grid is derived from billiard-knot parameters (p, q)."
    )
    ap.add_argument("--dir", required=True,
                    help="Folder containing the element CSV grids")
    ap.add_argument("--out", required=True,
                    help="Output parquet file path")

    # Explicit mode
    ap.add_argument("--p", type=int, default=None,
                    help="Billiard-knot p (explicit mode)")
    ap.add_argument("--q", type=int, default=None,
                    help="Billiard-knot q (explicit mode)")

    # Search mode
    ap.add_argument("--p0", type=int, default=None,
                    help="Starting p for tile-size search (search mode)")
    ap.add_argument("--q0", type=int, default=None,
                    help="Starting q for tile-size search (search mode)")
    ap.add_argument("--min-tile", type=int, default=50,
                    help="Minimum tile size in pixels (search mode, default: 50)")
    ap.add_argument("--max-tile", type=int, default=200,
                    help="Maximum tile size in pixels (search mode, default: 200)")
    ap.add_argument("--maximize-reflections", action="store_true",
                    help="Search higher (p, q) first for denser paths (search mode)")
    ap.add_argument("--search-window", type=int, default=40,
                    help="Integer window size around (p0, q0) to search (default: 40)")

    # Crop
    ap.add_argument("--crop", type=str, default=None,
                    help="Crop region as x0,y0,w,h in pixels (applied before tiling)")

    # Common options
    ap.add_argument("--pixel-size", type=float, default=1.0,
                    help="um per pixel for synthetic stage coordinates (default: 1.0)")
    ap.add_argument("--pattern", default="Montaged Map Data-{element} Wt%.csv",
                    help="Filename pattern with {element} placeholder")
    args = ap.parse_args()

    # Validate mode
    explicit = args.p is not None and args.q is not None
    search = args.p0 is not None and args.q0 is not None
    if not explicit and not search:
        ap.error("Provide either --p/--q (explicit mode) or --p0/--q0 (search mode).")
    if explicit and search:
        ap.error("Provide --p/--q OR --p0/--q0, not both.")

    directory = Path(args.dir).resolve()
    if not directory.is_dir():
        raise SystemExit(f"[error] Not a directory: {directory}")

    # 1. Discover element CSVs
    element_paths = discover_element_csvs(directory, args.pattern)
    if not element_paths:
        raise SystemExit(
            f"[error] No files matching pattern '{args.pattern}' in {directory}"
        )
    elements_str = ", ".join(sorted(element_paths.keys()))
    print(f"[discover] Found {len(element_paths)} element grids: {elements_str}")

    # 2. Load grids
    print("[load] Reading CSV grids ...")
    elements, stack = load_grids(element_paths)
    _, height, width = stack.shape
    print(f"[load] Grid dimensions: {height} rows x {width} cols")

    # 2b. Optional crop
    if args.crop:
        parts = [int(v) for v in args.crop.split(",")]
        if len(parts) != 4:
            ap.error("--crop must be x0,y0,w,h")
        crop_x0, crop_y0, crop_w, crop_h = parts
        stack = stack[:, crop_y0:crop_y0 + crop_h, crop_x0:crop_x0 + crop_w]
        _, height, width = stack.shape
        print(f"[crop] Cropped to ({crop_x0},{crop_y0}) + {crop_w}x{crop_h} -> {height} rows x {width} cols")

    # 3. Determine tile grid from (p, q)
    if explicit:
        p, q = args.p, args.q
        gcd = math.gcd(p, q)
        num_tiles_x = q // gcd
        num_tiles_y = p // gcd
        tile_w = width // num_tiles_x
        tile_h = height // num_tiles_y
        if tile_w < 1 or tile_h < 1:
            raise SystemExit(
                f"[error] Tile size too small: {tile_w}x{tile_h} px "
                f"(grid {num_tiles_x}x{num_tiles_y} does not fit in {width}x{height} canvas)"
            )
        if not check_full_coverage(p, q, num_tiles_x, num_tiles_y):
            raise SystemExit(
                f"[error] (p={p}, q={q}) does not give full coverage "
                f"on a {num_tiles_x}x{num_tiles_y} grid."
            )
        print(f"[grid] Explicit: p={p}, q={q} -> "
              f"{num_tiles_x}x{num_tiles_y} tiles of {tile_w}x{tile_h} px")
    else:
        tile_w, tile_h, num_tiles_x, num_tiles_y, p, q = find_valid_tile_size(
            (height, width), args.min_tile, args.max_tile,
            args.p0, args.q0, args.maximize_reflections, args.search_window,
        )
        print(f"[grid] Search found: p={p}, q={q} -> "
              f"{num_tiles_x}x{num_tiles_y} tiles of {tile_w}x{tile_h} px")

    # Report discarded edge pixels
    discard_y = height - num_tiles_y * tile_h
    discard_x = width - num_tiles_x * tile_w
    if discard_y or discard_x:
        print(f"[grid] Discarding edge remainders: {discard_y} px bottom, {discard_x} px right")

    # 4. Tile & aggregate
    df = tile_and_aggregate(
        elements, stack, tile_w, tile_h, num_tiles_x, num_tiles_y, args.pixel_size
    )

    # 5. Write parquet
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outp, index=False)

    valid = int(df["__valid"].sum())
    print(f"[cache] Wrote {len(df)} rows -> {outp}")
    print(f"[cache] Valid tiles: {valid}")


if __name__ == "__main__":
    main()
