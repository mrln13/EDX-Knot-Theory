#!/usr/bin/env python3
"""
simulate_scan_order_from_cache.py 

Examples
--------
# All edge starts, 8 workers, CSV + preview:
python simulate_scan_order_from_cache.py \
  --cache composition_cache.parquet \
  --pattern billiard --p 38 --q 53 \
  --starts-edges \
  --workers 8 \
  --out-template "out/{pattern}_edge_r{r}_c{c}.csv" \
  --preview-template "out/preview_{pattern}_edge_r{r}_c{c}.png"

# Explicit list of starts, TXT only, use all CPU cores:
python simulate_scan_order_from_cache.py \
  --index grid.csv --pattern raster --snake 1 \
  --start-list "0,0; 3,7; 5,2" \
  --workers -1 \
  --out-template "runs/{pattern}_start_{r}_{c}.txt"
"""

from __future__ import annotations
import argparse
import re
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # headless, faster and safe in subprocesses
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from concurrent.futures import ProcessPoolExecutor, as_completed

from methods import (
    billiard_knot,       # (t,p,q) with reflection
    compute_time_steps,  # exact center-hit schedule
)

# ---------------- I/O ----------------
def _load_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def load_grid(cache_or_index: Path, root: Path | None = None) -> tuple[list[tuple[int,int]], dict[tuple[int,int], str]]:
    df = _load_df(cache_or_index)
    need = {"row", "col", "path"}
    if not need.issubset(df.columns):
        raise SystemExit(f"{cache_or_index} must contain columns {need}")
    df = df.dropna(subset=["row", "col", "path"]).copy()
    df["row"] = df["row"].astype(int)
    df["col"] = df["col"].astype(int)
    df["path"] = df["path"].astype(str)

    coords_sorted = sorted(set(zip(df["row"], df["col"])))
    rc_to_path: dict[tuple[int,int], str] = {}

    root = root.resolve() if root else None
    for r, c, pstr in zip(df["row"], df["col"], df["path"]):
        p = Path(pstr)
        if root and not p.is_absolute():
            p = (root / p).resolve()
        rc_to_path[(int(r), int(c))] = str(p)
    return coords_sorted, rc_to_path

# ------------- grid helpers -------------
def _normalize_grid(coords_sorted: List[Tuple[int, int]]):
    rmin = min(r for r, _ in coords_sorted)
    rmax = max(r for r, _ in coords_sorted)
    cmin = min(c for _, c in coords_sorted)
    cmax = max(c for _, c in coords_sorted)
    R = rmax - rmin + 1
    C = cmax - cmin + 1
    present = [[False] * C for _ in range(R)]
    for r, c in coords_sorted:
        present[r - rmin][c - cmin] = True
    return R, C, rmin, cmin, present

def _rotate_to_start(order: list[tuple[int,int]], start_rc: tuple[int,int] | None) -> list[tuple[int,int]]:
    # Rotate a ready-made order so it *starts* at start_rc (or nearest if missing)
    if not start_rc or not order:
        return order
    if start_rc in order:
        k = order.index(start_rc)
        return order[k:] + order[:k]
    # fallback: rotate to nearest
    sr, sc = start_rc
    kmin, dmin = 0, abs(order[0][0]-sr) + abs(order[0][1]-sc)
    for i,(r,c) in enumerate(order):
        d = abs(r-sr) + abs(c-sc)
        if d < dmin:
            kmin, dmin = i, d
    # (warn only in parent process; worker stays quiet)
    return order[kmin:] + order[:kmin]

def edge_starts(coords_sorted: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    """
    All tiles that lie on the bounding-box perimeter:
    r == rmin or r == rmax or c == cmin or c == cmax.
    (Only returns tiles that actually exist in coords_sorted.)
    """
    if not coords_sorted:
        return []
    rmin = min(r for r, _ in coords_sorted)
    rmax = max(r for r, _ in coords_sorted)
    cmin = min(c for _, c in coords_sorted)
    cmax = max(c for _, c in coords_sorted)
    return [(r, c) for (r, c) in coords_sorted
            if r == rmin or r == rmax or c == cmin or c == cmax]

# ----------- non-billiard patterns -----------
def order_raster_row(coords_sorted, start_rc=None, snake=False):
    R, C, rmin, cmin, present = _normalize_grid(coords_sorted)
    order = []
    for rr in range(R):
        cols = range(C-1, -1, -1) if (snake and rr % 2 == 1) else range(C)
        for cc in cols:
            if present[rr][cc]:
                order.append((rmin+rr, cmin+cc))
    return _rotate_to_start(order, start_rc)

def order_raster_col(coords_sorted, start_rc=None, snake=False):
    R, C, rmin, cmin, present = _normalize_grid(coords_sorted)
    order = []
    for cc in range(C):
        rows = range(R-1, -1, -1) if (snake and cc % 2 == 1) else range(R)
        for rr in rows:
            if present[rr][cc]:
                order.append((rmin+rr, cmin+cc))
    return _rotate_to_start(order, start_rc)

def order_spiral(coords_sorted, start_rc=None, clockwise=True):
    R, C, rmin, cmin, present = _normalize_grid(coords_sorted)
    top, left, bottom, right = 0, 0, R-1, C-1
    dense = []
    while top <= bottom and left <= right:
        if clockwise:
            for c in range(left, right+1): dense.append((top,c))
            for r in range(top+1, bottom+1): dense.append((r,right))
            if top < bottom:
                for c in range(right-1, left-1, -1): dense.append((bottom,c))
            if left < right:
                for r in range(bottom-1, top, -1): dense.append((r,left))
        else:
            for r in range(top, bottom+1): dense.append((r,left))
            for c in range(left+1, right+1): dense.append((bottom,c))
            if left < right:
                for r in range(bottom-1, top-1, -1): dense.append((r,right))
            if top < bottom:
                for c in range(right-1, left, -1): dense.append((top,c))
        top += 1; left += 1; bottom -= 1; right -= 1
    order = [(rmin+r, cmin+c) for (r,c) in dense if present[r][c]]
    return _rotate_to_start(order, start_rc)

# ---------------- billiard via methods.py ----------------
def order_billiard_methods(coords_sorted: list[tuple[int,int]],
                           p: int, q: int,
                           start_rc: tuple[int,int] | None) -> list[tuple[int,int]]:
    R, C, rmin, cmin, present = _normalize_grid(coords_sorted)
    filled = sum(sum(row) for row in present)
    if filled != R*C:
        print(f"[warn] grid has holes: {filled}/{R*C}. Missing cells will be skipped.")

    # exact center hits from your code
    t_vals = compute_time_steps(C, R, p, q)
    if t_vals.size == 0:
        raise SystemExit("compute_time_steps produced no hits — check (p,q) and grid size.")

    seen = set()
    order: list[tuple[int,int]] = []
    for t in t_vals:  # tight loop; keep it lean
        x, y = billiard_knot(float(t), p, q)
        gx = int(x * C); gy = int(y * R)
        if 0 <= gx < C and 0 <= gy < R and present[gy][gx]:
            rc = (rmin + gy, cmin + gx)
            if rc not in seen:
                seen.add(rc)
                order.append(rc)
    return _rotate_to_start(order, start_rc)

# ---------------- router ----------------
def build_scan_order(coords_sorted,
                     pattern="billiard",
                     start=None,
                     snake=False,
                     clockwise=True,
                     p: int = 38,
                     q: int = 53):
    start_rc = None
    if start:
        m = re.match(r"^\s*(-?\d+)\s*,\s*(-?\d+)\s*$", start)
        if m:
            start_rc = (int(m.group(1)), int(m.group(2)))

    patt = pattern.lower()
    if patt in ("raster", "row", "row-major"):
        return order_raster_row(coords_sorted, start_rc, snake=bool(snake))
    if patt in ("col", "column", "column-major"):
        return order_raster_col(coords_sorted, start_rc, snake=bool(snake))
    if patt in ("spiral",):
        return order_spiral(coords_sorted, start_rc, clockwise=bool(clockwise))
    if patt in ("billiard", "billiard-knot", "lissajous"):
        return order_billiard_methods(coords_sorted, int(p), int(q), start_rc)
    raise ValueError(f"Unknown pattern: {pattern}")

# -------------- text & CSV output --------------
def _emit_path(p: str, absolute: bool, rel_base: Path | None) -> str:
    P = Path(p)
    if absolute:
        return str(P.resolve())
    if rel_base:
        try:
            return str(P.resolve().relative_to(rel_base.resolve()))
        except Exception:
            return str(P)
    return str(P)

def write_sequence_csv(seq: List[Tuple[int,int]],
                       rc_to_path: Dict[Tuple[int,int], str],
                       out_csv: Path,
                       absolute: bool = False,
                       rel_base: Path | None = None):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["acq_index", "row", "col", "path"])
        for k, rc in enumerate(seq):
            p = rc_to_path.get(rc)
            if p is None:
                continue
            w.writerow([k, rc[0], rc[1], _emit_path(p, absolute, rel_base)])

def write_sequence_txt(seq: List[Tuple[int,int]],
                       rc_to_path: Dict[tuple[int,int], str],
                       out_txt: Path,
                       absolute: bool = False,
                       rel_base: Path | None = None):
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("# acq_index,row,col,path\n")
        for k, rc in enumerate(seq):
            p = rc_to_path.get(rc)
            if p is None:
                continue
            f.write(f"{k},{rc[0]},{rc[1]},{_emit_path(p, absolute, rel_base)}\n")

# -------------- Preview (classic center-to-center arrows) --------------
def render_preview(seq: list[tuple[int,int]],
                   coords_sorted: list[tuple[int,int]],
                   out_file: Path,
                   label_skip: int = 10,
                   dpi: int = 180,
                   pad: float = 0.15):
    if not seq:
        return

    rmin = min(r for r,_ in coords_sorted); rmax = max(r for r,_ in coords_sorted)
    cmin = min(c for _,c in coords_sorted); cmax = max(c for _,c in coords_sorted)
    R = rmax - rmin + 1; C = cmax - cmin + 1

    fig_w = max(4.0, C * 0.4); fig_h = max(4.0, R * 0.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    for (r,c) in coords_sorted:
        ax.add_patch(Rectangle((c - cmin, r - rmin), 1, 1, fill=False, linewidth=0.6, alpha=0.6))

    # arrows
    for i in range(len(seq) - 1):
        r0, c0 = seq[i];   x0, y0 = c0 - cmin + 0.5, r0 - rmin + 0.5
        r1, c1 = seq[i+1]; x1, y1 = c1 - cmin + 0.5, r1 - rmin + 0.5
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=0.9, alpha=0.9, color=(0.1, 0.1, 0.1, 1.0))
        )

    # start/end markers
    rs, cs = seq[0]; re, ce = seq[-1]
    ax.plot([cs - cmin + 0.5], [rs - rmin + 0.5], marker="o", markersize=6, color="tab:green", zorder=3)
    ax.plot([ce - cmin + 0.5], [re - rmin + 0.5], marker="s", markersize=6, color="tab:red", zorder=3)

    if label_skip and label_skip > 0:
        for i, (r, c) in enumerate(seq):
            if i % label_skip == 0:
                ax.text(c - cmin + 0.05, r - rmin + 0.2, f"{i}", fontsize=6, color="k")

    ax.set_xlim(-pad, C + pad); ax.set_ylim(R + pad, -pad)
    ax.set_aspect("equal"); ax.axis("off")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.1)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

# ---------- parse starts & templating ----------
def parse_start_list(s: str) -> list[tuple[int,int]]:
    out = []
    for item in re.split(r"[;]+", s.strip()):
        item = item.strip()
        if not item:
            continue
        m = re.match(r"^\s*(-?\d+)\s*,\s*(-?\d+)\s*$", item)
        if not m:
            raise ValueError(f"Bad start token: '{item}' (expected 'r,c')")
        out.append((int(m.group(1)), int(m.group(2))))
    return out

def render_template(tmpl: str, pattern: str, r: int, c: int) -> str:
    return tmpl.replace("{pattern}", pattern).replace("{r}", str(r)).replace("{c}", str(c))

# ---------- worker (runs in subprocess) ----------
def _worker_per_start(payload):
    """
    payload dict keys:
      start_rc, base_seq, coords_sorted, rc_to_path, patt, args_public
    """
    start_rc = payload["start_rc"]
    base_seq = payload["base_seq"]
    coords_sorted = payload["coords_sorted"]
    rc_to_path = payload["rc_to_path"]
    patt = payload["patt"]
    A = payload["args_public"]

    # Rotate canonical order for this start
    seq = _rotate_to_start(base_seq, start_rc)

    # Determine (r,c) used for filenames
    if start_rc is None:
        if not seq:
            return (None, "empty sequence")
        r0, c0 = seq[0]
    else:
        r0, c0 = start_rc

    # Output path(s)
    out_path = Path(render_template(A["out_template"], patt, r0, c0))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write data
    ext = out_path.suffix.lower()
    if ext == ".txt":
        write_sequence_txt(seq, rc_to_path, out_path, absolute=A["absolute"], rel_base=A["rel_base"])
    elif ext == ".csv":
        write_sequence_csv(seq, rc_to_path, out_path, absolute=A["absolute"], rel_base=A["rel_base"])
    else:
        out_path = out_path.with_suffix(".csv")
        write_sequence_csv(seq, rc_to_path, out_path, absolute=A["absolute"], rel_base=A["rel_base"])

    # Optional preview
    if A["preview_template"]:
        prev_path = Path(render_template(A["preview_template"], patt, r0, c0))
        render_preview(seq, coords_sorted, prev_path, label_skip=A["label_skip"], dpi=A["dpi"])
        return (str(out_path), str(prev_path))
    return (str(out_path), None)

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Simulate scan order from a cache/index without reading BCFs (fast, parallel).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--cache", type=str, help="composition_cache.parquet (row,col,path)")
    src.add_argument("--index", type=str, help="grid index CSV/Parquet (row,col,path)")

    ap.add_argument("--pattern", type=str, default="billiard", help="billiard | raster | column | spiral")
    # starts
    ap.add_argument("--start-at", type=str, default="", help="Single start (row,col), e.g. '0,0'")
    ap.add_argument("--start-list", type=str, default="", help="Multiple starts: 'r,c;r,c;...'")
    ap.add_argument("--starts-edges", action="store_true", help="Use all tiles on the perimeter of the grid as starts")
    ap.add_argument("--starts-all", action="store_true", help="Use every tile present in the grid as a start")

    # pattern modifiers
    ap.add_argument("--snake", type=int, default=0, help="Raster/column: 1=snake")
    ap.add_argument("--clockwise", type=int, default=1, help="Spiral: 1=clockwise, 0=counter")
    ap.add_argument("--p", type=int, default=38, help="Billiard p (default 38)")
    ap.add_argument("--q", type=int, default=53, help="Billiard q (default 53)")

    # output templating
    ap.add_argument("--out-template", type=str, default="simulated_{pattern}_r{r}_c{c}.csv",
                    help="Output file template; use {pattern},{r},{c}. Extension decides format (.csv or .txt).")
    ap.add_argument("--preview-template", type=str, default="",
                    help="Optional per-start preview template; use {pattern},{r},{c} (e.g., 'previews/{pattern}_r{r}_c{c}.png')")

    # path rewrites
    ap.add_argument("--root", type=str, default="", help="Resolve relative input paths against this folder")
    ap.add_argument("--rewrite-relative-to", type=str, default="", help="Rewrite output paths relative to this folder")
    ap.add_argument("--absolute", action="store_true", help="Force output paths to be absolute (overrides rewrite-relative-to)")

    # preview controls
    ap.add_argument("--label-skip", type=int, default=0, help="Label acquisition index every N tiles (0=off)")
    ap.add_argument("--dpi", type=int, default=180, help="Preview DPI")

    # parallelism
    ap.add_argument("--workers", type=int, default=0,
                    help="Number of worker processes (-1 = all cores; 0/1 = no parallelism; N>1 = N workers)")

    args = ap.parse_args()

    # Load grid and map
    root = Path(args.root).resolve() if args.root else None
    src_path = Path(args.cache if args.cache else args.index)
    coords_sorted, rc_to_path = load_grid(src_path, root=root)
    if not coords_sorted:
        raise SystemExit("No (row,col) entries found.")

    # Decide list of starts (priority order)
    if args.starts_all:
        starts = coords_sorted[:]
    elif args.starts_edges:
        starts = edge_starts(coords_sorted)
        if not starts:
            raise SystemExit("No edge tiles found.")
    elif args.start_list.strip():
        starts = parse_start_list(args.start_list)
    elif args.start_at.strip():
        starts = parse_start_list(args.start_at)
    else:
        starts = [None]  # will use the canonical order as-is

    # Build *canonical* sequence ONCE (no start rotation here)
    patt = args.pattern.lower()
    canonical_seq = build_scan_order(
        coords_sorted,
        pattern=patt,
        start="",                  # critical: don't rotate here
        snake=bool(args.snake),
        clockwise=bool(args.clockwise),
        p=args.p, q=args.q,
    )
    if not canonical_seq:
        raise SystemExit("Canonical sequence is empty.")

    # Prepare common args for workers
    rel_base = Path(args.rewrite_relative_to).resolve() if args.rewrite_relative_to else None
    if args.absolute:
        rel_base = None

    args_public = dict(
        out_template=args.out_template,
        preview_template=args.preview_template,
        absolute=bool(args.absolute),
        rel_base=rel_base,
        label_skip=max(0, args.label_skip),
        dpi=max(72, args.dpi),
    )

    # Build payloads
    payloads = []
    for start_rc in starts:
        payloads.append(dict(
            start_rc=start_rc,
            base_seq=canonical_seq,
            coords_sorted=coords_sorted,
            rc_to_path=rc_to_path,
            patt=patt,
            args_public=args_public,
        ))

    # Decide number of workers
    if args.workers == -1:
        workers = max(1, os.cpu_count() or 1)
    elif args.workers in (0, 1):
        workers = 1
    else:
        workers = max(1, args.workers)

    # Execute per-start tasks (parallel if workers>1)
    if workers == 1:
        # sequential (still fast thanks to canonical rotation)
        for pl in tqdm(payloads, desc="Processing starts", unit="start"):
            out_path, prev_path = _worker_per_start(pl)
            if out_path:
                print(f"[simulate] Wrote → {out_path}")
            if prev_path:
                print(f"[preview] Saved → {prev_path}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_worker_per_start, pl) for pl in payloads]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Processing starts (x{workers})", unit="start"):
                out_path, prev_path = fut.result()
                if out_path:
                    print(f"[simulate] Wrote → {out_path}")
                if prev_path:
                    print(f"[preview] Saved → {prev_path}")

if __name__ == "__main__":
    main()
