#!/usr/bin/env python3
"""
analyze_orders_aggregate.py

Batch-analyze cumulative (semi-quant) composition over MANY simulated scan orders
WITHOUT re-reading BCFs. Writes ONE aggregate JSON for all runs.

Inputs:
  --cache semiquant_cache.parquet         (from build_semiquant_cache.py; element columns are % per tile)
  --orders "*.csv"  OR  --orders file1.csv file2.csv ...
  [optional] --base-dir <dir>             (prefer to read inputs here; write the aggregate here)
  [optional] --out  all_metrics.json      (single output JSON)

Output JSON structure:
{
  "runs": [
    {
      "file": "dataset_total.csv",
      "elements": ["Si","O",...],           # order used when composing vectors
      "composition": [ {<el>: frac, ...}, ...],  # fractions (0..1) cumulative by step
      "n_tiles": 123
    },
    ...
  ]
}
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import json
import math

import numpy as np
import pandas as pd
from tqdm import tqdm

# ----------------------- Utilities -----------------------

def resolve_input_path(arg_path: str, base_dir: str | None) -> Path:
    given = Path(arg_path)
    candidates = []
    if base_dir:
        bd = Path(base_dir).resolve()
        candidates.append(bd / given.name)
        if not given.is_absolute():
            candidates.append(bd / given)
    candidates.append(given if given.is_absolute() else Path.cwd() / given)
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(str(given))

def resolve_many_orders(order_args: List[str], base_dir: str | None) -> List[Path]:
    out: List[Path] = []
    if len(order_args) == 1 and any(ch in order_args[0] for ch in "*?[]"):
        # glob pattern
        base = Path(base_dir).resolve() if base_dir else Path.cwd()
        out = sorted(base.glob(order_args[0]))
    else:
        for a in order_args:
            out.append(resolve_input_path(a, base_dir))
    # keep only CSV files
    out = [p for p in out if p.suffix.lower() == ".csv"]
    return out

def _find_element_columns(df: pd.DataFrame) -> list[str]:
    fixed = {"path","row","col","stage_x","stage_y","weight","__valid","filename","acq_index"}
    cols = [c for c in df.columns if c not in fixed and pd.api.types.is_numeric_dtype(df[c])]
    return sorted(cols)

def _merge_order_with_cache(order_df: pd.DataFrame, cache_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    # Try (row, col)
    if {"row","col"}.issubset(order_df.columns) and {"row","col"}.issubset(cache_df.columns):
        oc = order_df.merge(cache_df, on=["row","col"], how="left", suffixes=("","_cache"))
        matched = int(oc["weight"].notna().sum()) if "weight" in oc.columns else 0
        if matched > 0:
            oc = oc.sort_values(["acq_index","path"], na_position="last").groupby("acq_index", as_index=False).first()
            return oc, "(row,col)"
    # Fallback: filename
    cache2 = cache_df.copy()
    cache2["filename"] = cache2["path"].astype(str).apply(lambda p: Path(p).name if isinstance(p, str) else "")
    if "path" in order_df.columns and "filename" not in order_df.columns:
        order_df = order_df.copy()
        order_df["filename"] = order_df["path"].astype(str).apply(lambda p: Path(p).name if isinstance(p, str) else "")
    if "filename" in order_df.columns and "filename" in cache2.columns:
        oc = order_df.merge(cache2, on="filename", how="left", suffixes=("","_cache"))
        matched = int(oc["weight"].notna().sum()) if "weight" in oc.columns else 0
        if matched > 0:
            return oc, "filename"
    raise RuntimeError("Could not merge order with cache by (row,col) or by filename.")

# ----------------------- Main -----------------------

def main():
    ap = argparse.ArgumentParser(description="Batch analyze orders → single aggregate JSON of compositions.")
    ap.add_argument("--cache", required=True, help="semiquant_cache.parquet")
    ap.add_argument("--orders", nargs="+", required=True, help="glob or list of CSVs")
    ap.add_argument("--base-dir", type=str, default="", help="Prefer inputs here and write outputs here")
    ap.add_argument("--out", type=str, default="all_metrics.json", help="single output JSON filename")
    args = ap.parse_args()

    base_dir = args.base_dir if args.base_dir else None

    cache_path = resolve_input_path(args.cache, base_dir)
    orders = resolve_many_orders(args.orders, base_dir)
    if not orders:
        raise SystemExit("No order CSVs found for given --orders.")

    out_dir = Path(base_dir).resolve() if base_dir else (orders[0].parent if orders else Path.cwd())
    out_path = out_dir / args.out

    # Load cache
    cache = pd.read_parquet(cache_path)
    for c in ("row","col"):
        if c in cache.columns:
            cache[c] = pd.to_numeric(cache[c], errors="coerce")
    if "weight" in cache.columns:
        cache["weight"] = pd.to_numeric(cache["weight"], errors="coerce").fillna(0.0)

    elem_cols = _find_element_columns(cache)
    if not elem_cols:
        raise SystemExit("No element columns found in cache file.")

    runs: list[dict] = []

    for op in tqdm(orders, desc="Processing orders", unit="order"):
        try:
            order = pd.read_csv(op)
        except Exception as e:
            print(f"[warn] {op.name}: cannot read CSV ({e}); skipping.")
            continue

        if "acq_index" in order.columns:
            order = order.sort_values("acq_index").reset_index(drop=True)
        for c in ("row","col"):
            if c in order.columns:
                order[c] = pd.to_numeric(order[c], errors="coerce")

        # Merge
        try:
            seq, joined_on = _merge_order_with_cache(order, cache)
        except Exception as e:
            print(f"[warn] {op.name}: {e} Skipping.")
            continue

        seq = seq.dropna(subset=["weight"]).reset_index(drop=True)
        if len(seq) == 0:
            print(f"[warn] {op.name}: no valid tiles after merge; skipping.")
            continue

        # Accumulate compositions (fractions 0..1)
        comp_total = {el: 0.0 for el in elem_cols}
        wsum = 0.0
        composition_series: List[Dict[str, float]] = []

        for _, r in seq.iterrows():
            w = float(r["weight"])
            if w <= 0:
                continue
            for el in elem_cols:
                pct = float(r.get(el, 0.0))
                if not math.isfinite(pct):
                    pct = 0.0
                comp_total[el] += w * (pct / 100.0)
            wsum += w

            cur = {el: (comp_total[el] / wsum if wsum > 0 else 0.0) for el in elem_cols}
            composition_series.append(cur)

        if not composition_series:
            print(f"[warn] {op.name}: empty composition series; skipping.")
            continue

        runs.append({
            "file": op.name,
            "elements": elem_cols,
            "composition": composition_series,
            "n_tiles": len(composition_series),
        })

    aggregate = {"runs": runs}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f)
    print(f"[export] Wrote {len(runs)} runs → {out_path.as_posix()}")

if __name__ == "__main__":
    main()
