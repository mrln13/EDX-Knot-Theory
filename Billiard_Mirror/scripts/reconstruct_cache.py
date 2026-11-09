"""Rebuild a per-tile composition cache by matching a stored cumulative run
to the SPX quant results.

analyze_orders.py accumulates C_t = sum_i w_i * pct_i/100 / sum_i w_i along
an order. Given the order (pattern + start, hence (row, col) per step) and the
candidate tiles (pct, w) parsed from all SPX files, each step identifies
exactly one file: the one whose addition reproduces C_t.

usage: python reconstruct_cache.py DATASET_DIR FIRST_RUN.json ROWS COLS PATTERN OUT.parquet
  PATTERN: column-snake | row-snake   (start taken from the run's file name)
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"C:/Users/4029771/OneDrive - Universiteit Utrecht/Dev/EDX-Knot-Theory/Data Analysis")
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("bcc", REPO / "build_composition_cache.py")
bcc = importlib.util.module_from_spec(spec); spec.loader.exec_module(bcc)
spec2 = importlib.util.spec_from_file_location("sim", REPO / "simulate_scan_order_from_cache.py")
sim = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(sim)

base, run_f, R, C, patt, out = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
run = json.load(open(run_f))
els = run["elements"]
Cm = np.array([[step.get(e, 0.0) for e in els] for step in run["composition"]])
r0, c0 = map(int, re.search(r"_r(\d+)_c(\d+)", run["file"]).groups())

# candidate tiles
spx = sorted(base.rglob("*.spx"))
cand_pct, cand_w, cand_path = [], [], []
for f in spx:
    comp, w = bcc._spx_quant_for_file(f)
    if w <= 0 or not comp:
        continue
    cand_pct.append([float(comp.get(e, 0.0)) for e in els])
    cand_w.append(float(w))
    cand_path.append(str(f.relative_to(base)))
    extra = set(comp) - set(els)
    if extra:
        print("note: elements not in run:", extra, f.name)
P = np.array(cand_pct) / 100.0
Wc = np.array(cand_w)
print(f"{len(spx)} SPX files, {len(Wc)} with quant; run {run['file']} with {len(Cm)} steps")

coords = [(r, c) for r in range(R) for c in range(C)]
if patt == "column-snake":
    order = sim.order_raster_col(coords, (r0, c0), snake=True)
else:
    order = sim.order_raster_row(coords, (r0, c0), snake=True)
assert len(order) == len(Cm), (len(order), len(Cm))

A = np.zeros(len(els)); W = 0.0
pick, errs, second = [], [], []
for t in range(len(Cm)):
    pred = (A[None, :] + Wc[:, None] * P) / (W + Wc)[:, None]
    e = np.abs(pred - Cm[t][None, :]).max(1)
    j = int(np.argmin(e)); s = np.partition(e, 1)[:2]
    pick.append(j); errs.append(s[0]); second.append(s[1])
    A = A + Wc[j] * P[j]; W += Wc[j]
errs, second = np.array(errs), np.array(second)
print(f"match error: max {errs.max():.2e}, median {np.median(errs):.2e}; "
      f"closest runner-up min {second.min():.2e}; files used {len(set(pick))} of {len(pick)} steps")

rows = []
for (r, c), j in zip(order, pick):
    d = {"path": cand_path[j], "row": r, "col": c, "stage_x": np.nan, "stage_y": np.nan,
         "weight": Wc[j], "__valid": 1}
    d.update({e: P[j, k] * 100.0 for k, e in enumerate(els)})
    rows.append(d)
df = pd.DataFrame(rows).sort_values(["row", "col"]).reset_index(drop=True)
df.to_parquet(out, index=False)
print("wrote", out, df.shape)
