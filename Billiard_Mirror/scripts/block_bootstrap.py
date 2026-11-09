"""Spatial block bootstrap of tiles-to-convergence differences between patterns.

Runs from neighbouring start tiles share most of their scan sequence, so their
t* values are correlated. We therefore resample square k x k blocks of start
tiles (the same blocks for every pattern, so comparisons are paired), and take
all runs (both orientations) whose start lies in a drawn block.

usage: python block_bootstrap.py all_tstar.csv OUT_PREFIX
"""
import re
import sys

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261109)
B = 10000
KS = [1, 3, 5, 8, 12, 16]
PATTERNS = ["Billiard", "Raster", "Spiral"]
GROUPS = ["Schist", "Diorite", "Granulite"]

df = pd.read_csv(sys.argv[1])
out = sys.argv[2]
rc = df["file"].str.extract(r"r(\d+)_c(\d+)").astype(float)
df["r"], df["c"] = rc[0], rc[1]
assert df[["r", "c"]].notna().all().all(), df[df.r.isna()].head()
df["r"] = df["r"].astype(int); df["c"] = df["c"].astype(int)

# ---------- 1. how far does the correlation between neighbouring starts reach?
ac_rows = []
for (g, s, src), d in df.dropna(subset=["conv_step"]).groupby(["group", "slot", "source"]):
    grid = d.pivot_table(index="r", columns="c", values="conv_step")
    for lag in [1, 2, 3, 5, 8, 12]:
        for axis, name in [(1, "along row"), (0, "down column")]:
            a = grid.to_numpy()
            if axis == 1:
                x, y = a[:, :-lag], a[:, lag:]
            else:
                x, y = a[:-lag, :], a[lag:, :]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() > 20:
                ac_rows.append(dict(group=g, slot=s, source=src, lag=lag, direction=name,
                                    r=np.corrcoef(x[m], y[m])[0, 1], n=int(m.sum())))
ac = pd.DataFrame(ac_rows)
ac.to_csv(out + "_autocorr.csv", index=False)
print("\nMean correlation of t* between start tiles `lag` apart (averaged over sources, both directions):")
print(ac.groupby(["group", "slot", "lag"])["r"].mean().unstack("lag").round(2).to_string())

# ---------- 2. block bootstrap
res = []
for g in GROUPS:
    dg = df[df.group == g]
    conv = dg.dropna(subset=["conv_step"])
    point = {p: conv[conv.slot == p]["conv_step"] for p in PATTERNS}
    pm = {p: point[p].mean() for p in PATTERNS}
    ps = {p: point[p].std(ddof=1) for p in PATTERNS}
    for k in KS:
        blk = (conv["r"] // k).astype(str) + "_" + (conv["c"] // k).astype(str)
        blocks = np.array(sorted((dg["r"] // k).astype(str).add("_").add((dg["c"] // k).astype(str)).unique()))
        bidx = {b: i for i, b in enumerate(blocks)}
        nb = len(blocks)
        # per block, per pattern: count, sum, sum of squares
        stats = {}
        for p in PATTERNS:
            sel = conv.slot == p
            ii = blk[sel].map(bidx).to_numpy()
            v = conv.loc[sel, "conv_step"].to_numpy(float)
            stats[p] = (np.bincount(ii, minlength=nb), np.bincount(ii, v, nb), np.bincount(ii, v * v, nb))
        draws = rng.integers(0, nb, size=(B, nb))
        counts = np.stack([np.bincount(row, minlength=nb) for row in draws])  # B x nb
        boot = {}
        for p in PATTERNS:
            n, s1, s2 = (counts @ x for x in stats[p])
            mean = s1 / n
            var = (s2 - s1 ** 2 / n) / (n - 1)
            boot[p] = (mean, np.sqrt(var), n)
        for other in ["Raster", "Spiral"]:
            dm = boot["Billiard"][0] - boot[other][0]
            sr = boot["Billiard"][1] / boot[other][1]
            res.append(dict(
                group=g, k=k, n_blocks=nb, comparison=f"Billiard - {other}",
                mean_diff=pm["Billiard"] - pm[other],
                diff_lo=np.percentile(dm, 2.5), diff_hi=np.percentile(dm, 97.5),
                p_diff=min(1.0, 2 * min((dm >= 0).mean(), (dm <= 0).mean())),
                sd_ratio=ps["Billiard"] / ps[other],
                ratio_lo=np.percentile(sr, 2.5), ratio_hi=np.percentile(sr, 97.5),
                p_ratio=min(1.0, 2 * min((sr >= 1).mean(), (sr <= 1).mean())),
                min_spiral_runs=int(boot["Spiral"][2].min())))
R = pd.DataFrame(res)
R.to_csv(out + "_bootstrap.csv", index=False)
pd.set_option("display.width", 220)
print("\nBlock bootstrap (B=%d). Differences in tiles; SD ratio billiard/other." % B)
print(R.round(3).to_string(index=False))

# point estimates as a check against Table 1
print("\nPoint estimates (converged runs; never-converged excluded):")
for g in GROUPS:
    conv = df[(df.group == g)].dropna(subset=["conv_step"])
    print(g, {p: (round(conv[conv.slot == p].conv_step.mean(), 1), round(conv[conv.slot == p].conv_step.std(), 1),
                  int((conv.slot == p).sum())) for p in PATTERNS})
