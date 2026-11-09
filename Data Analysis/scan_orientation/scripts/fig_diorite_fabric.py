"""Visualise the column-parallel compositional fabric of the diorite tile grid.

Panels
  a  K wt% per tile (row 0 = top of the grid, as in the scan-order cache)
  b  2D spatial autocorrelation of the tile CLR composition (9 majors + rest)
  c  directional semivariogram of the CLR composition: along rows vs down columns
  d  per element: variance of column means and of row means, each scaled so that
     a spatially random arrangement of the same tiles gives 1 (with the 95%
     permutation range shown as a band)

usage: python fig_diorite_fabric.py CACHE.parquet OUT_STEM
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

MAJ = ["O", "Si", "Al", "Fe", "Ca", "Mg", "Na", "K", "Ti"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ROWS_C, COLS_C = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
SEQ = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div", ["#0d366b", "#3987e5", "#f0efec", "#e8794f", "#a8321b"])

cache, stem = sys.argv[1], sys.argv[2]
df = pd.read_parquet(cache)
df = df[df["__valid"] == 1]
R, C = df.row.max() + 1, df.col.max() + 1
print(f"grid {R} rows x {C} cols, {len(df)} tiles")


def to_grid(values):
    G = np.full((R, C) + values.shape[1:], np.nan)
    G[df.row.to_numpy(), df.col.to_numpy()] = values
    return G


# tile CLR of (9 majors + rest), centred on the map mean
M = df[MAJ].to_numpy() / 100.0
G = np.hstack([M, np.clip(1 - M.sum(1, keepdims=True), 0, None)])
G = np.clip(G, 1e-12, None); G /= G.sum(1, keepdims=True)
Y = np.log(G); Y -= Y.mean(1, keepdims=True)
Z = to_grid(Y); Z -= np.nanmean(Z.reshape(-1, Z.shape[2]), 0)
tot = np.nanmean((Z ** 2).sum(2))

# b: 2D autocorrelation
LAG = 12
AC = np.full((2 * LAG + 1, 2 * LAG + 1), np.nan)
for dy in range(-LAG, LAG + 1):
    for dx in range(-LAG, LAG + 1):
        a = Z[max(0, -dy):R - max(0, dy), max(0, -dx):C - max(0, dx)]
        b = Z[max(0, dy):R + min(0, dy), max(0, dx):C + min(0, dx)]
        AC[dy + LAG, dx + LAG] = np.nanmean((a * b).sum(2)) / tot

# c: directional semivariogram (1 = uncorrelated)
lags = np.arange(1, 16)
g_row = [np.nanmean(((Z[:, h:] - Z[:, :-h]) ** 2).sum(2)) / (2 * tot) for h in lags]
g_col = [np.nanmean(((Z[h:, :] - Z[:-h, :]) ** 2).sum(2)) / (2 * tot) for h in lags]

# d: variance of column / row means relative to a random arrangement
rng = np.random.default_rng(0)
W = to_grid(df[MAJ].to_numpy())


def mean_var_ratio(Gr):
    v = np.nanvar(Gr.reshape(-1, Gr.shape[2]), 0)
    col = np.nanvar(np.nanmean(Gr, 0), 0) * R / v     # column means average R tiles
    row = np.nanvar(np.nanmean(Gr, 1), 0) * C / v     # row means average C tiles
    return col, row


col_r, row_r = mean_var_ratio(W)
flat = W.reshape(-1, len(MAJ))
perm = np.array([np.concatenate(mean_var_ratio(flat[rng.permutation(len(flat))].reshape(W.shape)))
                 for _ in range(500)])
null_hi = np.percentile(perm, 97.5); null_lo = np.percentile(perm, 2.5)

print("lag-1 semivariance rows %.3f cols %.3f" % (g_row[0], g_col[0]))
print("autocorr (dy=+1,dx=0) %.3f  (dy=0,dx=+1) %.3f" % (AC[LAG + 1, LAG], AC[LAG, LAG + 1]))
print("element  col_ratio  row_ratio")
for e, c_, r_ in zip(MAJ, col_r, row_r):
    print(f"  {e:3s} {c_:8.1f} {r_:8.1f}")
print(f"permutation 95% range {null_lo:.2f}-{null_hi:.2f}")

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
                     "text.color": INK, "figure.facecolor": "white"})
fig = plt.figure(figsize=(7.2, 6.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], hspace=0.42, wspace=0.5)

ax = fig.add_subplot(gs[0, 0])
K = to_grid(df["K"].to_numpy())
im = ax.imshow(K, cmap=SEQ, vmin=np.nanpercentile(K, 2), vmax=np.nanpercentile(K, 98),
               interpolation="nearest")
ax.set_title("a  K per tile", loc="left", fontweight="bold")
ax.set_xlabel(f"tile column (1-{C})"); ax.set_ylabel(f"tile row (1-{R})")
ax.set_xticks([0, C - 1], [1, C]); ax.set_yticks([0, R - 1], [1, R])
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("K (wt%)"); cb.outline.set_visible(False)

ax = fig.add_subplot(gs[0, 1])
ACp = AC.copy(); ACp[LAG, LAG] = np.nan                 # lag 0 is 1 by definition
lim = np.nanmax(np.abs(ACp))
im = ax.imshow(ACp, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim),
               extent=[-LAG - .5, LAG + .5, LAG + .5, -LAG - .5], interpolation="nearest")
ax.contour(np.arange(-LAG, LAG + 1), np.arange(-LAG, LAG + 1), ACp, levels=[0.1, 0.2, 0.3],
           colors=INK, linewidths=0.6)
ax.set_title("b  Spatial autocorrelation", loc="left", fontweight="bold")
ax.set_xlabel("offset along row (tiles)"); ax.set_ylabel("offset down column (tiles)")
ax.set_xticks([-10, -5, 0, 5, 10]); ax.set_yticks([-10, -5, 0, 5, 10])
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("correlation (CLR)"); cb.outline.set_visible(False)
ax.text(0.03, 0.03, "contours 0.1, 0.2, 0.3", transform=ax.transAxes, color=INK2, fontsize=7)

ax = fig.add_subplot(gs[1, 0])
ax.axhline(1, color=INK2, lw=0.8, ls=(0, (3, 3)))
ax.plot(lags, g_row, color=ROWS_C, lw=2, marker="o", ms=4, label="along rows")
ax.plot(lags, g_col, color=COLS_C, lw=2, marker="o", ms=4, label="down columns")
ax.text(3.3, g_row[2] + 0.012, "along rows", color=INK, va="bottom", ha="right", fontsize=8)
ax.text(3.7, g_col[2] - 0.012, "down columns", color=INK, va="top", ha="left", fontsize=8)
ax.text(1, 1.01, "uncorrelated", color=INK2, fontsize=7, va="bottom")
ax.set_xlim(0.5, lags[-1] + 0.5); ax.set_ylim(0.5, 1.08)
ax.set_xticks([1, 5, 10, 15])
ax.set_title("c  Directional semivariogram", loc="left", fontweight="bold")
ax.set_xlabel("lag (tiles)"); ax.set_ylabel("normalised semivariance (CLR)")
ax.legend(frameon=False, loc="lower right", fontsize=7.5)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax = fig.add_subplot(gs[1, 1])
order = np.argsort(col_r)
yy = np.arange(len(MAJ))
ax.axvspan(null_lo, null_hi, color=GRID, lw=0)
ax.text((null_lo + null_hi) / 2, len(MAJ) / 2 - 0.5, "random arrangement (95%)", color=INK2,
        fontsize=7, rotation=90, ha="center", va="center")
for i, k in enumerate(order):
    ax.plot([row_r[k], col_r[k]], [i, i], color="#c3c2b7", lw=1, zorder=1)
ax.scatter(row_r[order], yy, s=28, color=ROWS_C, edgecolor="white", lw=0.8, zorder=2, label="row means")
ax.scatter(col_r[order], yy, s=28, color=COLS_C, edgecolor="white", lw=0.8, zorder=2, label="column means")
ax.set_xlim(0, 7); ax.set_xticks([0, 1, 2, 4, 6])
ax.set_yticks(yy, [MAJ[k] for k in order])
ax.set_title("d  Variance of row/column means", loc="left", fontweight="bold")
ax.set_xlabel("variance / random expectation (wt%)")
ax.legend(frameon=False, loc="lower right", fontsize=7.5, handletextpad=0.2)
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.savefig(stem + ".png", dpi=300, bbox_inches="tight")
fig.savefig(stem + ".pdf", bbox_inches="tight")
print("wrote", stem + ".png/.pdf")
