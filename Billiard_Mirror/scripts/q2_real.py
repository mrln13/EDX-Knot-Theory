"""Q2 on real per-tile data: directional structure, job layout, weights."""
import sys
import numpy as np
import pandas as pd

S = "C:/Users/4029771/AppData/Local/Temp/claude/a3300fef5b34d016/dd33cf2e-2b67-4e1b-a664-322805f2111a/scratchpad/"
MAJ = ["O", "Si", "Al", "Fe", "Ca", "Mg", "Na", "K", "Ti"]


def grid(df, cols):
    R, C = df.row.max() + 1, df.col.max() + 1
    X = np.full((R, C, len(cols)), np.nan)
    X[df.row.to_numpy(), df.col.to_numpy()] = df[cols].to_numpy()
    return X


def clr_tiles(df):
    M = df[MAJ].to_numpy() / 100.0
    other = np.clip(1 - M.sum(1, keepdims=True), 0, None)
    G = np.hstack([M, other]); G = np.clip(G, 1e-12, None); G /= G.sum(1, keepdims=True)
    L = np.log(G)
    return L - L.mean(1, keepdims=True)


def semivar(Z):
    tot = np.nanmean((Z ** 2).sum(2))
    out = []
    for lag in (1, 2, 3, 5, 8):
        gh = np.nanmean(((Z[:, lag:] - Z[:, :-lag]) ** 2).sum(2)) / (2 * tot)
        gv = np.nanmean(((Z[lag:, :] - Z[:-lag, :]) ** 2).sum(2)) / (2 * tot)
        out.append(f"lag{lag}: rows {gh:.3f} / cols {gv:.3f}")
    return out


for name in ("diorite", "schist"):
    df = pd.read_parquet(S + f"{name}_cache.parquet")
    Y = clr_tiles(df)
    cdf = pd.DataFrame(Y, columns=[f"y{i}" for i in range(Y.shape[1])]); cdf["row"] = df.row; cdf["col"] = df.col
    Z = grid(cdf, [f"y{i}" for i in range(Y.shape[1])])
    Z = (Z - np.nanmean(Z.reshape(-1, Z.shape[2]), 0))
    print(f"\n=== {name}: grid {Z.shape[0]}x{Z.shape[1]}")
    print("  normalised semivariance (1 = uncorrelated):")
    for s in semivar(Z):
        print("   ", s)
    # acquisition job per tile
    df["job"] = df.path.str.extract(r"(Mapping job_[0-9_]+)")[0]
    jobs = sorted(df.job.unique())
    df["jid"] = df.job.map({j: i for i, j in enumerate(jobs)})
    J = np.full((df.row.max() + 1, df.col.max() + 1), -1)
    J[df.row, df.col] = df.jid
    print("  jobs:", {i: (j[-23:], int((df.jid == i).sum())) for i, j in enumerate(jobs)})
    print("  job id per tile (rows top->bottom, every 2nd row/col):")
    for r in range(0, J.shape[0], 2):
        print("   ", "".join(str(v) for v in J[r, ::2]))
    g = df.groupby("jid").agg(weight=("weight", "mean"), O=("O", "mean"), Si=("Si", "mean"), Fe=("Fe", "mean"))
    print("  per job mean weight and wt%:\n", g.round(1).to_string())
    W = np.full(J.shape, np.nan); W[df.row, df.col] = df.weight
    print(f"  weight CV {np.nanstd(W) / np.nanmean(W):.2f}; share of weight variance between rows "
          f"{np.nanvar(np.nanmean(W, 1)) / np.nanvar(W):.2f}, between columns {np.nanvar(np.nanmean(W, 0)) / np.nanvar(W):.2f}")
