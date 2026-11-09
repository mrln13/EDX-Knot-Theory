import numpy as np, pandas as pd
S = "C:/Users/4029771/AppData/Local/Temp/claude/a3300fef5b34d016/dd33cf2e-2b67-4e1b-a664-322805f2111a/scratchpad/"
MAJ = ["O", "Si", "Al", "Fe", "Ca", "Mg", "Na", "K", "Ti"]
for name in ("diorite", "schist"):
    df = pd.read_parquet(S + f"{name}_cache.parquet")
    R, C = df.row.max() + 1, df.col.max() + 1
    print(f"== {name}: per element, share of tile variance carried by column means vs row means; lag-1 semivariance rows/cols")
    for e in MAJ:
        X = np.full((R, C), np.nan); X[df.row, df.col] = df[e]
        v = np.nanvar(X)
        gh = np.nanmean((X[:, 1:] - X[:, :-1]) ** 2) / (2 * v); gv = np.nanmean((X[1:] - X[:-1]) ** 2) / (2 * v)
        print(f"  {e:3s} mean {np.nanmean(X):5.2f} wt%  col-means {np.nanvar(np.nanmean(X, 0)) / v:.3f}  row-means {np.nanvar(np.nanmean(X, 1)) / v:.3f}   lag1 rows {gh:.2f} cols {gv:.2f}")
    if name == "diorite":
        for e in ("Fe", "Mg", "Ca", "K"):
            X = np.full((R, C), np.nan); X[df.row, df.col] = df[e]
            cm = np.nanmean(X, 0)
            print(f"  {e} column means (left->right, every col):", " ".join(f"{v:.1f}" for v in cm))
