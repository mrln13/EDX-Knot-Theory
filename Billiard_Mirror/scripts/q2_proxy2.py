import importlib.util, numpy as np, pandas as pd
S = "C:/Users/4029771/AppData/Local/Temp/claude/a3300fef5b34d016/dd33cf2e-2b67-4e1b-a664-322805f2111a/scratchpad/"
REPO = "C:/Users/4029771/OneDrive - Universiteit Utrecht/Dev/EDX-Knot-Theory/Data Analysis/"
spec = importlib.util.spec_from_file_location("sim", REPO + "simulate_scan_order_from_cache.py")
sim = importlib.util.module_from_spec(spec); spec.loader.exec_module(sim)
MAJ = ["O", "Si", "Al", "Fe", "Ca", "Mg", "Na", "K", "Ti"]
def clr(P):
    G = np.hstack([P, np.clip(1 - P.sum(-1, keepdims=True), 0, None)]); G = np.clip(G, 1e-12, None)
    G = G / G.sum(-1, keepdims=True); L = np.log(G); return L - L.mean(-1, keepdims=True)
for name, Ns in (("schist", (60, 125, 140)), ("diorite", (100, 150, 175))):
    df = pd.read_parquet(S + f"{name}_cache.parquet")
    R, C = df.row.max() + 1, df.col.max() + 1
    allel = [c for c in df.columns if c not in {"path","row","col","stage_x","stage_y","weight","__valid"}]
    F = np.zeros((R, C, len(MAJ))); tot = np.zeros((R, C))
    F[df.row, df.col] = df[MAJ].to_numpy() / 100; tot[df.row, df.col] = df[allel].sum(1) / 100
    Wt = np.zeros((R, C)); Wt[df.row, df.col] = df.weight
    # cumulative fractions as in analyze_orders (sum_i w_i pct_i/100 / sum w), majors + Other = 1 - sum(majors)
    g = clr((F * Wt[..., None]).sum((0, 1)) / Wt.sum())
    coords = [(r, c) for r in range(R) for c in range(C)]
    print(f"== {name}: CLR distance of running composition after N tiles to full-map composition")
    for lab, canon in (("row snake", sim.order_raster_row(coords, None, snake=True)),
                       ("column snake", sim.order_raster_col(coords, None, snake=True))):
        res = []
        for s in coords:
            o = np.array(sim._rotate_to_start(canon, s)[:max(Ns)])
            f, w = F[o[:, 0], o[:, 1]], Wt[o[:, 0], o[:, 1]]
            cf, cw = np.cumsum(f * w[:, None], 0), np.cumsum(w)
            res.append([np.sqrt(((clr(cf[N - 1] / cw[N - 1]) - g) ** 2).sum()) for N in Ns])
        res = np.array(res)
        print(f"  {lab:13s}", "  ".join(f"N={N}: mean {v:.3f} (median {m:.3f})" for N, v, m in zip(Ns, res.mean(0), np.median(res, 0))))
