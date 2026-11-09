"""Per-tile composition cache from SPX quant + stitch_manifest grid positions.

Same composition/weight extraction as build_composition_cache.py
(_spx_quant_for_file), but row/col come from the stitch manifest instead of
loading every BCF for its stage coordinates.
usage: python cache_from_manifest.py <dataset_dir> <out.parquet>
"""
import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(r"C:/Users/4029771/OneDrive - Universiteit Utrecht/Dev/EDX-Knot-Theory/Data Analysis")
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("bcc", REPO / "build_composition_cache.py")
bcc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bcc)

base = Path(sys.argv[1])
out = Path(sys.argv[2])
man = pd.read_csv(base / "stitch_manifest.csv")
print(f"manifest rows {len(man)}, grid {man.row.max() + 1} x {man.col.max() + 1}, "
      f"unique cells {man[['row', 'col']].drop_duplicates().shape[0]}")

recs = []
for _, m in man.iterrows():
    rel = Path(m["file"].replace("\\", "/"))
    inner = re.search(r"\((.*)\)", rel.name).group(1)   # e.g. '12,32' or '1,33[2]'
    name = f"Spectrum_({inner}).spx"
    spx = next((base / rel.parent / sub / name for sub in ("Quantified", "SPX_q", "")
                if (base / rel.parent / sub / name).exists()), base / rel.parent / name)
    if not spx.exists():
        print("missing", spx)
        continue
    comp, w = bcc._spx_quant_for_file(spx)
    r = {"path": str(rel.parent / spx.name), "row": int(m["row"]), "col": int(m["col"]),
         "stage_x": float(m["x"]), "stage_y": float(m["y"]), "weight": float(w),
         "__valid": 1 if (w > 0 and comp) else 0}
    if r["__valid"]:
        r.update(comp)
    recs.append(r)

df = pd.DataFrame(recs)
for c in [c for c in df.columns if c not in {"path", "row", "col", "stage_x", "stage_y", "weight", "__valid"}]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
df = df.sort_values(["row", "col", "path"]).reset_index(drop=True)
df.to_parquet(out, index=False)
print(f"wrote {len(df)} tiles, valid {int(df['__valid'].sum())} -> {out}")
t = df[(df.row == 0) & (df.col == 0)].iloc[0]
els = [c for c in df.columns if c not in {"path", "row", "col", "stage_x", "stage_y", "weight", "__valid"}]
s = sum(t[e] for e in els)
print("tile r0_c0 fractions:", {e: round(t[e] / s, 5) for e in els if t[e] > 0})
