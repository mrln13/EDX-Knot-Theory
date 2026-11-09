"""Write scan orders per start using simulate_scan_order_from_cache.py generators.

usage: python orders.py CACHE PATTERN OUTDIR P Q N STARTS
  PATTERN: billiard | billiardmirror | billiardflip | row | column
    billiardmirror = left-right mirror of the billiard path (second path for schist/diorite)
    billiardflip   = top-bottom mirror of the billiard path (second path for the granulite)
  N: truncate each order to first N tiles (0 = full)
  STARTS: 'all' | 'sample:K' (K starts spread evenly over the grid) | 'r,c;r,c'
"""
import importlib.util
import sys
from pathlib import Path

REPO = Path(r"C:/Users/4029771/OneDrive - Universiteit Utrecht/Dev/EDX-Knot-Theory/Data Analysis")
spec = importlib.util.spec_from_file_location("sim", REPO / "simulate_scan_order_from_cache.py")
sim = importlib.util.module_from_spec(spec); spec.loader.exec_module(sim)

cache, patt, outdir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
p, q, N, starts_arg = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
coords, rc_to_path = sim.load_grid(Path(cache))
cmin = min(c for _, c in coords); cmax = max(c for _, c in coords)

rmin = min(r for r, _ in coords); rmax = max(r for r, _ in coords)

if patt in ("billiard", "billiardmirror", "billiardflip"):
    # For given (p, q) there are two distinct billiard paths. One of the two
    # mirror images is the second path, the other is the original reversed:
    # schist/diorite -> left-right mirror is distinct ('billiardmirror');
    # granulite      -> top-bottom mirror is distinct ('billiardflip').
    canon = sim.order_billiard_methods(coords, p, q, None)
    if patt == "billiardmirror":
        canon = [(r, cmin + cmax - c) for r, c in canon]
    elif patt == "billiardflip":
        canon = [(rmin + rmax - r, c) for r, c in canon]
elif patt == "row":
    canon = sim.order_raster_row(coords, None, snake=True)
elif patt == "column":
    canon = sim.order_raster_col(coords, None, snake=True)
else:
    raise SystemExit(patt)

if starts_arg == "all":
    starts = coords
elif starts_arg.startswith("sample:"):
    k = int(starts_arg.split(":")[1])
    starts = [coords[i] for i in range(0, len(coords), max(1, len(coords) // k))][:k]
else:
    starts = [tuple(int(v) for v in s.split(",")) for s in starts_arg.split(";")]

outdir.mkdir(parents=True, exist_ok=True)
for s in starts:
    o = sim._rotate_to_start(canon, s)
    if N:
        o = o[:N]
    sim.write_sequence_csv(o, rc_to_path, outdir / f"{patt}_r{s[0]}_c{s[1]}.csv")
print(f"wrote {len(starts)} {patt} orders ({N or 'all'} tiles) -> {outdir}")
