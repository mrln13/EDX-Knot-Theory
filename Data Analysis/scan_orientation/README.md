# Scan orientation

Scripts and small result tables for comparing scan patterns in **both scan
orientations**: row and column serpentine raster, clockwise and
counter-clockwise inward spiral, and the two billiard paths for a given
`(p, q)`.

For a given `(p, q)` the tiling admits two distinct billiard paths. Of the two
mirror images of the evaluated path, one is the second path and the other is the
same path run backwards:

| Dataset (grid) | Left-right mirror | Top-bottom mirror |
|---|---|---|
| Schist (38 x 53) | second path | original reversed |
| Diorite (40 x 57) | second path | original reversed |
| Granulite (23 x 20) | original reversed | second path |

Large inputs are not in git: the per-tile caches
(`*_cache.parquet`), the per-run R workspaces (`*_dfall.RData`) and
`all_runs_tstar.csv`. The scripts below regenerate them.

---

## Scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `orders.py` | Scan orders per start for `billiard`, `billiardmirror` (left-right mirror), `billiardflip` (top-bottom mirror), `row`, `column`, `spiralcw` or `spiralccw`, from every tile, every perimeter tile (`edges`) or a list of starts, optionally truncated to the first N tiles |
| `json_to_dfall.R` | Runs `Calc_Stats.R`'s `process_run` (lines 1-215, unchanged) on an `analyze_orders.py` JSON and saves a compact per-run workspace (`file, step, dist, conv_step`) for `plot_manuscript_pooled.R` |
| `conv_from_json.R` | As above, but writes one convergence step per run to a CSV |
| `cache_from_manifest.py` | Per-tile composition cache from a stitch manifest and quantified SPX files (avoids reading every BCF) |
| `reconstruct_cache.py`, `first_run.py` | Recover the per-tile cache from a stored aggregate JSON by matching each step of its first run; used to verify a rebuilt cache against stored results |
| `fig_diorite_fabric.py` | Directional-similarity figure for the diorite tile grid (K map, 2D autocorrelation, directional semivariogram, row/column-mean variance against a permutation null) |
| `q2_elems.py`, `q2_real.py`, `q2_proxy2.py` | Exploratory checks of why one raster orientation is slower (per-element directional similarity, representativeness of the first tiles) |
| `export_all_tstar.R` | Per-run convergence steps for every source in `plot_manuscript_pooled.R`, written to one CSV |
| `block_bootstrap.py` | Correlation of convergence steps between neighbouring starts, and a spatial block bootstrap of differences in mean and SD between patterns (k x k blocks of starts, paired across patterns) |

`json_to_dfall.R` and `conv_from_json.R` read `Calc_Stats.R` and, in `fixed`
mode, the major-element list from `Sensitivity_analysis.RData` via absolute
paths at the top of each script; edit these before running elsewhere.

### Typical use

Second billiard path from every start, then convergence per run (run from the
repository root):
```bash
python "Data Analysis/scan_orientation/scripts/orders.py" cache.parquet billiardmirror orders_mirror 38 53 600 all
python "Data Analysis/analyze_orders.py" --cache cache.parquet --orders "orders_mirror/*.csv" --out mirror.json
Rscript "Data Analysis/scan_orientation/scripts/json_to_dfall.R" mirror.json mirror_dfall.RData fixed
```

Cumulative-composition files published on Yoda (schist and diorite, six scan
orders each). Billiard and raster orders start from every tile, spirals from
every perimeter tile; all orders are full length. For the schist (use
`diorite_cache.parquet`, 40 57 and `Diorite_` for the diorite):
```bash
DA="Data Analysis"; C="$DA/scan_orientation/schist_cache.parquet"
for v in billiard:Billiard_path1:all billiardmirror:Billiard_path2:all row:Raster_row:all \
         column:Raster_column:all spiralcw:Spiral_CW:edges spiralccw:Spiral_CCW:edges; do
  IFS=: read pat name starts <<< "$v"
  python "$DA/scan_orientation/scripts/orders.py" "$C" $pat orders_$pat 38 53 0 $starts
  python "$DA/analyze_orders.py" --cache "$C" --orders "orders_$pat/*.csv" --out ../Schist_$name.json
done
```
Each full-length billiard or raster file is about 3 GB.

Block bootstrap over all sources of the convergence figure:
```bash
Rscript "Data Analysis/scan_orientation/scripts/export_all_tstar.R" all_runs_tstar.csv
python "Data Analysis/scan_orientation/scripts/block_bootstrap.py" all_runs_tstar.csv bootstrap
```

---

## Result tables

| File | Contents |
|---|---|
| `table1_all_variants.csv` | Output of `../all_variants_stats.R`: statistics per pattern and orientation |
| `billiard_mirror_summary.csv` | Statistics for the original, second and pooled billiard paths |
| `*_billiardmirror_tstar.csv`, `granulite_billiardflip_tstar.csv` | Convergence step per start for the second billiard path |
| `granulite_billiardreversed_tstar.csv` | The granulite billiard path run backwards (not a second path; kept for reference) |
| `block_bootstrap_results.csv` | Bootstrap intervals for mean differences and SD ratios, per dataset and block size |
| `tstar_neighbour_autocorr.csv` | Correlation of convergence steps between starts a given number of tiles apart |
| `diorite_fabric.png`, `diorite_fabric.pdf` | Output of `fig_diorite_fabric.py` |
