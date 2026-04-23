# EDX-Knot-Theory

This repository accompanies the paper **"Convergence-Guided EDS Mapping for Bulk Rock Composition Using Knot Theory"**.

## Overview

We present an adaptive SEM–EDS mapping strategy that combines a deterministic
billiard-knot tiling trajectory with a data-driven convergence criterion. The
goal is to obtain representative bulk rock compositions from large-area EDS
maps more efficiently, with fewer tiles and reduced beam time.

### Abstract (from the manuscript)

Bulk rock composition (BRC) controls phase equilibria, mineral stability and
reconstructed pressure–temperature paths, but is traditionally obtained from
destructive whole-rock analyses that lack spatial context. SEM–EDS mapping
offers a route to BRC that preserves textural information, yet full thin-section
maps are time-consuming and dose-intensive, and conventional raster scans can
introduce grid-alignment bias.

We present an adaptive SEM–EDS mapping strategy that combines a deterministic
billiard-knot tiling trajectory with a data-driven convergence criterion. The
stage follows a periodic billiard path that visits tile centres in a
non-redundant, non-grid-aligned order, promoting uniform coverage and reducing
sensitivity to sample fabric. We track the cumulative bulk composition in
centred log-ratio (clr) space using an exponentially weighted moving average and
monitor the clr-distance between successive states, terminating acquisition once
this metric stabilizes.

We test this approach on two contrasting rocks, a strongly foliated pelitic
schist and a more equigranular diorite. Across both datasets, the billiard-knot
trajectory reaches convergence in fewer tiles than raster and spiral patterns,
reducing tiles-to-convergence by roughly 10–20%. Stable BRC estimates are
obtained after only ~7.7% and ~5.5% of the available tiles in the schist and
diorite mosaics, respectively, with tighter clustering of convergence steps
across starting positions. These gains translate directly into reduced beam time
for large-area EDS mapping. The combination of structured non-raster scan paths
with a compositional convergence metric is broadly applicable to other spectral
imaging and streaming compositional data.

---

## Installation

Python ≥ 3.9 and R ≥ 4.2 are recommended.

### Python

```bash
pip install numpy pandas pyarrow matplotlib tifffile scipy tqdm \
            pillow scikit-image scikit-learn hyperspy
```

`hyperspy` is only required for the BCF/SPX ingestion path
(`build_composition_cache.py`). `scikit-learn` is only required for the PCA
RGB composite (`composite_maps.py`).

### R

```r
install.packages(c("jsonlite","dplyr","tidyr","ggplot2","readr","purrr",
                   "tibble","cowplot","patchwork","stringr"))
```

---

## Repository layout

| Script | Purpose | Manuscript link |
|---|---|---|
| `Trajectory Generation/methods.py` | Billiard-knot math, coverage check, tile-grid search | §Tiling Strategy |
| `Trajectory Generation/ScanPatterns.py` | CLI wrapper to visualize/search trajectories | Fig. 1 |
| `Trajectory Generation/PosCalc.py` | Export stage positions for Bruker ESPRIT | §Data Acquisition |
| `Data Analysis/build_composition_cache.py` | Build per-tile composition cache from BCF/SPX files | §Data Acquisition |
| `Data Analysis/build_cache_from_grids.py` | Build per-tile composition cache from pre-exported element-map CSV grids | alternative ingestion path |
| `Data Analysis/simulate_scan_order_from_cache.py` | Generate scan orders (billiard/raster/spiral) over a cache | §Results |
| `Data Analysis/analyze_orders.py` | Aggregate cumulative compositions per scan order into JSON | §Results |
| `Data Analysis/Calc_Stats.R` | EWMA–CLR convergence statistics and ribbon plots | §Evaluating Data Convergence |
| `Data Analysis/plot.R` | Publication-ready multi-panel convergence figure | Fig. 5 |
| `Data Analysis/sensitivity_analysis.R` | Monte Carlo sensitivity analysis on convergence hyperparameters | §Sensitivity Analysis, Fig. 6 |
| `Data Analysis/composite_maps.py` | PCA RGB composite of EDS element maps | Fig. 2b |
| `Data Analysis/stitch.py` | BSE mosaic stitching (helper) | §Data Acquisition |
| `Data Analysis/eds_map.py` | EDS cube / banded map utilities (helper) | — |

---

## End-to-end pipeline

The full workflow follows four stages: **(1)** generate the trajectory and export
stage positions, **(2)** acquire the SEM data, **(3)** build a per-tile
composition cache from the acquired data, and **(4)** evaluate convergence and
generate figures. Stage (2) is performed on the instrument and is outside the
scope of this repository.

### 1. Generate trajectory and export stage positions

Search for a valid `(p, q)` and tile size on a blank canvas:
```bash
python "Trajectory Generation/ScanPatterns.py" \
  --blank-w 2000 --blank-h 1200 \
  --min-tile 50 --max-tile 200 \
  --p0 57 --q0 40 \
  --maximize-reflections \
  --draw-path --mark-center \
  --fig-out pattern.png
```

Export tile-centre stage coordinates as a `positions.ini` file for Bruker
ESPRIT (edit constants at the top of the file first):
```bash
python "Trajectory Generation/PosCalc.py"
```

### 2. Acquire data

Import `positions.ini` into ESPRIT and run the map. This produces per-tile
BCF files with accompanying SPX quantification.

### 3. Build a composition cache

**From BCF/SPX (primary path):**
```bash
python "Data Analysis/build_composition_cache.py" \
  --dir path/to/data_root \
  --out composition_cache.parquet \
  --on-mismatch fail --scope-level 1
```

**From pre-exported wt% CSV grids (alternative path, e.g. when per-pixel
quantification is already done by the vendor software):**
```bash
python "Data Analysis/build_cache_from_grids.py" \
  --dir path/to/element_csvs \
  --p 27 --q 23 \
  --pixel-size 4.0 \
  --out cache.parquet
```

Both produce the same parquet schema (`path, row, col, stage_x, stage_y,
weight, __valid`, element columns).

### 4. Evaluate convergence

Generate scan-order sequences for each pattern:
```bash
python "Data Analysis/simulate_scan_order_from_cache.py" \
  --cache composition_cache.parquet \
  --pattern billiard --p 57 --q 40 \
  --starts-all --workers 8 \
  --out-template "orders/billiard_r{r}_c{c}.csv"
```
Repeat with `--pattern raster --snake 1` and `--pattern spiral --starts-edges`.

Aggregate cumulative compositions:
```bash
python "Data Analysis/analyze_orders.py" \
  --cache composition_cache.parquet \
  --orders "orders/*.csv" \
  --out metrics.json
```

Compute convergence statistics and plot (edit user settings at top of the
script first):
```bash
Rscript "Data Analysis/Calc_Stats.R"
```

### Optional: reproduce remaining manuscript figures

**PCA RGB composite (Fig. 2b):**
```bash
python "Data Analysis/composite_maps.py" --dir path/to/downsampled_element_maps
```

**Sensitivity analysis (§Sensitivity Analysis, Fig. 6):**
The analysis operates on raster-pattern aggregate JSONs. After generating
metrics JSONs for the raster scans of both datasets, edit the `AGGREGATE_JSONS`
paths at the top of the script and run:
```bash
Rscript "Data Analysis/sensitivity_analysis.R"
```

**Multi-panel figure (Fig. 5):**
After running `Calc_Stats.R` for each dataset-pattern combination and saving
the R workspaces, edit the file lists at the top of `plot.R` and run:
```bash
Rscript "Data Analysis/plot.R"
```

---

## Subdirectory READMEs

Per-component details, CLI reference, and troubleshooting are in the
subdirectory READMEs:

- [Trajectory Generation/README.md](Trajectory%20Generation/README.md)
- [Data Analysis/README.md](Data%20Analysis/README.md)
