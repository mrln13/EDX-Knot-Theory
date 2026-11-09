# EDX-Knot-Theory

This repository accompanies the manuscript **"When to Stop Scanning: Convergence-Guided Trajectories for Efficient Compositional Mapping"** (in preparation).

## Overview

An adaptive SEM–EDS mapping strategy that combines a deterministic
billiard-knot tiling trajectory with a data-driven stopping rule, so that a
representative bulk composition is obtained from a large-area EDS mosaic with
fewer tiles and less beam time than a full map.

### Summary

The approach has two parts. A billiard-knot trajectory visits every tile of a
mosaic exactly once; it moves to a neighbouring tile at each step, but crosses
the whole sample in long diagonal passes rather than filling it block by block.
A stopping rule tracks the cumulative bulk composition in centred log-ratio
(clr) space with an exponentially weighted moving average, and ends the
acquisition once successive estimates stop changing.

The code here generates the trajectories and exports stage positions, builds
per-tile composition caches from acquired data, emulates billiard, raster and
spiral scans from every starting position and in both scan orientations, and
computes the convergence statistics and figures. It was applied to three
thin-section mosaics: a foliated schist, an equigranular diorite, and a
granulite whose compositional domains are comparable in size to the field of
view.

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
| `Trajectory Generation/ScanPatterns.py` | CLI wrapper to visualize/search trajectories | Scan-pattern figure |
| `Trajectory Generation/PosCalc.py` | Export stage positions for Bruker ESPRIT | §Data Acquisition |
| `Data Analysis/build_composition_cache.py` | Build per-tile composition cache from BCF/SPX files | §Data Acquisition |
| `Data Analysis/build_cache_from_grids.py` | Build per-tile composition cache from pre-exported element-map CSV grids | alternative ingestion path |
| `Data Analysis/simulate_scan_order_from_cache.py` | Generate scan orders (billiard/raster/spiral) over a cache | §Results |
| `Data Analysis/analyze_orders.py` | Aggregate cumulative compositions per scan order into JSON | §Results |
| `Data Analysis/Calc_Stats.R` | EWMA–CLR convergence statistics and ribbon plots | §Evaluating Data Convergence |
| `Data Analysis/plot.R` | Multi-panel convergence figure (one to three dataset columns) from one workspace per pattern | earlier convergence figure |
| `Data Analysis/plot_manuscript_pooled.R` | Convergence figure with every pattern pooled over both scan orientations | Convergence figure |
| `Data Analysis/all_variants_stats.R` | Tiles-to-convergence statistics for every pattern and scan orientation | Summary table, appendix table |
| `Data Analysis/fig_coverage_maps.py` | Tiles visited before convergence for one starting position, per pattern | Coverage-map figure |
| `Data Analysis/granulite_figures.py` | Granulite tile grid and scan paths on PCA and greyscale backgrounds | Granulite figure |
| `Data Analysis/scan_orientation/` | Second billiard path, per-start convergence steps, diorite fabric check, block bootstrap | §Results, §Discussion |
| `Data Analysis/sensitivity_analysis.R` | Monte Carlo sensitivity analysis on convergence hyperparameters | §Sensitivity Analysis, sensitivity figure |
| `Data Analysis/composite_maps.py` | PCA RGB composite of EDS element maps | PCA composite figure |
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
Repeat for both orientations of the conventional patterns:
`--pattern raster --snake 1` and `--pattern column --snake 1` (with `--starts-all`),
and `--pattern spiral --clockwise 1` and `--clockwise 0` (with `--starts-edges`).
The second billiard path for the same `(p, q)` is written by
`Data Analysis/scan_orientation/scripts/orders.py` (see the README there).

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

**PCA RGB composite:**
```bash
python "Data Analysis/composite_maps.py" --dir path/to/downsampled_element_maps
```

**Sensitivity analysis (§Sensitivity Analysis):**
The analysis operates on raster-pattern aggregate JSONs. After generating
metrics JSONs for the raster scans of both datasets, edit the `AGGREGATE_JSONS`
paths at the top of the script and run:
```bash
Rscript "Data Analysis/sensitivity_analysis.R"
```

**Convergence figure, both orientations pooled (as in the manuscript):**
After saving a `Calc_Stats.R` workspace (or an `analyze_orders.py` JSON) for
every dataset, pattern and orientation, edit `SOURCES` at the top of the
script and run from the repository root:
```bash
Rscript "Data Analysis/plot_manuscript_pooled.R" convergence_figure.pdf
```
`plot.R` draws the same layout from one workspace per pattern (edit the file
lists and `PLOT_GROUPS` at the top first).

**Statistics for every pattern and orientation:**
Edit the `jobs` list at the top of the script, then run from the repository root:
```bash
Rscript "Data Analysis/all_variants_stats.R" all_variants.csv
```

**Coverage maps for one starting position:**
```bash
python "Data Analysis/fig_coverage_maps.py" --bse path/to/BSE.png \
  --rows 38 --cols 53 --p 38 --q 53 --start 0,52 \
  --out-prefix coverage --panel a:billiard:111:125
```

---

## Subdirectory READMEs

Per-component details, CLI reference, and troubleshooting are in the
subdirectory READMEs:

- [Trajectory Generation/README.md](Trajectory%20Generation/README.md)
- [Data Analysis/README.md](Data%20Analysis/README.md)
