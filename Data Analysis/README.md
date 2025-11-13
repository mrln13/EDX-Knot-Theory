# Data Analysis

This folder contains scripts to **aggregate, analyze, and visualize** scan-order behavior and composition/convergence metrics for SEM‑EDS mosaics. It combines Python (data preparation) and R (statistics + plotting).

The workflow assumes **Bruker ESPRIT** `.BCF` files as inputs, using their embedded stage metadata (and optional EDS metadata).

---

## Contents

### R
- **`Calc_Stats.R`**  
  Reads a **single aggregated JSON** (`{"runs":[...]}`) produced by Python and computes **EWMA–CLR convergence** statistics across many runs. It outputs a combined figure with min–max and P10–P90 ribbons, a median line, and a compact convergence‑steps histogram (no legend/titles).

- **`plot.R`**  
  Produces a **Big Beautiful Figure** (3×2 grid: A left, B right), stacking three rows with separate histogram scaling for the bottom row. Adds global left/right axis titles using `cowplot`, and shows per‑panel mean (solid) and ±1 SD (dashed) vertical lines.

### Python
- **`analyze_orders.py`**  
  Batch‑analyzes cumulative (semi‑quant) composition for **many simulated scan orders _without_ re‑reading BCFs** by joining order CSVs with a cached parquet. Emits **one aggregated JSON** with per‑step composition vectors (fractions, 0..1).

- **`build_composition_cache.py`**  
  Builds a **per‑tile composition cache** (`.parquet`) by pairing `.spx` and `.bcf` files using **scoped keys** derived from folder structure and `(i,j[rep])` tokens in filenames. Composition prefers quantified SPX %wt (fallback: BCF relative intensities). Stage X/Y always come from BCF and are used to compose a coarse grid. Quiet by default; logs only true mismatches or duplicates.

- **`simulate_scan_order_from_cache.py`**  
  Generates **scan orders** over a grid (raster/column/spiral/**billiard**) using only a composition cache or index. Supports many starts, previews, and parallel execution. Can emit `.csv` or `.txt` plus an arrow‑preview PNG.

#### Helper modules

  `eds_map.py` and `stitch.py` are used only for a few specific helper utilities. They are not part of the core data‑analysis pipeline; the Python analysis scripts simply import a couple of functions from them.

- **`eds_map.py`**  
  Utilities to create an **EDS tile manifest** and to assemble **aligned EDS cubes or banded element maps** for a BSE‑aligned ROI. These are used downstream for ROI‑based element visualization or quantitative summaries.

- **`stitch.py`**  
  Full **BSE stitching** pipeline with robust neighbor matching, constrained least‑squares placement, multiscale overlap refinement, tiny per‑tile shear correction, and optional **EDS previews**. Writes a `stitch_manifest.csv` used by `eds_map.py` for ROI extraction.

---

## Typical Workflow

### 1) Build a composition cache (once per dataset)
```bash
python build_composition_cache.py \
  --dir path/to/data_root \
  --out composition_cache.parquet \
  --on-mismatch fail \
  --scope-level 1
```
**Tips**
- `--scope-level` controls how filename keys are grouped by folders (0 = global; 1 = top‑level; 2 = two levels, …).
- If SPX & BCF sets don’t perfectly overlap, choose one of `use_spx | use_bcf | intersect` instead of `fail`.

### 2) Simulate scan orders (optional)
```bash
python simulate_scan_order_from_cache.py \
  --cache composition_cache.parquet \
  --pattern billiard --p 38 --q 53 \
  --starts-edges \
  --workers 8 \
  --out-template "orders/{pattern}_edge_r{r}_c{c}.csv" \
  --preview-template "orders/preview_{pattern}_edge_r{r}_c{c}.png"
```

### 3) Aggregate composition across many orders (Python → JSON)
```bash
python analyze_orders.py \
  --cache composition_cache.parquet \
  --orders "orders/*.csv" \
  --base-dir . \
  --out all_metrics.json
```
**Output structure (`all_metrics.json`)**
```json
{
  "runs": [
    {
      "file": "dataset_total.csv",
      "elements": ["Si","O","..."],
      "composition": [{"Si":0.32,"O":0.55,"...":0.13}, ...],
      "n_tiles": 123
    }
  ]
}
```

### 4) Convergence statistics & ribbons (R → combined plot)
Edit the **user settings** at the top of `Calc_Stats.R`:
```r
AGGREGATE_JSON    <- "path/to/all_metrics.json"
SAVE_PNG_COMBINED <- TRUE
COMBINED_PNG_NAME <- "EWMA_CLR_Combined.png"
```
Then run:
```bash
Rscript Calc_Stats.R
```
This generates a figure with **P10–P90** and **min–max** ribbons, a **median** line, and a **convergence histogram** (scaled to ≤ `HIST_MAX_FRAC` of the main y‑range).

For multi-panel figures (see below) save the workspace.
### 5) Publication‑ready multi‑panel figure (R)
Edit the file lists in `plot.R`:
```r
GROUP_A_FILES <- c("path/to/dataset1a.RData", "path/to/dataset1b.RData", "path/to/dataset1c.RData")
GROUP_B_FILES <- c("path/to/dataset2a.RData", "path/to/dataset2b.RData", "path/to/dataset2c.RData")
OUT_NAME <- "BigBeautifulFigure.pdf"
```
Run:
```bash
Rscript plot.R
```

---

## Key Parameters & Concepts

- **EWMA–CLR** (in `Calc_Stats.R`):  
  - `lambda`: smoothing factor (0–1].  
  - Dynamic window `w_min..w_max` grows with `alpha_grow`.  
  - Convergence requires `L_ok` **consecutive OK windows**, each passing thresholds for `median`, `spread` (P90–P10), and `slope`.
- **Threshold bootstrapping**: early‑window statistics adapt the base thresholds: `theta_med_base`, `theta_spread_base`, `theta_slope_base`.
- **Majors**: set `major_elems` or use top‑`k_majors` by mean fraction (auto). Others collapse into `Other`.
- **Histogram scaling**: histogram counts are mapped to the primary axis via a scale factor so the bars occupy at most `HIST_MAX_FRAC` of `Y_MAX`.
- **Billiard/Lissajous orders** (in `simulate_scan_order_from_cache.py`): exact time‑step alignment with tile centers; supports rotation to arbitrary starts.

---

## Prerequisites

### Python (3.10+ recommended)
Install with `pip` (or conda):
```bash
pip install numpy pandas pyarrow tqdm matplotlib pillow scikit-image scipy hyperspy
```
> *`hyperspy` may require extra system packages; refer to the HyperSpy docs for your OS.*

### R (≥ 4.2 recommended)
Packages used:
```r
install.packages(c("jsonlite","dplyr","tidyr","ggplot2","readr","purrr","tibble","cowplot","patchwork"))
```

---

## Conventions and Tips

- Paths in outputs are **relative** by default where feasible (see `--rewrite-relative-to` and `--absolute` in simulators).  
- Stage coordinates follow the SEM convention used in these scripts (origin top‑right; X leftward; Y downward) when deriving grids from BCF metadata.  
- For large datasets, prefer Parquet I/O and limit R memory by pre‑aggregating in Python.


---

## Troubleshooting

- **“Could not merge order with cache …”** → check that your order CSV has either `row,col` or `filename/path` columns matching the cache.  
- **Empty composition series** → verify `weight` and element columns in the cache are numeric and non‑zero.  
- **R figure looks clipped** → adjust `X_MAX`, `Y_MAX`, and `HIST_MAX_FRAC`.
- Common BCF-related causes include missing or corrupted stage metadata, inconsistent units/signs for X–Y, duplicated or out-of-order tiles, unreadable/cropped files, or ESprit/format-version quirks (e.g., energy-axis mismatches in EDS blocks).
