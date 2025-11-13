# Trajectory Generation

Utilities to generate **billiard-knot** scan trajectories, choose tile sizes, and export tile/stage coordinates for SEM–EDS mapping. 

---

## What’s here

- **`methods.py`** — Core library:
  - Billiard-knot path on a rectangle (`billiard_knot`), exact center-hit timestamps (`compute_time_steps`), and LCM/GCD helpers.
  - Tile/grid search that ensures the trajectory visits **every tile center exactly once** under tile-size bounds (`find_valid_tile_size`).
  - Tile placement & visualization (`place_tiles`, `visualize`, `draw_full_path`).
  - Demo convergence utilities using variance/entropy (for illustration only).
- **`ScanPatterns.py`** — Command-line tool to run end-to-end on:
  - a **blank canvas**, or
  - a **real RGB TIFF** (segmented phase map).
  - Supports search mode (find `(p,q)` and tile size) or explicit grid/pairs.
- **`PosCalc.py`** — Export **stage coordinates** (`positions.ini`) for a selected `(p,q)`/grid, given pixel size and stage origin; the file can be imported in Bruker's **ESPRIT** SEM control software.

> **Note on convergence**  
> The manuscript’s convergence analysis uses **EWMA of clr-transformed compositions**. The variance/entropy functions in `methods.py` are **demo helpers** only.

---

## Installation

Python ≥ 3.9 is recommended.

```bash
pip install numpy matplotlib tifffile tqdm scipy
```

(Also uses stdlib modules: `argparse`, `pathlib`, `configparser`, `math`, `collections`.)

---

## Quick start

### A) Generate a trajectory on a blank canvas

Search for a valid grid/tile size and `(p,q)` near `(p0,q0)`:

```bash
python ScanPatterns.py   --blank-w 2000 --blank-h 1200   --min-tile 50 --max-tile 200   --p0 57 --q0 40   --maximize-reflections   --draw-path --mark-center   --limit 1.0   --fig-out pattern.png --fig-title "Billiard tiling"   --legend-label "Normalized Time Step"
```

**Outputs (current folder):**
- `blank_p{p}_q{q}_{W}x{H}_tsmin{min}_tsmax{max}_input.tif` — cropped canvas
- `blank_p{p}_q{q}_..._canvas.tif` — colored visitation map (color = normalized time 0→1)
- `pattern.png` — optional figure with title & colorbar

### B) Run on an existing RGB TIFF

Requires a segmented phase map as input.

```bash
python ScanPatterns.py   --input your_map.tif   --min-tile 50 --max-tile 200   --p0 57 --q0 40   --fig-out your_map_billiard.png
```

> Use `--crop-bottom N` if your TIFF has a bottom legend to remove.

### C) Export stage positions for the instrument

Edit constants in `PosCalc.py`:

```python
min_tile_size, max_tile_size = 768, 1024
p, q = 45, 64
pixel_size = 430  # nm per pixel

# Stage origin in µm (top-left reference)
initial_x = 48000
initial_y = 52000
z = 36357
```

Then:

```bash
python PosCalc.py
```

This writes **`positions.ini`** with:
```
[Global]
PositionCount = N
[Position_1]
X = ...
Y = ...
Z = ...
...
```

Coordinates are in **µm**, ordered by the billiard sequence (tile centers). The generated `positions.ini` can be directly imported into the **Bruker ESPRIT** SEM control software as a stage position list.

---

## Core concepts

- **Billiard-knot path**  
  A piecewise-linear trajectory on the unit square reflecting at boundaries. Integer pair `(p,q)` controls reflection density per axis. With an appropriate grid `(n_x, n_y)`, the path intersects **all tile centers** exactly once.

- **Coverage condition**  
  Coverage depends on `gcd(p,q)` and grid divisibility. The search in `find_valid_tile_size(...)` enforces valid `(p,q, n_x, n_y)` and tile sizes within `[min_tile, max_tile]`.

- **Center-hit timestamps**  
  `compute_time_steps(...)` computes exact normalized times `t ∈ [0,1]` when the path crosses tile centers. These define the acquisition order.

- **Color encoding & origin**  
  Tile color encodes **normalized time** (0–1). The origin is **top-left**.

---

## Command-line reference (`ScanPatterns.py`)

**Inputs**
- `--input PATH` : RGB TIFF. If omitted, a blank canvas is used.
- `--blank-w INT --blank-h INT` : size (pixels) for blank canvas.
- `--crop-bottom INT` : strip bottom rows (e.g., legend).

**Grid & search**
- **Explicit grid**: `--tiles-x INT --tiles-y INT` and `--p INT --q INT` (or `--p0 --q0` as fallback).
- **Search mode**: `--min-tile INT --max-tile INT --p0 INT --q0 INT`  
  Optional: `--maximize-reflections` (tries larger `(p,q)` first → smaller tiles, denser reflections).

**Placement / convergence (demo)**
- `--draw-path` : overlay fine billiard path.
- `--mark-center` : draw center markers (white → red after demo convergence if `--data`).
- `--limit FLOAT` : fraction of normalized time to include (0 < limit ≤ 1).
- `--data` : treat input as data to run demo convergence plots.
- `--roc-threshold FLOAT` : rate-of-change threshold (demo).
- `--min-tiles INT` : minimum tiles before checking convergence (demo).
- `--stop-at-convergence` : stop when demo convergence is detected.
- `--no-show` : skip on-screen plot window.

**Outputs**
- `--out-prefix STR` : file name prefix (default: input stem or `blank`).
- `--fig-out PATH` : save PNG with title & colorbar (`--fig-title`, `--legend-label`).

---

## API highlights (`methods.py`)

```python
find_valid_tile_size(canvas_shape, min_tile, max_tile, p0, q0, maximize_reflections=True)
compute_time_steps(num_tiles_x, num_tiles_y, p, q)
billiard_knot(t, p, q)
place_tiles(canvas, tile_w, tile_h, nx, ny, p, q, t_vals, ...)
visualize(canvas, sm, tile_w, tile_h, p, q)
```

- `place_tiles(...)` returns `[canvas, (optional overlay_if_data)]` and a `ScalarMappable` for colorbars.
- Use `draw_full_path(...)` to overlay the high-resolution billiard curve.

---

## Notes

- **Images are cropped** to be divisible by `(n_x, n_y)` (see `adjust_canvas_size`).
- **Stage origin** in `PosCalc.py` is **top-right**; positions are **tile centers**.
- **Units**: CLI operates in pixels; `PosCalc.py` converts to µm via `pixel_size` (nm/px).
- **Demo convergence** (variance/entropy) ≠ manuscript EWMA-clr metric.

---

## Troubleshooting

- *“No valid tile size found within the bounds.”*  
  Loosen `--min-tile/--max-tile` or try different `--p0/--q0` (toggle `--maximize-reflections`).
- *Canvas looks cropped or misaligned.*  
  Check that `blank-w x blank-h` (or your TIFF) is sensible for your desired `(n_x, n_y)`.
- *Stage coordinates appear offset.*  
  Verify origin convention, `pixel_size`, and the 0.5–tile center offset in `PosCalc.py`.

---

## Reproducibility

- Save the exact command lines and the generated file names (they embed `{p,q}`, sizes, and limits).
- Keep `positions.ini`, the PNG/TIFF outputs, and your input TIFFs under version control for traceability.

---
