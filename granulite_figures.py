#!/usr/bin/env python3
"""
granulite_figures.py -- generate publication-quality tile-placement figures for Granulite.

Produces six 300 DPI PNGs combining:
- backgrounds: PCA RGB composite (from Granulite_Element_Maps/) and grayscale sum-of-elements
- layouts:
  (1) single panel: background + tile grid + centers
  (3) three panels: billiard / raster / spiral acquisition paths over the same grid
  (4) two panels: (a) grid + centers, (b) billiard path from a representative start
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Constants (Granulite-specific)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ELEM_DIR = ROOT / "Granulite_Element_Maps"
CACHE = ROOT / "granulite_cache.parquet"
PIXEL_UM = 3.0
P, Q = 23, 20
NX, NY = Q, P          # 20, 23
TW, TH = 49, 40        # tile w/h in px
W_USED = NX * TW       # 980
H_USED = NY * TH       # 920
SCALEBAR_UM = 500
SCALEBAR_PX = SCALEBAR_UM / PIXEL_UM  # 166.7 px

BILLIARD_CSV  = ROOT / "Granulite_Billiard_All" / "granulite_billiard_r0_c0.csv"
RASTER_CSV    = ROOT / "Granulite_Raster_All"   / "granulite_raster_r0_c0.csv"
SPIRAL_CSV    = ROOT / "Granulite_Spiral_Edges" / "granulite_spiral_r0_c0.csv"

COLOR_BILLIARD = "#ffcc00"
COLOR_RASTER   = "#33aaff"
COLOR_SPIRAL   = "#ff3333"
GRID_COLOR_PCA  = (1.0, 1.0, 1.0, 0.55)
GRID_COLOR_GRAY = (1.0, 0.25, 0.25, 0.85)
CENTER_COLOR_PCA  = (1.0, 1.0, 1.0, 0.9)
CENTER_COLOR_GRAY = (1.0, 0.25, 0.25, 1.0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_element_stack() -> tuple[np.ndarray, list[str]]:
    files = sorted(ELEM_DIR.glob("*.tif"))
    names = [f.stem for f in files]
    arrs = [tifffile.imread(f).astype(np.float32) for f in files]
    stack = np.stack(arrs, axis=-1)
    return stack, names


def compute_pca_rgb(stack: np.ndarray, clip=(1.0, 99.0)) -> np.ndarray:
    H, W, C = stack.shape
    X = stack.reshape(-1, C)
    X_s = StandardScaler().fit_transform(X)
    pca = PCA(n_components=3).fit_transform(X_s)
    rgb = np.zeros((H * W, 3), dtype=np.float32)
    for i in range(3):
        lo, hi = np.percentile(pca[:, i], clip)
        v = np.clip(pca[:, i], lo, hi)
        rgb[:, i] = (v - lo) / (hi - lo + 1e-9)
    return rgb.reshape(H, W, 3)


def compute_grayscale(stack: np.ndarray, clip=(1.0, 99.0)) -> np.ndarray:
    total = stack.sum(axis=-1)
    lo, hi = np.percentile(total, clip)
    g = np.clip(total, lo, hi)
    return (g - lo) / (hi - lo + 1e-9)


def load_order(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df.sort_values("acq_index").reset_index(drop=True)


def tile_centers_px(df: pd.DataFrame) -> np.ndarray:
    """Return (N, 2) array of (x_px, y_px) tile centers from an order dataframe."""
    xs = (df["col"].to_numpy() + 0.5) * TW
    ys = (df["row"].to_numpy() + 0.5) * TH
    return np.column_stack([xs, ys])


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
def draw_background(ax, bg: np.ndarray, is_gray: bool):
    if is_gray:
        ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    else:
        ax.imshow(bg, interpolation="nearest")
    ax.set_xlim(-0.5, bg.shape[1] - 0.5)
    ax.set_ylim(bg.shape[0] - 0.5, -0.5)  # image convention: y flipped
    ax.set_aspect("equal")
    ax.axis("off")


def draw_tile_grid(ax, is_gray: bool):
    color = GRID_COLOR_GRAY if is_gray else GRID_COLOR_PCA
    lw = 0.7
    for c in range(NX + 1):
        x = c * TW
        ax.plot([x, x], [0, H_USED], color=color, lw=lw)
    for r in range(NY + 1):
        y = r * TH
        ax.plot([0, W_USED], [y, y], color=color, lw=lw)


def draw_tile_centers(ax, is_gray: bool):
    color = CENTER_COLOR_GRAY if is_gray else CENTER_COLOR_PCA
    xs = np.array([(c + 0.5) * TW for c in range(NX) for _ in range(NY)])
    ys = np.array([(r + 0.5) * TH for _ in range(NX) for r in range(NY)])
    ax.scatter(xs, ys, s=3, c=[color], linewidths=0)


def draw_path(ax, centers: np.ndarray, color: str):
    ax.plot(centers[:, 0], centers[:, 1], color=color, lw=0.9, alpha=0.95)
    ax.scatter(centers[:, 0], centers[:, 1], s=6, c=color,
               edgecolors="black", linewidths=0.25, zorder=5)
    # mark start
    ax.scatter(centers[:1, 0], centers[:1, 1], s=55, c="#44ff44",
               edgecolors="black", linewidths=0.6, zorder=6)


def draw_scalebar(ax, is_gray: bool, bg_shape: tuple[int, int]):
    H, W = bg_shape
    pad = 22
    x0 = W - pad - SCALEBAR_PX
    y0 = H - pad
    color = "black" if is_gray else "white"
    ax.add_patch(Rectangle((x0, y0 - 6), SCALEBAR_PX, 6, facecolor=color,
                           edgecolor="none"))
    ax.text(x0 + SCALEBAR_PX / 2, y0 - 12, f"{SCALEBAR_UM} µm",
            color=color, ha="center", va="bottom", fontsize=9,
            fontweight="bold")


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------
def fig_layout1(bg: np.ndarray, is_gray: bool, out: Path):
    fig, ax = plt.subplots(figsize=(7, 6.5), dpi=300)
    draw_background(ax, bg, is_gray)
    draw_tile_grid(ax, is_gray)
    draw_tile_centers(ax, is_gray)
    draw_scalebar(ax, is_gray, bg.shape[:2])
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_pca_grid_only(bg: np.ndarray, out: Path):
    """PCA background with just the tile outlines (no centers, no scale bar)."""
    fig, ax = plt.subplots(figsize=(7, 6.5), dpi=300)
    draw_background(ax, bg, is_gray=False)
    draw_tile_grid(ax, is_gray=False)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_layout3(bg: np.ndarray, is_gray: bool, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    patterns = [
        ("Billiard", BILLIARD_CSV, COLOR_BILLIARD),
        ("Raster",   RASTER_CSV,   COLOR_RASTER),
        ("Spiral",   SPIRAL_CSV,   COLOR_SPIRAL),
    ]
    for ax, (name, csv, color) in zip(axes, patterns):
        draw_background(ax, bg, is_gray)
        draw_tile_grid(ax, is_gray)
        draw_path(ax, tile_centers_px(load_order(csv)), color)
        ax.set_title(name, fontsize=13, fontweight="bold")
    draw_scalebar(axes[-1], is_gray, bg.shape[:2])
    plt.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.01, wspace=0.02)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_layout4(bg: np.ndarray, is_gray: bool, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), dpi=300)
    # (a) grid + centers
    draw_background(axes[0], bg, is_gray)
    draw_tile_grid(axes[0], is_gray)
    draw_tile_centers(axes[0], is_gray)
    axes[0].set_title("Tile grid", fontsize=13, fontweight="bold")
    # (b) billiard path
    draw_background(axes[1], bg, is_gray)
    draw_tile_grid(axes[1], is_gray)
    draw_path(axes[1], tile_centers_px(load_order(BILLIARD_CSV)), COLOR_BILLIARD)
    axes[1].set_title("Billiard-knot path (start r0 c0)", fontsize=13, fontweight="bold")
    draw_scalebar(axes[1], is_gray, bg.shape[:2])
    plt.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.01, wspace=0.02)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("[load] Element stack ...")
    stack, names = load_element_stack()
    print(f"  {stack.shape}, elements: {names}")

    print("[compute] PCA RGB ...")
    pca_rgb = compute_pca_rgb(stack)
    print("[compute] Grayscale sum-of-elements ...")
    gray = compute_grayscale(stack)

    tasks = [
        ("PCA",  pca_rgb, False),
        ("Gray", gray,    True),
    ]
    print("[render] Generating figures ...")
    for tag, bg, is_gray in tasks:
        fig_layout1(bg, is_gray, ROOT / f"granulite_tiles_{tag.lower()}.png")
        fig_layout3(bg, is_gray, ROOT / f"granulite_three_patterns_{tag.lower()}.png")
        fig_layout4(bg, is_gray, ROOT / f"granulite_billiard_example_{tag.lower()}.png")
    fig_pca_grid_only(pca_rgb, ROOT / "granulite_pca_grid.png")

    print("[done]")


if __name__ == "__main__":
    main()
