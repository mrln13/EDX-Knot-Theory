#!/usr/bin/env python3
"""
composite_maps.py -- PCA RGB composite of EDS element maps.

Implements the PCA RGB composite figure described in the manuscript
(Dataset Description, schist): stack downsampled per-element maps, treat each
pixel as an observation and each element channel as a variable, z-score
standardize, run PCA, and visualize the first three principal components as
an RGB image (PC1->R, PC2->G, PC3->B).

Usage:
  python composite_maps.py --dir path/to/element_maps [--out pca_rgb.pdf]
"""
from __future__ import annotations
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.io import imread  # handles TIFF/PNG/JPEG/etc.

EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
CLIP_PERCENTILES = (1, 99)


def load_element_maps(folder, extensions=EXTENSIONS):
    """Load all element map images from *folder* and stack them.

    Returns (stack, names) where stack has shape (H, W, N_elements) and names
    holds the element symbols inferred from the filenames (extension stripped).
    """
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    files = sorted(files)

    if len(files) == 0:
        raise RuntimeError(f"No image files with extensions {extensions} found in {folder}")

    print("Found files:")
    for f in files:
        print("  ", os.path.basename(f))

    first_img = imread(files[0])
    if first_img.ndim == 3:
        first_img = first_img[..., 0]
    H, W = first_img.shape
    stack = np.zeros((H, W, len(files)), dtype=np.float32)

    names = []
    for i, path in enumerate(files):
        img = imread(path)
        if img.ndim == 3:
            img = img[..., 0]
        if img.shape != (H, W):
            raise RuntimeError(
                f"Image {path} has shape {img.shape}, expected {(H, W)}. "
                "All element maps must have identical shape."
            )
        stack[..., i] = img.astype(np.float32)
        names.append(os.path.splitext(os.path.basename(path))[0])

    return stack, names


def pca_to_rgb(element_stack, clip_percentiles=(1, 99)):
    """Compute the first three PCs of a stacked element map and map to RGB.

    Each element channel is standardized to zero mean and unit variance before
    PCA. Principal-component images are percentile-clipped and normalized to
    [0, 1] independently.

    Returns (rgb, pcs, pca) where rgb has shape (H, W, 3), pcs is a list of
    the three PC images before normalization, and pca is the fitted PCA object.
    """
    H, W, C = element_stack.shape
    X = element_stack.reshape(-1, C)  # (n_pixels, n_elements)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    pcs = [X_pca[:, i].reshape(H, W) for i in range(3)]

    low_p, high_p = clip_percentiles
    rgb_channels = []
    for pc in pcs:
        lo, hi = np.percentile(pc, [low_p, high_p])
        pc_clipped = np.clip(pc, lo, hi)
        pc_norm = (pc_clipped - lo) / (hi - lo + 1e-9)
        rgb_channels.append(pc_norm)

    rgb = np.stack(rgb_channels, axis=-1)
    return rgb, pcs, pca


def main():
    ap = argparse.ArgumentParser(
        description="Build a PCA RGB composite from a folder of element maps."
    )
    ap.add_argument("--dir", required=True,
                    help="Folder containing the per-element image files")
    ap.add_argument("--out", default="pca_rgb.pdf",
                    help="Output file path (default: pca_rgb.pdf in --dir)")
    ap.add_argument("--clip-low", type=float, default=1.0,
                    help="Lower percentile for contrast clipping of each PC (default: 1)")
    ap.add_argument("--clip-high", type=float, default=99.0,
                    help="Upper percentile for contrast clipping of each PC (default: 99)")
    ap.add_argument("--dpi", type=int, default=300, help="Output DPI (default: 300)")
    args = ap.parse_args()

    folder = Path(args.dir).resolve()
    if not folder.is_dir():
        raise SystemExit(f"[error] Not a directory: {folder}")

    element_stack, element_names = load_element_maps(str(folder), EXTENSIONS)
    print(f"\nLoaded stack with shape {element_stack.shape} (H, W, N_elements)")
    print("Element channels (in order):", element_names)

    rgb, pcs, pca = pca_to_rgb(
        element_stack, clip_percentiles=(args.clip_low, args.clip_high)
    )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = folder / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=args.dpi)
    ax.imshow(rgb)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"\nSaved PCA RGB composite to: {out_path}")

    print("\nExplained variance ratio (PC1, PC2, PC3):")
    print(pca.explained_variance_ratio_)

    print("\nPCA loadings (rows=PC1..PC3, cols=elements):")
    print("Columns correspond to:", element_names)
    print(pca.components_)


if __name__ == "__main__":
    main()
