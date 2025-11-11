# eds_map.py
from __future__ import annotations
import csv
from pathlib import Path
from typing import Iterable, Tuple, Dict, List, Optional

import numpy as np
import hyperspy.api as hs
from tqdm import tqdm


# ---------- EDS discovery ----------

def load_eds_signal_lazy(p: Path):
    """Return a lazy HyperSpy EDS signal for a given BCF path, or None if not found."""
    try:
        sigs = hs.load(p, lazy=True)
    except Exception:
        return None

    it = sigs if isinstance(sigs, (list, tuple)) else [sigs]
    for s in it:
        try:
            st = str(s.metadata.Signal.signal_type)
        except Exception:
            st = ""
        if "EDS" in st.upper():
            return s
    return None


# ---------- Metadata helpers ----------

def extract_eds_metadata_from_signal(sig) -> dict:
    """
    Extract beam energy, sample elements/lines, and energy axis from a HyperSpy EDS signal.
    Returns:
      {
        'beam_energy_keV': float|None,
        'elements': [str,...],
        'xray_lines': [str,...],  # e.g. 'Si_Ka'
        'E0': float|None, 'dE': float|None, 'nE': int|None, 'units': str
      }
    """
    out = {
        'beam_energy_keV': None,
        'elements': [],
        'xray_lines': [],
        'E0': None, 'dE': None, 'nE': None, 'units': 'keV',
    }

    md = getattr(sig, 'metadata', None)
    if md is not None:
        # beam energy
        try:
            out['beam_energy_keV'] = float(md.Acquisition_instrument.SEM.beam_energy)
        except Exception:
            try:
                out['beam_energy_keV'] = float(md.get_item('Acquisition_instrument.SEM.beam_energy'))
            except Exception:
                pass
        # sample lists
        try:
            out['elements'] = [str(e) for e in list(md.Sample.elements)]
        except Exception:
            pass
        try:
            out['xray_lines'] = [str(x) for x in list(md.Sample.xray_lines)]
        except Exception:
            pass

    # energy axis
    try:
        ea = sig.axes_manager.signal_axes[0]
        out['E0'] = float(ea.offset)
        out['dE'] = float(ea.scale)
        out['nE'] = int(ea.size)
        out['units'] = str(ea.units) or 'keV'
    except Exception:
        pass
    return out


# ---------- Manifest I/O ----------

def write_tile_manifest(
    base_dir: Path,
    coords_sorted: Iterable[Tuple[int, int]],
    pos: Dict[Tuple[int, int], np.ndarray],
    rc_to_path: Dict[Tuple[int, int], Path],
    tile_shear: Dict[Tuple[int,int], Dict[str, float]],
    th: int,
    tw: int,
) -> Path:
    """
    Write a lightweight manifest mapping mosaic coords → BCF files and EDS metadata.

    Columns:
      row,col,y,x,file,bse_h,bse_w,eds_h,eds_w,scale_y,scale_x,shx,shy,
      E0,dE,nE,energy_units,beam_energy,elements,xray_lines
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    out_csv = base_dir / "stitch_manifest.csv"

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "row","col","y","x","file",
            "bse_h","bse_w","eds_h","eds_w",
            "scale_y","scale_x","shx","shy",
            "E0","dE","nE","energy_units",
            "beam_energy","elements","xray_lines",
        ])

        for rc in coords_sorted:
            p = rc_to_path[rc]
            y, x = pos[rc]
            shx = float(tile_shear.get(rc, {}).get("shx", 0.0))
            shy = float(tile_shear.get(rc, {}).get("shy", 0.0))

            eds_h = eds_w = th
            E0 = 0.0; dE = 1.0; nE = 0; units = "keV"
            beam_energy = ""
            elements_csv = ""
            lines_csv = ""

            eds = load_eds_signal_lazy(p)
            if eds is not None:
                # nav shape
                try:
                    nav = eds.axes_manager.navigation_shape
                    if len(nav) >= 2:
                        eds_h, eds_w = int(nav[0]), int(nav[1])
                except Exception:
                    if hasattr(eds, "data") and hasattr(eds.data, "shape") and len(eds.data.shape) >= 2:
                        eds_h, eds_w = int(eds.data.shape[0]), int(eds.data.shape[1])

                meta = extract_eds_metadata_from_signal(eds)
                E0 = meta.get('E0', E0); dE = meta.get('dE', dE); nE = meta.get('nE', nE); units = meta.get('units', units)
                be = meta.get('beam_energy_keV', None)
                if be is not None:
                    beam_energy = f"{be:.6g}"
                els = meta.get('elements') or []
                lns = meta.get('xray_lines') or []
                if els: elements_csv = ';'.join(els)
                if lns: lines_csv = ';'.join(lns)

            # EDS vs BSE scale
            scale_y = eds_h / float(th) if th > 0 else 1.0
            scale_x = eds_w / float(tw) if tw > 0 else 1.0

            w.writerow([
                rc[0], rc[1], y, x, str(p),
                th, tw, eds_h, eds_w,
                scale_y, scale_x, shx, shy,
                E0, dE, nE, units,
                beam_energy, elements_csv, lines_csv
            ])

    print(f"[manifest] Wrote tile manifest → {out_csv}")
    return out_csv


def read_manifest(manifest_path: Path):
    rows = []
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # numeric casts (tolerant)
            for k in ["row","col","bse_h","bse_w","eds_h","eds_w","nE"]:
                try: row[k] = int(float(row.get(k, "0") or 0))
                except Exception: row[k] = 0
            for k in ["y","x","scale_y","scale_x","shx","shy","E0","dE","beam_energy"]:
                try: row[k] = float(row.get(k, "0") or 0.0)
                except Exception: row[k] = 0.0
            row["file"] = Path(row["file"])
            row["elements_list"] = [s for s in (row.get("elements","") or "").split(";") if s]
            row["xray_lines_list"] = [s for s in (row.get("xray_lines","") or "").split(";") if s]
            rows.append(row)
    return rows


# ---------- EDS ROI → cube ----------

def roi_to_cube(
    manifest_path: Path,
    roi: tuple[int, int, int, int],
    apply_subpixel: bool = True,
    feather: bool = True,
    blend_mode: str = "mean",   # "mean" or "sum"
    tukey_alpha: float = 0.08,
):
    """
    Assemble an EDS cube (H, W, nE) for ROI in BSE pixel coords.
    Align each tile exactly as in BSE composite:
      1) resample EDS nav grid -> BSE tile size (uses scale_y/scale_x),
      2) per-tile shear (shx, shy),
      3) sub-pixel shift from fractional (y, x).
    """
    import hyperspy.api as hs
    from scipy.ndimage import affine_transform
    from scipy.signal.windows import tukey as tukey1d
    from skimage.transform import resize

    rows = read_manifest(manifest_path)
    if not rows:
        raise RuntimeError("Manifest is empty.")

    # Common energy axis
    E0 = dE = None; nE = None; units = "keV"
    for r in rows:
        sig = load_eds_signal_lazy(r["file"])
        if sig is None:
            continue
        meta = extract_eds_metadata_from_signal(sig)
        if meta.get("nE", 0) > 0:
            E0 = meta["E0"]; dE = meta["dE"]; nE = meta["nE"]
            units = meta.get("energy_units", "keV") or "keV"
            break
    if nE is None:
        raise RuntimeError("No EDS data found in manifest.")

    y0, y1, x0, x1 = map(int, roi)
    H = max(0, y1 - y0); W = max(0, x1 - x0)
    if H <= 0 or W <= 0:
        raise ValueError("ROI must have positive size.")

    accum  = np.zeros((H, W, nE), dtype=np.float32)
    weight = np.zeros((H, W),      dtype=np.float32)

    def tukey_window(h, w, a):
        wy = tukey1d(h, alpha=a).astype(np.float32)
        wx = tukey1d(w, alpha=a).astype(np.float32)
        return np.outer(wy, wx).astype(np.float32)

    for r in rows:
        sig = load_eds_signal_lazy(r["file"])
        if sig is None:
            continue
        meta = extract_eds_metadata_from_signal(sig)
        if not (meta.get("nE",0) == nE and abs(meta.get("E0",0)-E0) < 1e-6 and abs(meta.get("dE",0)-dE) < 1e-9):
            continue  # skip mismatched energy axis

        # Data as (h,w,nE)
        arr = np.asarray(sig.data)
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.ndim == 3 and arr.shape[0] == nE:
            arr = np.moveaxis(arr, 0, 2)
        if arr.ndim != 3 or arr.shape[2] != nE:
            continue

        th = int(r["bse_h"]); tw = int(r["bse_w"])
        sy = float(r.get("scale_y", 1.0) or 1.0)
        sx = float(r.get("scale_x", 1.0) or 1.0)
        shx = float(r.get("shx", 0.0) or 0.0)
        shy = float(r.get("shy", 0.0) or 0.0)
        ty  = float(r["y"]); tx = float(r["x"])

        # (1) Resample EDS -> BSE tile size
        tgt_h = max(1, int(round(th * sy)))
        tgt_w = max(1, int(round(tw * sx)))
        if arr.shape[0] != tgt_h or arr.shape[1] != tgt_w:
            tmp = np.empty((tgt_h, tgt_w, nE), dtype=np.float32)
            for k in range(nE):
                tmp[..., k] = resize(arr[..., k], (tgt_h, tgt_w), order=1, preserve_range=True, anti_aliasing=False)
            arr = tmp
        if arr.shape[0] != th or arr.shape[1] != tw:
            tmp = np.empty((th, tw, nE), dtype=np.float32)
            for k in range(nE):
                tmp[..., k] = resize(arr[..., k], (th, tw), order=1, preserve_range=True, anti_aliasing=False)
            arr = tmp

        # (2) Apply tiny per-tile shear (like BSE composite)
        if abs(shx) > 1e-7 or abs(shy) > 1e-7:
            M = np.array([[1.0, shx],
                          [shy, 1.0]], dtype=np.float64)
            Minv = np.linalg.inv(M)
            for k in range(nE):
                arr[..., k] = affine_transform(
                    arr[..., k], Minv, offset=0.0, order=1,
                    mode="nearest", cval=0.0, prefilter=False
                ).astype(np.float32)

        # (3) Sub-pixel shift from fractional (y,x)
        if apply_subpixel:
            fy = ty - np.floor(ty); fx = tx - np.floor(tx)
            if abs(fy) > 1e-6 or abs(fx) > 1e-6:
                from scipy.ndimage import fourier_shift
                for k in range(nE):
                    arr[..., k] = np.fft.ifftn(
                        fourier_shift(np.fft.fftn(arr[..., k]), (fy, fx))
                    ).real.astype(np.float32)

        # Intersect with ROI and accumulate
        iy = int(np.floor(ty)); ix = int(np.floor(tx))
        yA0 = max(0,   y0 - iy); yA1 = min(th, y1 - iy)
        xA0 = max(0,   x0 - ix); xA1 = min(tw, x1 - ix)
        if yA1 <= yA0 or xA1 <= xA0:
            continue

        yB0 = yA0 + (iy - y0); yB1 = yB0 + (yA1 - yA0)
        xB0 = xA0 + (ix - x0); xB1 = xB0 + (xA1 - xA0)

        patch = arr[yA0:yA1, xA0:xA1, :]

        if feather:
            ww_full = tukey_window(th, tw, tukey_alpha)
            ww = ww_full[yA0:yA1, xA0:xA1].astype(np.float32)
        else:
            ww = np.ones((yA1 - yA0, xA1 - xA0), dtype=np.float32)

        for k in range(nE):
            accum[yB0:yB1, xB0:xB1, k] += patch[..., k] * ww
        weight[yB0:yB1, xB0:xB1] += ww

    # Mean normalization
    if blend_mode != "sum":
        m = weight > 0
        if m.any():
            invw = np.zeros_like(weight, dtype=np.float32)
            invw[m] = 1.0 / weight[m]
            for k in range(nE):
                plane = accum[..., k]
                plane[m] = plane[m] * invw[m]
                accum[..., k] = plane

    return accum, (E0, dE, nE, units)

def extract_eds_roi(
    manifest_path: Path,
    roi: tuple[int, int, int, int],
    out_path: Path,
    apply_subpixel: bool = False,
    feather: bool = True,
    blend_mode: str = "mean",
):
    """Assemble an EDS cube for a mosaic ROI and save it as a HyperSpy .hspy file.
    Thin wrapper over `roi_to_cube` kept for backward-compatibility with stitch.py.
    """
    cube, (E0, dE, nE, units) = roi_to_cube(
        manifest_path,
        roi,
        apply_subpixel=apply_subpixel,
        feather=feather,
        blend_mode=blend_mode,
    )
    import hyperspy.api as hs
    sig = hs.signals.EDSSEMSpectrum(cube)
    sig.axes_manager.navigation_axes[0].name = 'y'
    sig.axes_manager.navigation_axes[1].name = 'x'
    sig.axes_manager.signal_axes[0].name = 'Energy'
    sig.axes_manager.signal_axes[0].units = units
    sig.axes_manager.signal_axes[0].offset = E0
    sig.axes_manager.signal_axes[0].scale = dE
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sig.save(out_path)
    print(f"[eds] ROI saved → {out_path}")


# Reuse the same line-energy table as in the viewer
LINE_ENERGIES_KEV = {
    'B': 0.183, 'C': 0.277, 'N': 0.392, 'O': 0.525,
    'F': 0.676, 'Na': 1.041, 'Mg': 1.254, 'Al': 1.486, 'Si': 1.740,
    'P': 2.013, 'S': 2.307, 'Cl': 2.622, 'K': 3.312, 'Ca': 3.691,
    'Ti': 4.511, 'V': 4.952, 'Cr': 5.415, 'Mn': 5.900, 'Fe': 6.404,
    'Co': 6.930, 'Ni': 7.478, 'Cu': 8.048, 'Zn': 8.638,
    'Ga': 9.251, 'Ge': 9.886, 'As': 10.543, 'Se': 11.222,
    'Br': 11.924, 'Kr': 12.657,
    'Ag': 2.984, 'Cd': 3.133, 'Sn': 3.443, 'Sb': 3.605, 'Te': 3.769,
    'Ba': 4.466, 'W': 8.398, 'Au': 9.713, 'Pb': 10.551, 'U': 13.614,
}

def _band_indices(E0: float, dE: float, nE: int, E_center_keV: float, width_eV: float) -> Tuple[int,int]:
    half = 0.5*width_eV/1000.0
    lo = max(0, int(np.floor((E_center_keV - half - E0)/dE)))
    hi = min(nE, int(np.ceil((E_center_keV + half - E0)/dE)))
    if hi <= lo:
        hi = min(nE, lo+1)
    return lo, hi

def _sideband_centers(E_center_keV: float, width_eV: float, mult: float=1.5) -> Tuple[float,float]:
    # symmetric sidebands centered ±mult * window_width away (in keV)
    delta_keV = mult * width_eV / 1000.0
    return E_center_keV - delta_keV, E_center_keV + delta_keV


def _prefilter_rows_for_roi(rows: List[dict], roi: Tuple[int,int,int,int]) -> List[dict]:
    """Return only manifest rows that intersect the ROI (fast, no I/O)."""
    y0, y1, x0, x1 = map(int, roi)
    kept = []
    for r in rows:
        th = int(r["bse_h"]); tw = int(r["bse_w"])
        iy = int(np.floor(float(r["y"]))); ix = int(np.floor(float(r["x"])))
        # AABB intersection (with 1 px slack)
        if (iy + th + 1) <= y0 or iy >= (y1 + 1):
            continue
        if (ix + tw + 1) <= x0 or ix >= (x1 + 1):
            continue
        kept.append(r)
    return kept


def roi_to_element_maps(
    manifest_path: Path,
    roi: Tuple[int,int,int,int],
    elements: List[str],
    width_eV: float = 150.0,
    apply_subpixel: bool = True,
    feather: bool = True,
    tukey_alpha: float = 0.08,
    blend_mode: str = "mean",   # "mean" or "sum"
    progress: bool = True,
    roi_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Memory-lean: returns ONLY per-element 2D maps for the ROI (no HxWxnE cube).
    Applies scale → shear → sub-pixel (same as BSE compositor).
    """
    import hyperspy.api as hs
    from scipy.ndimage import affine_transform
    from scipy.signal.windows import tukey as tukey1d
    from skimage.transform import resize
    from tqdm import tqdm

    rows = read_manifest(manifest_path)
    if not rows:
        raise RuntimeError("Manifest is empty; run stitching first.")

    # Determine common energy axis once
    E0 = dE = None; nE = None; units = "keV"
    for r in rows:
        sig = load_eds_signal_lazy(r["file"])
        if sig is None:
            continue
        meta = extract_eds_metadata_from_signal(sig)
        if meta.get("nE", 0) > 0:
            E0, dE, nE = meta["E0"], meta["dE"], meta["nE"]
            units = meta.get("energy_units", "keV") or "keV"
            break
    if nE is None:
        raise RuntimeError("No EDS signals found in manifest.")

    # ROI dims
    y0, y1, x0, x1 = map(int, roi)
    H = max(0, y1 - y0); W = max(0, x1 - x0)
    if H <= 0 or W <= 0:
        raise ValueError("ROI must have positive size.")

    # Elements + band indices
    elements = [e for e in elements if e in LINE_ENERGIES_KEV]
    if not elements:
        raise ValueError("No valid elements requested.")
    band_idx = {e: _band_indices(E0, dE, nE, LINE_ENERGIES_KEV[e], width_eV) for e in elements}

    # >>> pre-filter intersecting tiles (no I/O)
    roi_rows = _prefilter_rows_for_roi(rows, roi)

    # Accumulators
    accum = {e: np.zeros((H, W), dtype=np.float32) for e in elements}
    weight = np.zeros((H, W), dtype=np.float32)

    def tukey_window(h, w, a):
        wy = tukey1d(h, alpha=a).astype(np.float32)
        wx = tukey1d(w, alpha=a).astype(np.float32)
        return np.outer(wy, wx).astype(np.float32)

    iterator = tqdm(roi_rows, desc="ROI EDS (elements)", unit="tile", total=len(roi_rows)) if progress else roi_rows
    for r in iterator:
        sig = load_eds_signal_lazy(r["file"])
        if sig is None:
            continue
        meta = extract_eds_metadata_from_signal(sig)
        if not (meta.get("nE",0) == nE and abs(meta.get("E0",0)-E0) < 1e-6 and abs(meta.get("dE",0)-dE) < 1e-9):
            continue

        th = int(r["bse_h"]); tw = int(r["bse_w"])
        sy = float(r.get("scale_y", 1.0) or 1.0)
        sx = float(r.get("scale_x", 1.0) or 1.0)
        shx = float(r.get("shx", 0.0) or 0.0)
        shy = float(r.get("shy", 0.0) or 0.0)
        ty  = float(r["y"]);       tx  = float(r["x"])

        iy = int(np.floor(ty)); ix = int(np.floor(tx))
        yA0 = max(0, y0 - iy); yA1 = min(th, y1 - iy)
        xA0 = max(0, x0 - ix); xA1 = min(tw, x1 - ix)
        if yA1 <= yA0 or xA1 <= xA0:
            continue

        yB0 = yA0 + (iy - y0); yB1 = yB0 + (yA1 - yA0)
        xB0 = xA0 + (ix - x0); xB1 = xB0 + (xA1 - xA0)

        ww_full = tukey_window(th, tw, tukey_alpha) if feather else np.ones((th, tw), np.float32)
        ww = ww_full[yA0:yA1, xA0:xA1].astype(np.float32)

        fy = ty - np.floor(ty); fx = tx - np.floor(tx)
        use_subpx = apply_subpixel and (abs(fy) > 1e-6 or abs(fx) > 1e-6)
        if use_subpx:
            from scipy.ndimage import fourier_shift

        for el in elements:
            lo, hi = band_idx[el]
            try:
                img = sig.isig[lo:hi].sum(axis='Energy').data
                img = np.asarray(img, dtype=np.float32)
            except Exception:
                continue
            if img.ndim != 2:
                continue

            # scale → shear → subpixel
            tgt_h = max(1, int(round(th*sy))); tgt_w = max(1, int(round(tw*sx)))
            if img.shape != (tgt_h, tgt_w):
                img = resize(img, (tgt_h, tgt_w), order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
            if img.shape != (th, tw):
                img = resize(img, (th, tw), order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)

            if abs(shx) > 1e-7 or abs(shy) > 1e-7:
                M = np.array([[1.0, shx], [shy, 1.0]], dtype=np.float64)
                Minv = np.linalg.inv(M)
                img = affine_transform(img, Minv, offset=0.0, order=1, mode='nearest', cval=0.0, prefilter=False).astype(np.float32)

            if use_subpx:
                img = np.fft.ifftn(fourier_shift(np.fft.fftn(img), (fy, fx))).real.astype(np.float32)

            patch = img[yA0:yA1, xA0:xA1]

            if roi_mask is not None:
                mask_sub = roi_mask[yB0:yB1, xB0:xB1]
                if mask_sub.shape != ww.shape:
                    # Should not happen, but guard anyway
                    mh, mw = mask_sub.shape
                    ww = ww[:mh, :mw]
                    patch = patch[:mh, :mw]
                    yB1 = yB0 + mh
                    xB1 = xB0 + mw
                ww_eff = ww * mask_sub.astype(np.float32)
            else:
                ww_eff = ww

            accum[el][yB0:yB1, xB0:xB1] += patch * ww_eff

            # weight once per tile (shared)
            weight[yB0:yB1, xB0:xB1] += ww_eff

    if blend_mode != "sum":
        m = weight > 0
        if m.any():
            invw = np.zeros_like(weight, dtype=np.float32)
            invw[m] = 1.0 / weight[m]
            for el in elements:
                tmp = accum[el]
                tmp[m] = tmp[m] * invw[m]
                accum[el] = tmp

    return accum



def roi_to_banded_maps(
    manifest_path: Path,
    roi: Tuple[int,int,int,int],
    elements: List[str],
    width_eV: float = 150.0,
    sideband_mult: float = 1.5,
    apply_subpixel: bool = True,
    feather: bool = True,
    tukey_alpha: float = 0.08,
    blend_mode: str = "mean",
    progress: bool = True,
    roi_mask: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns three images per element: {'main','sb_lo','sb_hi'} aligned with BSE.
    Uses only tiles intersecting ROI; progress bar reflects that count.
    """
    import hyperspy.api as hs
    from scipy.ndimage import affine_transform
    from scipy.signal.windows import tukey as tukey1d
    from skimage.transform import resize
    from tqdm import tqdm

    rows = read_manifest(manifest_path)
    if not rows:
        raise RuntimeError("Manifest is empty.")

    # Energy axis
    E0 = dE = None; nE = None
    for r in rows:
        sig = load_eds_signal_lazy(r["file"])
        if sig is None:
            continue
        meta = extract_eds_metadata_from_signal(sig)
        if meta.get("nE", 0) > 0:
            E0, dE, nE = meta["E0"], meta["dE"], meta["nE"]
            break
    if nE is None:
        raise RuntimeError("No EDS signals found in manifest.")

    y0, y1, x0, x1 = map(int, roi)
    H = max(0, y1 - y0); W = max(0, x1 - x0)
    if H <= 0 or W <= 0:
        raise ValueError("ROI must have positive size.")

    elements = [e for e in elements if e in LINE_ENERGIES_KEV]
    if not elements:
        raise ValueError("No valid elements requested.")

    # Precompute bands per element
    bands = {}
    for e in elements:
        Ec = LINE_ENERGIES_KEV[e]
        lo_main, hi_main = _band_indices(E0, dE, nE, Ec, width_eV)
        Ec_lo, Ec_hi = _sideband_centers(Ec, width_eV, sideband_mult)
        lo_lo, hi_lo = _band_indices(E0, dE, nE, Ec_lo, width_eV)
        lo_hi, hi_hi = _band_indices(E0, dE, nE, Ec_hi, width_eV)
        bands[e] = {'main': (lo_main, hi_main), 'sb_lo': (lo_lo, hi_lo), 'sb_hi': (lo_hi, hi_hi)}

    # >>> pre-filter intersecting tiles
    roi_rows = _prefilter_rows_for_roi(rows, roi)

    accum: Dict[str, Dict[str, np.ndarray]] = {
        e: {k: np.zeros((H, W), dtype=np.float32) for k in ('main','sb_lo','sb_hi')}
        for e in elements
    }
    weight = np.zeros((H, W), dtype=np.float32)

    def tukey_window(h, w, a):
        wy = tukey1d(h, alpha=a).astype(np.float32)
        wx = tukey1d(w, alpha=a).astype(np.float32)
        return np.outer(wy, wx).astype(np.float32)

    iterator = tqdm(roi_rows, desc="ROI EDS (bands)", unit="tile", total=len(roi_rows)) if progress else roi_rows
    for r in iterator:
        sig = load_eds_signal_lazy(r["file"])
        if sig is None:
            continue
        meta = extract_eds_metadata_from_signal(sig)
        if not (meta.get("nE",0) == nE and abs(meta.get("E0",0)-E0) < 1e-6 and abs(meta.get("dE",0)-dE) < 1e-9):
            continue

        th = int(r["bse_h"]); tw = int(r["bse_w"])
        sy = float(r.get("scale_y", 1.0) or 1.0)
        sx = float(r.get("scale_x", 1.0) or 1.0)
        shx = float(r.get("shx", 0.0) or 0.0)
        shy = float(r.get("shy", 0.0) or 0.0)
        ty  = float(r["y"]);       tx  = float(r["x"])

        iy = int(np.floor(ty)); ix = int(np.floor(tx))
        yA0 = max(0, y0 - iy); yA1 = min(th, y1 - iy)
        xA0 = max(0, x0 - ix); xA1 = min(tw, x1 - ix)
        if yA1 <= yA0 or xA1 <= xA0:
            continue

        yB0 = yA0 + (iy - y0); yB1 = yB0 + (yA1 - yA0)
        xB0 = xA0 + (ix - x0); xB1 = xB0 + (xA1 - xA0)

        ww_full = tukey_window(th, tw, tukey_alpha) if feather else np.ones((th, tw), np.float32)
        ww = ww_full[yA0:yA1, xA0:xA1].astype(np.float32)

        fy = ty - np.floor(ty); fx = tx - np.floor(tx)
        use_subpx = apply_subpixel and (abs(fy) > 1e-6 or abs(fx) > 1e-6)
        if use_subpx:
            from scipy.ndimage import fourier_shift

        for e in elements:
            for key, (lo, hi) in bands[e].items():
                try:
                    img = sig.isig[lo:hi].sum(axis='Energy').data
                    img = np.asarray(img, dtype=np.float32)
                except Exception:
                    continue
                if img.ndim != 2:
                    continue

                tgt_h = max(1, int(round(th*sy))); tgt_w = max(1, int(round(tw*sx)))
                if img.shape != (tgt_h, tgt_w):
                    img = resize(img, (tgt_h, tgt_w), order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
                if img.shape != (th, tw):
                    img = resize(img, (th, tw), order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)

                if abs(shx) > 1e-7 or abs(shy) > 1e-7:
                    M = np.array([[1.0, shx], [shy, 1.0]], dtype=np.float64)
                    Minv = np.linalg.inv(M)
                    img = affine_transform(img, Minv, offset=0.0, order=1, mode='nearest', cval=0.0, prefilter=False).astype(np.float32)

                if use_subpx:
                    img = np.fft.ifftn(fourier_shift(np.fft.fftn(img), (fy, fx))).real.astype(np.float32)

                patch = img[yA0:yA1, xA0:xA1]

                # --- NEW: mask support
                if roi_mask is not None:
                    mask_sub = roi_mask[yB0:yB1, xB0:xB1]
                    if mask_sub.shape != ww.shape:
                        mh, mw = mask_sub.shape
                        ww = ww[:mh, :mw]
                        patch = patch[:mh, :mw]
                        yB1 = yB0 + mh
                        xB1 = xB0 + mw
                    ww_eff = ww * mask_sub.astype(np.float32)
                else:
                    ww_eff = ww

                accum[e][key][yB0:yB1, xB0:xB1] += patch * ww_eff

                # shared weight
            weight[yB0:yB1, xB0:xB1] += ww_eff

    if blend_mode != "sum":
        m = weight > 0
        if m.any():
            invw = np.zeros_like(weight, dtype=np.float32)
            invw[m] = 1.0 / weight[m]
            for e in elements:
                for key in ('main','sb_lo','sb_hi'):
                    arr = accum[e][key]
                    arr[m] = arr[m] * invw[m]
                    accum[e][key] = arr

    return accum  # dict: element -> {'main':(H,W), 'sb_lo':(H,W), 'sb_hi':(H,W)}