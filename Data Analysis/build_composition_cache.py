#!/usr/bin/env python3
"""
build_composition_cache.py — pairing with folder-scoped keys (handles identical filenames in different subfolders)

- Recursively finds .spx/.bcf in subfolders under --dir.
- Pairs by composite key: (scope_prefix, i, j[,rep]), where scope_prefix is the first N folders under --dir.
- Stage X/Y always from BCF; composition prefers quantified SPX Mass% (fallback: BCF relative intensities).
- Console output is SILENT unless there are true mismatches (unpaired keys within the same scope) or duplicates (multiple files for the same scoped key).

Usage:
  python build_composition_cache.py --dir folder_name --out composition_cache.parquet \
         [--on-mismatch fail|use_spx|use_bcf|intersect] [--scope-level N]

Notes:
  • If SPX & BCF that belong together are under different top-level folders, use --scope-level 0 (global).
  • Default --scope-level 1 means pairing is independent per top-level subfolder.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Iterable
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as ET
import hyperspy.api as hs

from stitch import build_grid_from_stage_quantized

# ---------- Coord parsing (supports optional [rep]) ----------
_COORD_RE = re.compile(
    r"""
    \(?                      # optional '('
    \s*(-?\d+)\s*            # row i
    ,\s*                     # comma
    (-?\d+)                  # col j
    (?:\s*\[\s*(-?\d+)\s*\])?  # optional [rep]
    \s*\)?                   # optional ')'
    """,
    re.VERBOSE
)

def _extract_coord_from_name(name: str) -> Optional[Tuple[int, int, Optional[int]]]:
    """Extract the LAST (i,j[rep]) occurrence from a filename."""
    last = None
    for m in _COORD_RE.finditer(name):
        last = m
    if not last:
        return None
    i = int(last.group(1)); j = int(last.group(2))
    rep = last.group(3)
    return (i, j, (int(rep) if rep is not None else None))

# ---------- Scope helpers ----------
def _scope_tuple(p: Path, base: Path, scope_level: int) -> Tuple[str, ...]:
    """
    Return a tuple of the first `scope_level` path parts under base.
    scope_level = 0 → empty tuple (global scope)
    """
    rel = p.resolve().relative_to(base.resolve())
    parts = rel.parts[:-1]  # exclude filename
    if scope_level <= 0:
        return ()
    return tuple(parts[:min(scope_level, len(parts))])

def _scoped_key(p: Path, base: Path, scope_level: int) -> Optional[Tuple[Tuple[str, ...], int, int, Optional[int]]]:
    coords = _extract_coord_from_name(p.name)
    if coords is None:
        return None
    return (_scope_tuple(p, base, scope_level), coords[0], coords[1], coords[2])

def _scoped_key_display(k: Tuple[Tuple[str,...], int, int, Optional[int]]) -> str:
    scope, i, j, rep = k
    scope_str = "/".join(scope) if scope else "."
    ijr = f"({i},{j}[{rep}])" if rep is not None else f"({i},{j})"
    return f"[{scope_str}] {ijr}"

def _scoped_sort_key(k: Tuple[Tuple[str,...], int, int, Optional[int]]):
    scope, i, j, rep = k
    return (tuple(s.lower() for s in scope), i, j, -1 if rep is None else rep)

# ---------- Misc helpers ----------
def _rel_to(p: Path, base: Path) -> Path:
    try:
        return p.relative_to(base)
    except Exception:
        return p

# ---------- HyperSpy helpers ----------
def _first_eds_signal(obj: Any):
    if hasattr(obj, "get_lines_intensity"):
        return obj
    if isinstance(obj, Iterable):
            # iterables returned by hs.load for composite files
        for item in obj:
            if hasattr(item, "get_lines_intensity"):
                return item
    return None

def _extract_stage_xy_from_bcf(path: Path) -> Tuple[float, float]:
    try:
        loaded = hs.load(path)
        s = _first_eds_signal(loaded)
        if s is None:
            return 0.0, 0.0
        md = getattr(s, "metadata", None)
        if md is None:
            return 0.0, 0.0
        try:
            return float(md.Acquisition_instrument.SEM.Stage.X), float(md.Acquisition_instrument.SEM.Stage.Y)
        except Exception:
            pass
        for kx, ky in [
            ("Acquisition_instrument.SEM.Stage.x", "Acquisition_instrument.SEM.Stage.y"),
            ("Acquisition_instrument.SEM.Stage.X", "Acquisition_instrument.SEM.Stage.Y"),
            ("Acquisition_instrument.SEM.stage.x", "Acquisition_instrument.SEM.stage.y"),
        ]:
            try:
                return float(md.get_item(kx)), float(md.get_item(ky))
            except Exception:
                continue
    except Exception:
        pass
    return 0.0, 0.0

def _bcf_relative_composition(path: Path) -> Tuple[Dict[str,float], float]:
    """Fallback composition (BCF): integrate HyperSpy line-intensity maps."""
    try:
        loaded = hs.load(path)
        s = _first_eds_signal(loaded)
        if s is None:
            return {}, 0.0
        try:
            maps = s.get_lines_intensity()
        except Exception:
            maps = []
        if not maps:
            return {}, 0.0

        arrs: List[np.ndarray] = []
        for m in maps:
            a = np.array(getattr(m, "data", m), dtype=np.float64, copy=True)
            if a.ndim == 0:
                a = a.reshape(1)
            arrs.append(a)

        # Element names from metadata if present; else generic
        if hasattr(s, "metadata") and hasattr(s.metadata, "Sample") and getattr(s.metadata, "elements", None):
            names = list(s.metadata.Sample.elements)
            if len(names) != len(arrs):
                names = [f"el{i}" for i in range(len(arrs))]
        else:
            names = [f"el{i}" for i in range(len(arrs))]

        totals = {names[i]: float(np.sum(arrs[i])) for i in range(len(arrs))}
        total_sum = float(sum(totals.values()))
        if total_sum <= 0:
            return {}, 0.0
        comp = {el: 100.0 * v / total_sum for el, v in totals.items()}
        return comp, total_sum
    except Exception:
        return {}, 0.0

# ---------- SPX quantified extraction ----------
_PT = [None,
 'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar',
 'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr',
 'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe',
 'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',
 'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn',
 'Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr',
 'Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og'
]

def _fnum(x):
    if x is None:
        return None
    s = str(x).strip().replace(",", ".").replace("%", "")
    try:
        return float(s)
    except Exception:
        return None

def _spx_quant_for_file(spx_path: Path) -> Tuple[Dict[str,float], float]:
    """Parse quantified SPX: MassPercent (wt%) per element; weight=sum(NetIntens)."""
    try:
        root = ET.parse(spx_path).getroot()
    except Exception:
        return {}, 0.0

    rows = []
    sum_net = 0.0
    for ci in root.iter():
        if ci.attrib.get("Type") == "TRTResult":
            for r in ci.findall("Result"):
                z_txt = (r.findtext("Atom") or "").strip()
                try:
                    Z = int(z_txt)
                except Exception:
                    Z = None
                sym = _PT[Z] if (Z is not None and 0 < Z < len(_PT)) else None
                wt = _fnum(r.findtext("MassPercent"))
                net = _fnum(r.findtext("NetIntens"))
                if net is not None and net > 0:
                    sum_net += net
                if sym and wt is not None:
                    rows.append((sym, wt))
    if not rows:
        return {}, 0.0

    per_el: Dict[str, float] = {}
    for sym, wt in rows:
        per_el[sym] = per_el.get(sym, 0.0) + wt
    s = sum(per_el.values())
    if s > 0:
        per_el = {k: 100.0 * v / s for k, v in per_el.items()}
    return per_el, float(sum_net)

# ---------- Pairing & reporting (quiet unless real issues) ----------
def _list_ext(root: Path, ext: str) -> List[Path]:
    ext = ext.lower()
    return sorted(
        p.resolve()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() == ext
    )

def _index_by_scoped_coord(paths: List[Path], base: Path, scope_level: int) -> Dict[Tuple[Tuple[str,...], int, int, Optional[int]], List[Path]]:
    d: Dict[Tuple[Tuple[str,...], int, int, Optional[int]], List[Path]] = {}
    for p in paths:
        key = _scoped_key(p, base, scope_level)
        if key is not None:
            d.setdefault(key, []).append(p)
    return d

def _print_duplicates(base: Path,
                      dup_spx,
                      dup_bcf) -> None:
    if not dup_spx and not dup_bcf:
        return
    print("[pairing] Duplicate files for some scoped coordinate keys:")
    for k, files in sorted(dup_spx.items(), key=lambda kv: _scoped_sort_key(kv[0])):
        rel = " | ".join(str(_rel_to(p, base)) for p in sorted(files))
        print(f"  SPX {_scoped_key_display(k):>20} → {rel}")
    for k, files in sorted(dup_bcf.items(), key=lambda kv: _scoped_sort_key(kv[0])):
        rel = " | ".join(str(_rel_to(p, base)) for p in sorted(files))
        print(f"  BCF {_scoped_key_display(k):>20} → {rel}")

def _print_unpaired(base: Path,
                    only_spx,
                    only_bcf,
                    spx_idx,
                    bcf_idx) -> None:
    if only_spx:
        print("[pairing] Unpaired SPX (no matching BCF in the same scope):")
        for k in sorted(only_spx, key=_scoped_sort_key):
            files = [str(_rel_to(p, base)) for p in sorted(spx_idx.get(k, []))]
            print(f"  {_scoped_key_display(k):>20} → " + " | ".join(files))
    if only_bcf:
        print("[pairing] Unpaired BCF (no matching SPX in the same scope):")
        for k in sorted(only_bcf, key=_scoped_sort_key):
            files = [str(_rel_to(p, base)) for p in sorted(bcf_idx.get(k, []))]
            print(f"  {_scoped_key_display(k):>20} → " + " | ".join(files))

def _decide_coords(base: Path, on_mismatch: str, scope_level: int):
    """
    Return (coords_to_process, spx_by_coord, bcf_by_coord).
    Keys include the scope prefix (first N folders under --dir).
    Prints ONLY when there are duplicates or true mismatches (within-scope).
    """
    spx = _list_ext(base, ".spx")
    bcf = _list_ext(base, ".bcf")
    if not spx and not bcf:
        raise SystemExit(f"No .bcf or .spx files under {base}")

    # Index by scoped key
    spx_idx = _index_by_scoped_coord(spx, base, scope_level)
    bcf_idx = _index_by_scoped_coord(bcf, base, scope_level)

    # Report true duplicates (multiple files per scoped key)
    dup_spx = {k: v for k, v in spx_idx.items() if len(v) > 1}
    dup_bcf = {k: v for k, v in bcf_idx.items() if len(v) > 1}
    _print_duplicates(base, dup_spx, dup_bcf)

    # Deduplicate by taking the first sorted path per scoped key
    spx_by = {k: sorted(v)[0] for k, v in spx_idx.items()}
    bcf_by = {k: sorted(v)[0] for k, v in bcf_idx.items()}

    # Compare unique scoped key sets
    spx_keys = set(spx_by.keys())
    bcf_keys = set(bcf_by.keys())

    # If unique scoped key sets match → proceed silently (duplicates already reported)
    if spx_keys == bcf_keys and spx_keys:
        coords = sorted(spx_keys, key=_scoped_sort_key)
        return coords, spx_by, bcf_by

    # Otherwise, show ONLY the truly unpaired scoped keys
    only_spx = sorted(spx_keys - bcf_keys, key=_scoped_sort_key)
    only_bcf = sorted(bcf_keys - spx_keys, key=_scoped_sort_key)
    _print_unpaired(base, only_spx, only_bcf, spx_idx, bcf_idx)

    if on_mismatch == "fail":
        print("[pairing] Mismatch policy is 'fail'. Re-run with one of:")
        print("  --on-mismatch use_spx")
        print("  --on-mismatch use_bcf")
        print("  --on-mismatch intersect")
        raise SystemExit(2)

    if on_mismatch == "use_spx":
        coords = sorted(spx_keys & bcf_keys, key=_scoped_sort_key)
        return coords, spx_by, bcf_by

    if on_mismatch == "use_bcf":
        coords = sorted(bcf_keys, key=_scoped_sort_key)
        return coords, spx_by, bcf_by

    if on_mismatch == "intersect":
        coords = sorted(spx_keys & bcf_keys, key=_scoped_sort_key)
        if not coords:
            raise SystemExit("[pairing] No overlapping scoped keys between SPX and BCF.")
        return coords, spx_by, bcf_by

    raise SystemExit(f"[pairing] Unknown --on-mismatch option: {on_mismatch}")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Build per-tile composition cache with SPX quant (quiet & scoped pairing).")
    ap.add_argument("--dir", required=True, help="Folder with .bcf/.spx files (recursively searched)")
    ap.add_argument("--out", default="composition_cache.parquet", help="Output Parquet file")
    ap.add_argument("--on-mismatch", choices=["fail", "use_spx", "use_bcf", "intersect"], default="fail",
                    help="Behavior when SPX vs BCF keys differ (default: fail).")
    ap.add_argument("--scope-level", type=int, default=1,
                    help="Number of folder levels under --dir to include in the pairing scope (0=global, 1=top-level, 2=two levels, ...). Default: 1.")
    args = ap.parse_args()

    base = Path(args.dir).resolve()
    coords, spx_by, bcf_by = _decide_coords(base, args.on_mismatch, args.scope_level)

    rows: List[Dict[str, object]] = []
    stage_xy: List[Tuple[float, float]] = []

    for key in tqdm(coords, desc="Processing tiles", unit="tile"):
        bcf_path = bcf_by.get(key)
        if bcf_path is None:
            # Without BCF we cannot place on grid → skip
            continue

        spx_path = spx_by.get(key)
        sx, sy = _extract_stage_xy_from_bcf(bcf_path)

        if spx_path is not None:
            comp, weight = _spx_quant_for_file(spx_path)
            used_path = spx_path
        else:
            comp, weight = _bcf_relative_composition(bcf_path)
            used_path = bcf_path

        row = {
            "path": str(_rel_to(used_path, base)),
            "weight": float(weight),
            "__valid": 1 if (weight > 0 and comp) else 0,
            "stage_x": float(sx),
            "stage_y": float(sy),
        }
        if row["__valid"]:
            row.update(comp)
        rows.append(row)
        stage_xy.append((sx, sy))

    if not stage_xy:
        raise SystemExit("[error] No tiles processed; cannot build grid.")

    _, _, row_assign, col_assign, dy_stage, dx_stage = build_grid_from_stage_quantized(stage_xy)
    for i, r in enumerate(rows):
        r["row"] = int(row_assign[i])
        r["col"] = int(col_assign[i])

    df = pd.DataFrame(rows)
    fixed_cols = {"path","row","col","stage_x","stage_y","weight","__valid"}
    for c in [c for c in df.columns if c not in fixed_cols]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df = df.sort_values(["row","col","path"]).reset_index(drop=True)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outp, index=False)

    total_rows = len(df)
    valid_rows = int(df["__valid"].sum()) if "__valid" in df.columns else total_rows
    unique_cells = df[["row","col"]].drop_duplicates().shape[0]

    print(f"[cache] Wrote {total_rows} rows  → {outp}")
    print(f"[cache] Valid composition rows  : {valid_rows}")
    print(f"[cache] Unique (row,col) cells  : {unique_cells}")
    if (total_rows != unique_cells):
        print(f"[note] Multiple tiles can still share a grid cell if stage coords overlap.")
    print(f"[grid] stage dy ≈ {dy_stage:.6g} | dx ≈ {dx_stage:.6g}")

if __name__ == "__main__":
    main()
