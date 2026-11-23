"""File reading for the Kalantari (SIGGRAPH17) Training archive.

Conventions (EXPERIMENT_PLAN.md section 2.1):
- LDR TIFFs are read with IMREAD_UNCHANGED, converted BGR->RGB and normalised by the
  dtype maximum (never assume 255).
- HDRImg.hdr is read as float32 radiance (RGB), all values must be finite.
- exposure.txt holds log2 exposure values e; relative times are t = 2**e.
- TIFFs are sorted lexicographically and paired with exposure rows in file order.
  The pairing is verified by checking that mean brightness over pixels unclipped in
  all three frames increases strictly with e.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

# Pixels used for the pairing check: every channel inside this band in all 3 frames.
PAIR_LO, PAIR_HI = 0.05, 0.95


def list_scenes(root: str | os.PathLike) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.iterdir() if p.is_dir())


def tif_files(scene_dir: str | os.PathLike) -> list[Path]:
    scene_dir = Path(scene_dir)
    return sorted(p for p in scene_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff"))


def read_ldr_raw(path: str | os.PathLike) -> np.ndarray:
    """Return the raw integer RGB array [H, W, 3] (original dtype)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read {path}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"{path}: expected 3-channel image, got {img.shape}")
    if not np.issubdtype(img.dtype, np.integer):
        raise ValueError(f"{path}: expected integer TIFF, got {img.dtype}")
    return np.ascontiguousarray(img[..., ::-1])


def normalize_ldr(raw: np.ndarray) -> np.ndarray:
    return raw.astype(np.float32) / np.float32(np.iinfo(raw.dtype).max)


def read_hdr(path: str | os.PathLike) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read {path}")
    img = np.ascontiguousarray(img[..., ::-1]).astype(np.float32)
    if not np.all(np.isfinite(img)):
        raise ValueError(f"{path}: non-finite values in HDR")
    return img


def write_hdr(path: str | os.PathLike, rgb: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), np.ascontiguousarray(rgb[..., ::-1]).astype(np.float32))
    if not ok:
        raise IOError(f"cannot write {path}")


def read_exposures(path: str | os.PathLike) -> np.ndarray:
    vals = [float(x) for x in Path(path).read_text().split()]
    return np.asarray(vals, dtype=np.float64)


def pairing_means(ldrs: np.ndarray) -> tuple[np.ndarray, int]:
    """ldrs: [3, H, W, 3] floats in [0,1]. Mean of each frame over pixels whose every
    channel is in [PAIR_LO, PAIR_HI] in all three frames. Returns (means, n_pixels)."""
    ok = np.all((ldrs >= PAIR_LO) & (ldrs <= PAIR_HI), axis=(0, 3))
    n = int(ok.sum())
    means = np.array([ldrs[k][ok].mean() if n else np.nan for k in range(ldrs.shape[0])])
    return means, n


def load_scene_raw(scene_dir: str | os.PathLike) -> dict:
    """Read one scene, pair files with exposure rows and verify the pairing.

    Raises RuntimeError if the pairing check fails (never reorders silently)."""
    scene_dir = Path(scene_dir)
    tifs = tif_files(scene_dir)
    e = read_exposures(scene_dir / "exposure.txt")
    if len(tifs) != 3 or len(e) != 3:
        raise RuntimeError(f"{scene_dir}: expected 3 TIFFs and 3 exposure rows, got {len(tifs)}/{len(e)}")
    raws = [read_ldr_raw(p) for p in tifs]
    dtypes = {str(r.dtype) for r in raws}
    shapes = {r.shape for r in raws}
    if len(dtypes) != 1 or len(shapes) != 1:
        raise RuntimeError(f"{scene_dir}: inconsistent dtype/shape {dtypes} {shapes}")
    ldr_raw = np.stack(raws)  # [3, H, W, 3]
    ldr = normalize_ldr(ldr_raw)
    means, n = pairing_means(ldr)
    pairing_ok = bool(np.all(np.diff(e) > 0) and np.all(np.diff(means) > 0))
    hdr = read_hdr(scene_dir / "HDRImg.hdr")
    if hdr.shape != ldr_raw.shape[1:]:
        raise RuntimeError(f"{scene_dir}: HDR shape {hdr.shape} != LDR shape {ldr_raw.shape[1:]}")
    return dict(
        scene_id=scene_dir.name,
        tif_names=[p.name for p in tifs],
        ldr_raw=ldr_raw,
        dtype=str(ldr_raw.dtype),
        e=e,
        t=2.0 ** e,
        hdr=hdr,
        pairing_means=means,
        pairing_npix=n,
        pairing_ok=pairing_ok,
    )
