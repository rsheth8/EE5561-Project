"""Display tone mapping for figures only. Never used by metrics.

to_uint8(H) = round(255 * T(clip(H, 0, 1))), T = mu-law with mu = 5000.
Works on copies; the input array is never modified.
"""
from __future__ import annotations

import numpy as np

from .metrics import mu_law

DISPLAY_MAPPING = "to_uint8(H) = round(255*T(clip(H,0,1))), T = mu-law, mu=5000"
LDR_MAPPING = "LDR shown as stored (normalised by dtype max, x255, rounded)"


def to_uint8(h: np.ndarray) -> np.ndarray:
    return np.round(255.0 * mu_law(np.array(h, copy=True))).astype(np.uint8)


def ldr_to_uint8(l: np.ndarray) -> np.ndarray:
    return np.round(255.0 * np.clip(np.array(l, copy=True), 0, 1)).astype(np.uint8)


def write_sidecar(fig_path, lines: list[str]) -> None:
    from pathlib import Path

    p = Path(fig_path).with_suffix(".txt")
    p.write_text("\n".join(lines) + "\n")
