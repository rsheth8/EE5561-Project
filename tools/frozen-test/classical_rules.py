"""Exact merge functions copied from reviewed hdr/classical.py, commit 9f83e01. No tuning."""
import numpy as np
GAMMA=2.2

def hat(z):
    return 1.0 - np.abs(2.0 * z - 1.0)

def radiance(l, t):
    return l.astype(np.float64) ** GAMMA / t

def merge(ldr: list, t: list, valid: list, cfg: dict, deghost: bool) -> tuple[np.ndarray, float]:
    """ldr: [L1, L2, L3] HWC in [0,1] (possibly warped); valid: [v1, 1, v3] HW masks.
    Returns (H, fallback_fraction)."""
    lo, hi, thr, eps = cfg["ref_lo"], cfg["ref_hi"], cfg["disagree_thr"], cfg["eps_weight"]
    H = [radiance(ldr[i], t[i]) for i in range(3)]
    W = [hat(ldr[i].astype(np.float64)) for i in range(3)]
    for i in (0, 2):
        W[i] = W[i] * valid[i][..., None]
    if deghost:
        ref_ok = (ldr[1] >= lo) & (ldr[1] <= hi)  # per channel
        for i in (0, 2):
            rel = np.abs(H[i] - H[1]) / np.maximum(np.maximum(H[i], H[1]), 1e-4)
            W[i] = np.where(ref_ok & (rel > thr), 0.0, W[i])
    num = sum(W[i] * H[i] for i in range(3))
    den = sum(W)
    out = np.where(den >= eps, num / np.maximum(den, 1e-12), 0.0)
    fb = den < eps  # per channel
    m2 = ldr[1].max(-1)
    use_short = (m2 >= 0.5)[..., None]
    cand = np.where(use_short, H[0], H[2])
    cand_valid = np.where(use_short[..., 0], valid[0] > 0, valid[2] > 0)[..., None]
    fallback = np.where(cand_valid, cand, H[1])
    out = np.where(fb, fallback, out)
    return out.astype(np.float32), float(fb.mean())
