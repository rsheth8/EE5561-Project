"""Alignment module (EXPERIMENT_PLAN.md 5.1): exposure matching, DIS flow, backward warp,
validity, forward-backward check, exposure cues.

Flow convention (verified in tests/test_warp.py): F_f = dis.calc(ref_gray, src_gray)
gives, for reference pixel x, where to sample the source: ref(x) ~= src(x + F_f(x)).
"""
from __future__ import annotations

import cv2
import numpy as np

from . import GAMMA

FB_ALPHA = 0.01  # Sundaram et al. 2010
FB_BETA = 0.5


def exposure_match(l_dark: np.ndarray, t_dark: float, t_bright: float) -> np.ndarray:
    """L_dark->bright = clip(L_dark**gamma * t_bright/t_dark, 0, 1)**(1/gamma)."""
    return np.clip(l_dark.astype(np.float64) ** GAMMA * (t_bright / t_dark), 0, 1) ** (1 / GAMMA)


def to_gray8(l: np.ndarray) -> np.ndarray:
    u8 = np.round(255 * np.clip(l, 0, 1)).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)


def dis_flow(a_gray: np.ndarray, b_gray: np.ndarray) -> np.ndarray:
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return dis.calc(a_gray, b_gray, None)


def sample_map(flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return gx + flow[..., 0], gy + flow[..., 1]


def backward_warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    mx, my = sample_map(flow)
    return cv2.remap(img.astype(np.float32), mx, my, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def validity(flow: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    mx, my = sample_map(flow)
    return ((mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)).astype(np.uint8)


def fb_check(f_f: np.ndarray, f_b: np.ndarray) -> dict:
    """e(x) = |F_f(x) + F_b(x + F_f(x))|^2 with F_b sampled bilinearly.
    Inconsistent if e > 0.01*(|F_f|^2 + |F_b(x+F_f)|^2) + 0.5. Soft cue r = exp(-e/2)."""
    fb_w = backward_warp(f_b, f_f)  # F_b sampled at x + F_f(x)
    s = f_f + fb_w
    e = (s ** 2).sum(-1)
    thr = FB_ALPHA * ((f_f ** 2).sum(-1) + (fb_w ** 2).sum(-1)) + FB_BETA
    return {"e": e.astype(np.float32), "incons": (e > thr).astype(np.uint8), "r": np.exp(-e / 2).astype(np.float32)}


def m_of(l: np.ndarray) -> np.ndarray:
    return l.max(axis=-1)


def q_cue(l: np.ndarray) -> np.ndarray:
    return (1 - np.abs(2 * m_of(l) - 1)).astype(np.float32)


def align_pair(l_ref: np.ndarray, t_ref: float, l_src: np.ndarray, t_src: float) -> dict:
    """Align source frame to the reference. Images HWC float in [0,1]."""
    if t_src < t_ref:  # source darker: raise it to the reference exposure
        ref_m, src_m = l_ref, exposure_match(l_src, t_src, t_ref)
    else:  # reference darker: raise the reference to the source exposure
        ref_m, src_m = exposure_match(l_ref, t_ref, t_src), l_src
    rg, sg = to_gray8(ref_m), to_gray8(src_m)
    f_f = dis_flow(rg, sg)
    f_b = dis_flow(sg, rg)
    lw = np.clip(backward_warp(l_src, f_f), 0, 1)  # warp the ORIGINAL L_i
    v = validity(f_f)
    lw = lw * v[..., None]
    fb = fb_check(f_f, f_b)
    qw = (v * q_cue(lw)).astype(np.float32)
    return dict(F_f=f_f, F_b=f_b, Lw=lw.astype(np.float32), v=v, r=fb["r"], e=fb["e"], incons=fb["incons"],
                q_w=qw, ref_matched=ref_m.astype(np.float32), src_matched=src_m.astype(np.float32))


def align_scene(ldr: np.ndarray, t: np.ndarray) -> dict:
    """ldr [3, H, W, 3], t [3]. Returns stacked results for i = 1, 3 (index 0, 1)."""
    out = {k: [] for k in ("F_f", "F_b", "Lw", "v", "r", "incons", "q_w")}
    for i in (0, 2):
        a = align_pair(ldr[1], float(t[1]), ldr[i], float(t[i]))
        for k in out:
            out[k].append(a[k])
    res = {k: np.stack(v) for k, v in out.items()}
    res["q_ref"] = q_cue(ldr[1])
    return res
