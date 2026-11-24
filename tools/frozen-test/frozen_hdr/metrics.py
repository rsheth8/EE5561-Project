"""Metrics (EXPERIMENT_PLAN.md section 2.4). Inputs are HWC float arrays (linear radiance).

mu-law: T(H) = log(1 + mu*H) / log(1 + mu) applied to H clipped to [0, 1], mu = 5000.
PSNR-L on clip(H,0,1), PSNR-mu on T(H), peak 1. SSIM via skimage (channel_axis=-1,
data_range=1). Full image, no border excluded. Inputs are never modified.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from skimage.metrics import structural_similarity

MU = 5000.0
_LOG1P_MU = math.log(1.0 + MU)


def mu_law(h: np.ndarray) -> np.ndarray:
    return (np.log1p(MU * np.clip(h, 0.0, 1.0).astype(np.float64)) / _LOG1P_MU)


def mu_law_torch(h: torch.Tensor) -> torch.Tensor:
    return torch.log1p(MU * h.clamp(0.0, 1.0)) / _LOG1P_MU


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def psnr_l(pred: np.ndarray, gt: np.ndarray) -> float:
    return psnr(np.clip(pred, 0, 1), np.clip(gt, 0, 1))


def psnr_mu(pred: np.ndarray, gt: np.ndarray) -> float:
    return psnr(mu_law(pred), mu_law(gt))


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(structural_similarity(a, b, channel_axis=-1, data_range=1.0))


def ssim_l(pred: np.ndarray, gt: np.ndarray) -> float:
    return _ssim(np.clip(pred, 0, 1).astype(np.float64), np.clip(gt, 0, 1).astype(np.float64))


def ssim_mu(pred: np.ndarray, gt: np.ndarray) -> float:
    return _ssim(mu_law(pred), mu_law(gt))


def all_metrics(pred: np.ndarray, gt: np.ndarray, ssim: bool = True) -> dict:
    assert pred.shape == gt.shape and pred.ndim == 3 and pred.shape[-1] == 3, (pred.shape, gt.shape)
    out = {"psnr_l": psnr_l(pred, gt), "psnr_mu": psnr_mu(pred, gt)}
    if ssim:
        out["ssim_l"] = ssim_l(pred, gt)
        out["ssim_mu"] = ssim_mu(pred, gt)
    return out


METRIC_KEYS = ["psnr_l", "psnr_mu", "ssim_l", "ssim_mu"]
