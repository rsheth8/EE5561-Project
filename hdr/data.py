"""Per-scene data contract, cache and crop/augment sampler (EXPERIMENT_PLAN.md 2.3, 6.4).

Per-scene contract (float32 torch tensors):
    ldr      [3, 3, H, W]   exposures x RGB, normalised by dtype max
    exposure [3]            relative times t_i = 2**e_i
    radiance [3, 3, H, W]   H_i = L_i**gamma / t_i
    features [3, 6, H, W]   X_i = concat(L_i, H_i)
    target   [3, H, W]      H_gt
    scene_id str
Cache: cache/<scene_id>.npz holds the raw integer LDRs, target, e, t, file names.
Radiance/features are computed on load.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import GAMMA
from .io import load_scene_raw

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "cache"
RAW = REPO / "raw" / "Training"
FLOW_CACHE = CACHE / "flow"


# ----------------------------------------------------------------------------- cache
def build_cache(scene_dir, cache_dir=CACHE) -> Path:
    s = load_scene_raw(scene_dir)
    if not s["pairing_ok"]:
        raise RuntimeError(f"pairing check failed for {s['scene_id']}: means {s['pairing_means']} e {s['e']}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{s['scene_id']}.npz"
    np.savez(
        out,
        ldr_raw=s["ldr_raw"],  # [3, H, W, 3] original integer dtype, RGB
        target=s["hdr"],  # [H, W, 3] float32
        e=s["e"],
        t=s["t"],
        tif_names=np.array(s["tif_names"]),
        dtype_max=np.array(np.iinfo(s["ldr_raw"].dtype).max),
    )
    return out


def radiance_from_ldr(ldr: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """ldr [..., 3(exp), 3, H, W] or [3,3,H,W]; t [3] -> L**gamma / t."""
    shape = [1] * ldr.dim()
    shape[-4] = 3
    return ldr.clamp(0, 1).pow(GAMMA) / t.view(shape).to(ldr)


def features_from_ldr(ldr: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.cat([ldr, radiance_from_ldr(ldr, t)], dim=-3)


def load_scene_np(scene_id: str, cache_dir=CACHE) -> dict:
    z = np.load(Path(cache_dir) / f"{scene_id}.npz")
    raw = z["ldr_raw"]
    ldr = raw.astype(np.float32) / np.float32(z["dtype_max"])
    return dict(
        scene_id=scene_id,
        ldr_hwc=ldr,  # [3, H, W, 3]
        ldr_raw=raw,
        target_hwc=z["target"],
        e=z["e"],
        t=z["t"],
        tif_names=[str(x) for x in z["tif_names"]],
    )


def load_scene(scene_id: str, cache_dir=CACHE) -> dict:
    """The per-scene contract as float32 torch tensors."""
    s = load_scene_np(scene_id, cache_dir)
    ldr = torch.from_numpy(np.ascontiguousarray(s["ldr_hwc"].transpose(0, 3, 1, 2)))
    t = torch.from_numpy(s["t"].astype(np.float32))
    radiance = radiance_from_ldr(ldr, t)
    return dict(
        ldr=ldr,
        exposure=t,
        radiance=radiance,
        features=torch.cat([ldr, radiance], dim=1),
        target=torch.from_numpy(np.ascontiguousarray(s["target_hwc"].transpose(2, 0, 1))),
        scene_id=scene_id,
    )


def load_split(path=REPO / "configs" / "split.json") -> dict:
    return json.loads(Path(path).read_text())


# ----------------------------------------------------------------------------- flow cache
FLOW_KEYS_IMG = ("Lw",)  # [2, H, W, 3] warped LDR of frames 1 and 3 (index 0 -> L1, 1 -> L3)


def load_flow_np(scene_id: str, flow_dir=FLOW_CACHE) -> dict:
    z = np.load(Path(flow_dir) / f"{scene_id}.npz")
    return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------- scene arrays for training
class SceneArrays:
    """Channel-first arrays of one scene held in RAM, all in reference coordinates.

    ldr   uint16/uint8 [3, 3, H, W]  (raw, normalised on crop)
    tgt   float32 [3, H, W]
    t     float32 [3]
    and, if with_flow:
    lw    float32 [2, 3, H, W]   warped L1, L3 (0 outside valid region)
    cues  float32 [2, 4, H, W]   [q_ref, q_i^w, v_i, r_i] for i = 1, 3
    """

    def __init__(self, scene_id: str, with_flow: bool, cache_dir=CACHE, flow_dir=FLOW_CACHE):
        z = np.load(Path(cache_dir) / f"{scene_id}.npz")
        self.scene_id = scene_id
        self.ldr = np.ascontiguousarray(z["ldr_raw"].transpose(0, 3, 1, 2))
        self.dmax = np.float32(z["dtype_max"])
        self.tgt = np.ascontiguousarray(z["target"].transpose(2, 0, 1)).astype(np.float32)
        self.t = z["t"].astype(np.float32)
        self.H, self.W = self.tgt.shape[1:]
        self.with_flow = with_flow
        if with_flow:
            f = np.load(Path(flow_dir) / f"{scene_id}.npz")
            self.lw = np.ascontiguousarray(f["Lw"].transpose(0, 3, 1, 2)).astype(np.float32)
            q_ref = f["q_ref"].astype(np.float32)
            cues = []
            for k in range(2):
                cues.append(np.stack([q_ref, f["q_w"][k], f["v"][k].astype(np.float32), f["r"][k]]))
            self.cues = np.ascontiguousarray(np.stack(cues)).astype(np.float32)

    def crop(self, y: int, x: int, size: int) -> dict:
        sl = (slice(y, y + size), slice(x, x + size))
        d = {
            "ldr": self.ldr[..., sl[0], sl[1]].astype(np.float32) / self.dmax,
            "target": self.tgt[:, sl[0], sl[1]],
        }
        if self.with_flow:
            d["lw"] = self.lw[..., sl[0], sl[1]]
            d["cues"] = self.cues[..., sl[0], sl[1]]
        return d

    def full(self) -> dict:
        d = {"ldr": self.ldr.astype(np.float32) / self.dmax, "target": self.tgt}
        if self.with_flow:
            d["lw"] = self.lw
            d["cues"] = self.cues
        return d


def augment(d: dict, flip_h: bool, flip_v: bool, rot_k: int) -> dict:
    """Apply the same flip/rot90 to every array (last two axes). Valid because all
    arrays are in reference coordinates and no flow vectors are network inputs."""
    out = {}
    for k, a in d.items():
        if flip_h:
            a = a[..., :, ::-1]
        if flip_v:
            a = a[..., ::-1, :]
        if rot_k:
            a = np.rot90(a, rot_k, axes=(-2, -1))
        out[k] = np.ascontiguousarray(a)
    return out


def to_batch(items: list[dict], t_list: list[np.ndarray], device) -> dict:
    """Stack crops and build network inputs.

    feat   [B, 3, 6, h, w]  original X_i
    featw  [B, 3, 6, h, w]  X_1^w, X_2, X_3^w  (only if warped frames present)
    cues   [B, 2, 4, h, w]
    valid  [B, 2, 1, h, w]
    target [B, 3, h, w]
    """
    ldr = torch.from_numpy(np.stack([it["ldr"] for it in items])).to(device)
    t = torch.from_numpy(np.stack(t_list)).to(device)  # [B, 3]
    tv = t.view(t.shape[0], 3, 1, 1, 1)
    rad = ldr.clamp(0, 1).pow(GAMMA) / tv
    b = {"feat": torch.cat([ldr, rad], dim=2), "target": torch.from_numpy(np.stack([it["target"] for it in items])).to(device)}
    if "lw" in items[0]:
        lw = torch.from_numpy(np.stack([it["lw"] for it in items])).to(device)  # [B, 2, 3, h, w]
        ldr_w = torch.stack([lw[:, 0], ldr[:, 1], lw[:, 1]], dim=1)
        rad_w = ldr_w.clamp(0, 1).pow(GAMMA) / tv
        b["featw"] = torch.cat([ldr_w, rad_w], dim=2)
        cues = torch.from_numpy(np.stack([it["cues"] for it in items])).to(device)
        b["cues"] = cues
        b["valid"] = cues[:, :, 2:3]
    return b


class CropSampler:
    """Random 256x256 crops + flips/rot90 from scenes held in RAM (seeded)."""

    def __init__(self, scenes: list[SceneArrays], crop: int, augment_: bool, seed: int):
        self.scenes = scenes
        self.crop_size = crop
        self.augment = augment_
        self.rng = np.random.default_rng(seed)

    def sample(self, batch: int, device) -> dict:
        items, ts = [], []
        for _ in range(batch):
            s = self.scenes[self.rng.integers(len(self.scenes))]
            y = int(self.rng.integers(0, s.H - self.crop_size + 1))
            x = int(self.rng.integers(0, s.W - self.crop_size + 1))
            d = s.crop(y, x, self.crop_size)
            if self.augment:
                d = augment(d, bool(self.rng.integers(2)), bool(self.rng.integers(2)), int(self.rng.integers(4)))
            items.append(d)
            ts.append(s.t)
        return to_batch(items, ts, device)
