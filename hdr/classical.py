"""Classical merges C0-C3 (EXPERIMENT_PLAN.md 5.2).

python -m hdr.classical --config configs/classical.json --out runs/classical-v1

Hat weight w(z) = 1 - |2z - 1| per channel. Our own specified weighted merge, NOT a full
Debevec reproduction (no response calibration).
  C0 reference only: H = H_2
  C1 exposure-weighted merge, no alignment
  C2 reference-fallback deghosting, no alignment
  C3 C2 on flow-aligned L_i^w / H_i^w, non-reference weights multiplied by v_i
No usable source (sum w < eps): m(L_2) >= 0.5 -> shortest exposure (H_1 / H_1^w if valid),
else longest (H_3 / H_3^w); if that sample is invalid -> H_2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import GAMMA
from .data import load_flow_np, load_scene_np, load_split
from .io import write_hdr
from .metrics import METRIC_KEYS, all_metrics
from .runlog import big_dir, make_run_dir, write_notes

REPO = Path(__file__).resolve().parent.parent


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


def run_methods(sid: str, cfg: dict) -> dict:
    s = load_scene_np(sid)
    L, t = s["ldr_hwc"], [float(x) for x in s["t"]]
    ones = np.ones(L.shape[1:3], np.float32)
    out = {"C0": (radiance(L[1], t[1]).astype(np.float32), 0.0)}
    out["C1"] = merge([L[0], L[1], L[2]], t, [ones, ones, ones], cfg, deghost=False)
    out["C2"] = merge([L[0], L[1], L[2]], t, [ones, ones, ones], cfg, deghost=True)
    f = load_flow_np(sid)
    v = f["v"].astype(np.float32)
    out["C3"] = merge([f["Lw"][0], L[1], f["Lw"][1]], t, [v[0], ones, v[1]], cfg, deghost=True)
    return {"target": s["target_hwc"], "ldr": L, "t": t, "methods": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    split = load_split(REPO / cfg.get("split", "configs/split.json"))
    scenes = split[cfg.get("scenes", "val")]
    rd = make_run_dir(args.out, cfg)
    od = big_dir(rd, "outputs")
    rows = []
    for sid in scenes:
        r = run_methods(sid, cfg)
        for name, (h, fbf) in r["methods"].items():
            m = all_metrics(h, r["target"])
            rows.append({"scene": sid, "method": name, **m, "fallback_pct": 100 * fbf})
            np.save(od / f"{sid}_{name}.npy", h)
            write_hdr(od / f"{sid}_{name}.hdr", h)
        print(sid, {x["method"]: round(x["psnr_mu"], 3) for x in rows if x["scene"] == sid}, flush=True)
    df = pd.DataFrame(rows)
    mean = df.groupby("method")[METRIC_KEYS + ["fallback_pct"]].mean().reset_index()
    mean.insert(0, "scene", "mean")
    full = pd.concat([df, mean], ignore_index=True)
    full.to_csv(rd / "metrics.csv", index=False)
    tab = REPO / "results" / "tables" / "classical_val.csv"
    with tab.open("w") as fh:
        fh.write(f"# {len(scenes)} validation scenes ({', '.join(scenes)}), full image, no border excluded, mu = 5000; "
                 f"PSNR-L/SSIM-L on clip(H,0,1), PSNR-mu/SSIM-mu on mu-law; source run {rd.relative_to(REPO)}\n")
        full.to_csv(fh, index=False)
    (rd / "metrics.json").write_text(json.dumps({r["method"]: {k: r[k] for k in METRIC_KEYS + ["fallback_pct"]}
                                                 for _, r in mean.iterrows()}, indent=2))
    write_notes(rd, [
        f"C0-C3 on {len(scenes)} val scenes {scenes}; constants {json.dumps({k: cfg[k] for k in ('ref_lo','ref_hi','disagree_thr','eps_weight')})}",
        *[f"{r['method']}: PSNR-mu {r['psnr_mu']:.3f}, PSNR-L {r['psnr_l']:.3f}, SSIM-mu {r['ssim_mu']:.4f}, fallback {r['fallback_pct']:.3f}% (mean)"
          for _, r in mean.iterrows()],
        "outputs/<scene>_<method>.npy|.hdr hold the full-res predictions",
    ])
    print(mean.to_string())


if __name__ == "__main__":
    main()
