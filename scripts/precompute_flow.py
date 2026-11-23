"""Precompute alignment for all scenes (CPU, parallel) -> cache/flow/<scene>.npz.

Saved per scene (index 0 -> frame 1 (short), index 1 -> frame 3 (long); all in reference coords):
  F_f, F_b [2,H,W,2]; Lw [2,H,W,3]; v [2,H,W] uint8; r [2,H,W]; incons [2,H,W] uint8;
  e [2,H,W] (FB squared error); q_w [2,H,W]; q_ref [H,W].
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hdr.alignment import align_scene  # noqa: E402
from hdr.data import CACHE, FLOW_CACHE, load_scene_np  # noqa: E402
from hdr.runlog import make_run_dir, write_notes  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def work(sid: str) -> dict:
    cv2.setNumThreads(1)
    t0 = time.time()
    s = load_scene_np(sid)
    a = align_scene(s["ldr_hwc"], s["t"])
    t_align = time.time() - t0
    a.pop("e", None)
    np.savez(FLOW_CACHE / f"{sid}.npz", **a)
    return {"scene": sid, "align_s": t_align, "total_s": time.time() - t0,
            "valid_frac_1": float(a["v"][0].mean()), "valid_frac_3": float(a["v"][1].mean()),
            "incons_frac_1": float(a["incons"][0].mean()), "incons_frac_3": float(a["incons"][1].mean()),
            "median_flow_mag_1": float(np.median(np.linalg.norm(a["F_f"][0], axis=-1))),
            "median_flow_mag_3": float(np.median(np.linalg.norm(a["F_f"][1], axis=-1)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    FLOW_CACHE.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.stem for p in CACHE.glob("*.npz"))
    rd = make_run_dir(REPO / "runs" / "flow", {"n_scenes": len(ids), "workers": args.workers,
                                                "method": "DIS PRESET_MEDIUM, fb alpha 0.01 beta 0.5"})
    t0 = time.time()
    with Pool(args.workers) as p:
        rows = p.map(work, ids)
    df = pd.DataFrame(rows)
    df.to_csv(rd / "metrics.csv", index=False)
    summ = {"n": len(df), "wall_s": time.time() - t0, "mean_align_s": float(df.align_s.mean()),
            "max_align_s": float(df.align_s.max())}
    (rd / "metrics.json").write_text(json.dumps(summ, indent=2))
    write_notes(rd, [f"{len(df)} scenes aligned at full res; wall {summ['wall_s']:.0f} s with {args.workers} workers",
                     f"per-scene align time mean {summ['mean_align_s']:.2f} s, max {summ['max_align_s']:.2f} s (single thread)",
                     "per-scene runtime and valid/inconsistent fractions in metrics.csv",
                     "outputs: cache/flow/<scene>.npz"])
    print(summ)


if __name__ == "__main__":
    main()
