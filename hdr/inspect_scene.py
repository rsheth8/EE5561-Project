"""python -m hdr.inspect_scene --scene raw/Training/<id> --out runs/data-check/<id> [--why "..."]

Saves montage.png (L_1, L_2, L_3 with e and t, display(H_gt)) + stats.json of raw arrays.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .display import DISPLAY_MAPPING, LDR_MAPPING, ldr_to_uint8, to_uint8, write_sidecar  # noqa: E402
from .io import load_scene_raw  # noqa: E402
from .runlog import make_run_dir, write_notes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--why", default="")
    args = ap.parse_args()
    s = load_scene_raw(args.scene)
    rd = make_run_dir(args.out, {"scene": args.scene, "why": args.why})
    ldr = s["ldr_raw"].astype(np.float32) / np.float32(np.iinfo(s["ldr_raw"].dtype).max)
    stats = {"scene_id": s["scene_id"], "why_selected": args.why, "tif_names": s["tif_names"], "dtype": s["dtype"],
             "e": s["e"].tolist(), "t": s["t"].tolist(), "pairing_means": s["pairing_means"].tolist(),
             "pairing_ok": s["pairing_ok"], "frames": []}
    for k in range(3):
        r = s["ldr_raw"][k]
        m = ldr[k].max(-1)
        stats["frames"].append({"file": s["tif_names"][k], "shape": list(r.shape), "raw_min": int(r.min()),
                                "raw_max": int(r.max()), "raw_mean": float(r.mean()),
                                "pct_m_gt_095": float(100 * (m > 0.95).mean()),
                                "pct_m_lt_005": float(100 * (m < 0.05).mean())})
    h = s["hdr"]
    stats["hdr"] = {"shape": list(h.shape), "min": float(h.min()), "max": float(h.max()), "mean": float(h.mean()),
                    "p50": float(np.percentile(h, 50)), "p99": float(np.percentile(h, 99)),
                    "p999": float(np.percentile(h, 99.9)), "finite": bool(np.isfinite(h).all())}
    (rd / "stats.json").write_text(json.dumps(stats, indent=2))
    (rd / "metrics.json").write_text(json.dumps(stats, indent=2))
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
    for k in range(3):
        ax[k].imshow(ldr_to_uint8(ldr[k]))
        ax[k].set_title(f"L_{k+1}  {s['tif_names'][k]}  e={s['e'][k]:g}  t={s['t'][k]:g}", fontsize=9)
    ax[3].imshow(to_uint8(h))
    ax[3].set_title("display(H_gt)", fontsize=9)
    for a in ax:
        a.axis("off")
    fig.suptitle(f"scene {s['scene_id']}", fontsize=10)
    fig.tight_layout()
    fp = rd / "figures" / "montage.png"
    fig.savefig(fp, dpi=100)
    plt.close(fig)
    write_sidecar(fp, [f"Scene {s['scene_id']}; full frames, no crop; data inspection, no method.",
                       f"LDR: {LDR_MAPPING}. H_gt: {DISPLAY_MAPPING}.", f"Selected because: {args.why}"])
    write_notes(rd, [f"scene {s['scene_id']} files {s['tif_names']} dtype {s['dtype']}",
                     f"e {s['e'].tolist()} t {s['t'].tolist()} pairing_ok {s['pairing_ok']}",
                     f"H_gt min {stats['hdr']['min']:.4g} max {stats['hdr']['max']:.4g} finite {stats['hdr']['finite']}",
                     f"selected because: {args.why}"])
    print(rd)


if __name__ == "__main__":
    main()
