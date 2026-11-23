"""Exp 2: dataset inventory, pairing check, reference-index/scale diagnostic, cache.

python scripts/exp2_dataset.py [--workers 16]
Writes results/tables/dataset_inventory.csv, results/tables/ref_scale_diag.csv,
cache/<scene>.npz, and runs/exp2/<ts>/{metrics.json, notes.md, ...}.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hdr import GAMMA  # noqa: E402
from hdr.data import CACHE, RAW, build_cache  # noqa: E402
from hdr.io import list_scenes, load_scene_raw  # noqa: E402
from hdr.runlog import make_run_dir, write_notes  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FIT_LO, FIT_HI = 0.1, 0.9


def ref_fit(ldr_k: np.ndarray, t_k: float, gt: np.ndarray) -> dict:
    ok = np.all((ldr_k >= FIT_LO) & (ldr_k <= FIT_HI), axis=-1)
    x = (ldr_k[ok].astype(np.float64) ** GAMMA) / t_k  # [n, 3]
    y = gt[ok].astype(np.float64)
    x, y = x.ravel(), y.ravel()
    s = float((x * y).sum() / (x * x).sum())
    pos = y > 1e-8
    rel = np.abs(s * x[pos] - y[pos]) / y[pos]
    r = float(np.corrcoef(x, y)[0, 1])
    return dict(s=s, med_rel_err=float(np.median(rel)), pearson=r, npix=int(ok.sum()))


def process(scene_dir: Path) -> dict:
    row = {"scene": scene_dir.name}
    try:
        s = load_scene_raw(scene_dir)
    except Exception as exc:  # noqa: BLE001
        row["error"] = repr(exc)
        return row
    ldr = s["ldr_raw"].astype(np.float32) / np.float32(np.iinfo(s["ldr_raw"].dtype).max)
    hdr = s["hdr"]
    m2 = ldr[1].max(axis=-1)
    row.update(
        tif_files=";".join(s["tif_names"]),
        dtype=s["dtype"],
        shape="x".join(map(str, s["ldr_raw"].shape[1:])),
        raw_min=int(s["ldr_raw"].min()),
        raw_max=int(s["ldr_raw"].max()),
        e=";".join(f"{v:g}" for v in s["e"]),
        t=";".join(f"{v:g}" for v in s["t"]),
        pairing_means=";".join(f"{v:.5f}" for v in s["pairing_means"]),
        pairing_npix=s["pairing_npix"],
        pairing_ok=s["pairing_ok"],
        hdr_shape="x".join(map(str, hdr.shape)),
        hdr_min=float(hdr.min()),
        hdr_max=float(hdr.max()),
        hdr_p50=float(np.percentile(hdr, 50)),
        hdr_p99=float(np.percentile(hdr, 99)),
        hdr_p999=float(np.percentile(hdr, 99.9)),
        hdr_finite=bool(np.all(np.isfinite(hdr))),
        pct_m2_gt_095=float(100 * (m2 > 0.95).mean()),
        pct_m2_lt_005=float(100 * (m2 < 0.05).mean()),
    )
    for k in range(3):
        f = ref_fit(ldr[k], float(s["t"][k]), hdr)
        for kk, v in f.items():
            row[f"k{k+1}_{kk}"] = v
    errs = [row[f"k{k}_med_rel_err"] for k in (1, 2, 3)]
    row["best_k"] = int(np.argmin(errs)) + 1
    if s["pairing_ok"]:
        build_cache(scene_dir, CACHE)
        row["cached"] = True
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    scenes = list_scenes(RAW)
    rd = make_run_dir(REPO / "runs" / "exp2", {"exp": "exp2", "raw": str(RAW), "n_scenes_found": len(scenes),
                                                "fit_band": [FIT_LO, FIT_HI], "gamma": GAMMA})
    with Pool(args.workers) as p:
        rows = p.map(process, scenes)
    df = pd.DataFrame(rows).sort_values("scene")
    anomalies = []
    for _, r in df.iterrows():
        a = []
        if isinstance(r.get("error"), str):
            a.append("read_error")
        else:
            if not r["pairing_ok"]:
                a.append("pairing_failed")
            if r["shape"] != df["shape"].mode()[0]:
                a.append(f"shape_{r['shape']}")
            if r["dtype"] != df["dtype"].mode()[0]:
                a.append(f"dtype_{r['dtype']}")
            if not r["hdr_finite"]:
                a.append("hdr_nonfinite")
            if r["best_k"] != 2:
                a.append(f"best_k_{r['best_k']}")
            if r["k2_npix"] < 1000:
                a.append("few_fit_pixels")
        anomalies.append(";".join(a))
    df["anomalies"] = anomalies
    inv_cols = [c for c in df.columns if not c.startswith("k") and c != "best_k"]
    tab = REPO / "results" / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    df[inv_cols].to_csv(tab / "dataset_inventory.csv", index=False)
    diag_cols = ["scene"] + [c for c in df.columns if c.startswith("k")] + ["best_k"]
    df[diag_cols].to_csv(tab / "ref_scale_diag.csv", index=False)
    df.to_csv(rd / "metrics.csv", index=False)

    ok = df[~df["anomalies"].str.contains("read_error|pairing_failed")]
    s2 = ok["k2_s"].to_numpy()
    summary = dict(
        n_scenes=int(len(df)),
        n_pairing_ok=int(df["pairing_ok"].fillna(False).sum()),
        dtypes=df["dtype"].value_counts().to_dict(),
        shapes=df["shape"].value_counts().to_dict(),
        exposure_sets=df["e"].value_counts().to_dict(),
        best_k_counts=df["best_k"].value_counts().to_dict(),
        mean_med_rel_err={f"k{k}": float(ok[f"k{k}_med_rel_err"].mean()) for k in (1, 2, 3)},
        median_med_rel_err={f"k{k}": float(ok[f"k{k}_med_rel_err"].median()) for k in (1, 2, 3)},
        mean_pearson={f"k{k}": float(ok[f"k{k}_pearson"].mean()) for k in (1, 2, 3)},
        s2_stats=dict(min=float(s2.min()), p25=float(np.percentile(s2, 25)), median=float(np.median(s2)),
                      p75=float(np.percentile(s2, 75)), max=float(s2.max()), mean=float(s2.mean())),
        scenes_with_anomalies={r["scene"]: r["anomalies"] for _, r in df.iterrows() if r["anomalies"]},
        max_saturated_scene=str(df.loc[df["pct_m2_gt_095"].idxmax(), "scene"]),
        max_saturated_pct=float(df["pct_m2_gt_095"].max()),
    )
    (rd / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    write_notes(rd, [
        f"{summary['n_scenes']} scenes found; pairing ok for {summary['n_pairing_ok']}",
        f"dtypes {summary['dtypes']}; shapes {summary['shapes']}",
        f"best-fitting reference index counts {summary['best_k_counts']}",
        f"median of per-scene median rel. err: {summary['median_med_rel_err']}",
        f"s_2 median {summary['s2_stats']['median']:.4f} (range {summary['s2_stats']['min']:.4f}-{summary['s2_stats']['max']:.4f}); nothing rescaled",
        f"anomalies: {summary['scenes_with_anomalies'] or 'none'}",
        "tables: results/tables/dataset_inventory.csv, results/tables/ref_scale_diag.csv; cache written to cache/",
    ])
    print("run dir", rd)


if __name__ == "__main__":
    main()
