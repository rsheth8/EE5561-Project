"""python -m hdr.evaluate --config configs/pilot-eval.json --out runs/pilot-eval-v1

Loads a checkpoint selected from a training run's val_log.csv (best mean PSNR-mu among
eval steps <= max_step, or a fixed `step`), runs full-resolution inference on the val
scenes with cudnn.deterministic = True, and writes per-scene metrics, the outputs
(.npy float32 for exact comparison, .hdr, display .png) and a comparison against the
metrics logged during training.

Config:
{"scenes": "val",
 "entries": [{"name": "M0", "variant": "M0", "exp": "pilot_M0", "run_dir": "latest" | "<path>",
              "max_step": null | int, "step": null | int}],
 "save_outputs": true}
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from .data import SceneArrays, load_split
from .display import to_uint8
from .io import write_hdr
from .metrics import METRIC_KEYS, all_metrics
from .model import build_model
from .runlog import big_dir, make_run_dir, write_notes
from .torchutil import setup_torch
from .train import needs_flow, predict_full

REPO = Path(__file__).resolve().parent.parent


def resolve_run(entry: dict) -> Path:
    rd = entry.get("run_dir", "latest")
    if rd == "latest":
        base = REPO / "runs" / entry["exp"]
        cands = sorted(d for d in base.iterdir() if (d / "val_log.csv").exists())
        if not cands:
            raise FileNotFoundError(f"no run with val_log.csv under {base}")
        return cands[-1]
    return (REPO / rd).resolve()


def select_step(run_dir: Path, max_step=None, step=None) -> tuple[int, pd.DataFrame]:
    vl = pd.read_csv(run_dir / "val_log.csv", dtype={"scene": str})
    vl = vl.drop_duplicates(["step", "scene"], keep="last")
    if step is not None:
        return int(step), vl
    vm = vl[vl.scene == "mean"]
    if max_step is not None:
        vm = vm[vm.step <= max_step]
    if vm.empty:
        raise RuntimeError(f"no eval step <= {max_step} in {run_dir}")
    return int(vm.loc[vm.psnr_mu.idxmax(), "step"]), vl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    device = setup_torch(5561, deterministic=True)
    torch.backends.cudnn.deterministic = True
    split = load_split(REPO / cfg.get("split", "configs/split.json"))
    scenes = split[cfg.get("scenes", "val")]
    resolved = []
    for e in cfg["entries"]:
        rd = resolve_run(e)
        st, _ = select_step(rd, e.get("max_step"), e.get("step"))
        resolved.append({**e, "run_dir_resolved": str(rd), "selected_step": st})
    out = make_run_dir(args.out, {**cfg, "resolved": resolved})
    od = big_dir(out, "outputs")
    rows, cmp_rows, sel = [], [], {}
    for e in resolved:
        rd, st, var = Path(e["run_dir_resolved"]), e["selected_step"], e["variant"].upper()
        ck = torch.load(rd / "checkpoints" / f"step_{st:06d}.pt", map_location=device, weights_only=False)
        model = build_model(var).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        _, vl = select_step(rd, step=st)
        logged = vl[vl.step == st].set_index("scene")
        (od / e["name"]).mkdir(exist_ok=True)
        hashes = {}
        for sid in scenes:
            s = SceneArrays(sid, needs_flow(var))
            pred, gates = predict_full(model, var, s, device)
            m = all_metrics(pred, s.tgt.transpose(1, 2, 0))
            rows.append({"name": e["name"], "variant": var, "step": st, "scene": sid, **m})
            if cfg.get("save_outputs", True):
                np.save(od / e["name"] / f"{sid}.npy", pred)
                write_hdr(od / e["name"] / f"{sid}.hdr", pred)
                cv2.imwrite(str(od / e["name"] / f"{sid}.png"), to_uint8(pred)[..., ::-1])
                if gates is not None:
                    np.savez_compressed(od / e["name"] / f"{sid}_gates.npz", g1=gates[0], g3=gates[1])
            hashes[sid] = hashlib.sha256(np.ascontiguousarray(pred).tobytes()).hexdigest()
            if sid in logged.index:
                cmp_rows.append({"name": e["name"], "scene": sid, "step": st,
                                 **{f"d_{k}": m[k] - float(logged.loc[sid, k]) for k in METRIC_KEYS}})
            print(e["name"], st, sid, {k: round(v, 4) for k, v in m.items()}, flush=True)
        sel[e["name"]] = {"run_dir": str(Path(rd).relative_to(REPO)), "step": st, "max_step": e.get("max_step"),
                          "output_sha256": hashes}
    df = pd.DataFrame(rows)
    mean = df.groupby(["name", "variant", "step"])[METRIC_KEYS].mean().reset_index()
    mean["scene"] = "mean"
    full = pd.concat([df, mean], ignore_index=True)
    full.to_csv(out / "metrics.csv", index=False)
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(out / "vs_val_log.csv", index=False)
    maxdiff = {k: float(cmp[f"d_{k}"].abs().max()) for k in METRIC_KEYS} if len(cmp) else {}
    res = {"selection": sel, "mean": {r["name"]: {k: float(r[k]) for k in METRIC_KEYS} for _, r in mean.iterrows()},
           "max_abs_diff_vs_training_val_log": maxdiff, "cudnn_deterministic": True}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    write_notes(out, [
        f"evaluated {[e['name'] for e in resolved]} on {len(scenes)} val scenes {scenes}, full image, cudnn deterministic",
        *[f"{n}: run {v['run_dir']} step {v['step']} (max_step {v['max_step']})" for n, v in sel.items()],
        *[f"{r['name']}: PSNR-mu {r['psnr_mu']:.3f} PSNR-L {r['psnr_l']:.3f} SSIM-mu {r['ssim_mu']:.4f} SSIM-L {r['ssim_l']:.4f}"
          for _, r in mean.iterrows()],
        f"max |metric - value logged during training|: {maxdiff}",
        "outputs/<name>/<scene>.npy|.hdr|.png; sha256 of float outputs in metrics.json",
    ])
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
