"""python -m hdr.train --config configs/<cfg>.json [--run-dir DIR]

Pilot mode (default): random 256x256 crops + flips/rot90 from the training scenes,
Adam(0.9,0.999) lr 1e-4, batch 8, fp32 (TF32 off), mu-law L1 loss. Every `eval_every`
updates: checkpoint (model, optimizer, update count, RNG states, config, git hash) and
full-resolution validation on the val scenes (all 4 metrics) -> val_log.csv.
Resumes automatically from the latest checkpoint of the newest run dir of the same
`exp` that is not finished (or of --run-dir).

Overfit mode (config "overfit": {...}): one fixed batch of crops (one per training scene
chosen with the seed), no augmentation; loss every 10 updates, PSNR-mu on the crops every
`psnr_every` updates, and C0 PSNR-mu on the same crops.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .data import SceneArrays, load_split, to_batch
from .data import CropSampler
from .metrics import METRIC_KEYS, all_metrics, mu_law_torch
from .model import build_model, count_params, model_forward
from .runlog import git_commit_text, make_run_dir, write_notes
from .torchutil import setup_torch

REPO = Path(__file__).resolve().parent.parent

DEFAULTS = dict(seed=5561, lr=1e-4, betas=[0.9, 0.999], batch=8, crop=256, augment=True, n_updates=40000,
                eval_every=2000, log_every=10, val_scenes="val", split="configs/split.json")


def l1_mu(pred, target):
    return (mu_law_torch(pred) - mu_law_torch(target)).abs().mean()


def needs_flow(variant: str) -> bool:
    return variant.upper() != "M0"


# --------------------------------------------------------------------------- validation
@torch.no_grad()
def predict_full(model, variant, scene: SceneArrays, device):
    """Full-resolution prediction for one scene -> (pred HWC float32, gates or None)."""
    d = scene.full()
    b = to_batch([d], [scene.t], device)
    pred, extra = model_forward(model, variant, b)
    p = pred[0].permute(1, 2, 0).float().cpu().numpy()
    gates = None if extra is None else [g[0, 0].cpu().numpy() for g in extra]
    return p, gates


def validate(model, variant, scenes: list[SceneArrays], device) -> list[dict]:
    was_training = model.training
    model.eval()
    prev = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    rows = []
    for s in scenes:
        p, _ = predict_full(model, variant, s, device)
        m = all_metrics(p, s.tgt.transpose(1, 2, 0))
        rows.append({"scene": s.scene_id, **m})
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = prev
    model.train(was_training)
    return rows


# --------------------------------------------------------------------------- checkpoints
def rng_state(sampler_rng) -> dict:
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_sampler": sampler_rng.bit_generator.state if sampler_rng is not None else None,
            "numpy_global": np.random.get_state()}


def save_ckpt(path: Path, model, opt, step, sampler_rng, cfg):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "rng": rng_state(sampler_rng),
                "config": cfg, "git": git_commit_text()}, tmp)
    os.replace(tmp, path)


def find_resume_dir(exp: str) -> Path | None:
    base = REPO / "runs" / exp
    if not base.exists():
        return None
    for d in sorted(base.iterdir(), reverse=True):
        if (d / "checkpoints" / "latest.pt").exists() and not (d / "FINISHED").exists():
            return d
    return None


def append_csv(path: Path, rows: list[dict]):
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- overfit helpers
def overfit_batch(cfg, train_ids, device, with_flow):
    of = cfg["overfit"]
    rng = np.random.default_rng(cfg["seed"])
    crops_file = REPO / of.get("crops_file", "configs/overfit_crops.json")
    if crops_file.exists():
        crops = json.loads(crops_file.read_text())["crops"]
    else:
        sids = sorted(rng.choice(train_ids, size=of["n_crops"], replace=False).tolist())
        crops = []
        for sid in sids:
            s = SceneArrays(sid, with_flow=False)
            y = int(rng.integers(0, s.H - cfg["crop"] + 1))
            x = int(rng.integers(0, s.W - cfg["crop"] + 1))
            crops.append({"scene": sid, "y": y, "x": x, "size": cfg["crop"]})
        crops_file.write_text(json.dumps({"seed": cfg["seed"], "crops": crops}, indent=2))
    items, ts = [], []
    for c in crops:
        s = SceneArrays(c["scene"], with_flow=with_flow)
        items.append(s.crop(c["y"], c["x"], c["size"]))
        ts.append(s.t)
    return crops, to_batch(items, ts, device), items, ts


def crop_metrics(pred: torch.Tensor, target: torch.Tensor) -> list[dict]:
    out = []
    for k in range(pred.shape[0]):
        p = pred[k].permute(1, 2, 0).detach().cpu().numpy()
        g = target[k].permute(1, 2, 0).cpu().numpy()
        out.append(all_metrics(p, g))
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    cfg = dict(DEFAULTS)
    cfg.update(json.loads(Path(args.config).read_text()))
    cfg["config_file"] = args.config
    variant = cfg["variant"].upper()
    exp = cfg["exp"]
    device = setup_torch(cfg["seed"])
    split = load_split(REPO / cfg["split"])
    with_flow = needs_flow(variant)
    overfit = "overfit" in cfg

    rd = Path(args.run_dir) if args.run_dir else (None if overfit else find_resume_dir(exp))
    resuming = rd is not None and (rd / "checkpoints" / "latest.pt").exists()
    if not resuming:
        rd = make_run_dir(REPO / "runs" / exp, cfg, run_dir=rd)
    else:
        (rd / f"config.resume-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(
            {"resumed": True, "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
             "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}, indent=2))
    ckdir = rd / "checkpoints"
    if not ckdir.exists():
        store = Path(os.environ.get("HDR_STORE", "/projects/standard/ee5561/shared/zha00175_hdr"))
        if store.exists():  # big files live in project space (home quota); symlink from the run dir
            real = store / "checkpoints" / rd.relative_to(REPO / "runs") if rd.is_relative_to(REPO / "runs") else \
                store / "checkpoints" / rd.name
            real.mkdir(parents=True, exist_ok=True)
            ckdir.symlink_to(real)
        else:
            ckdir.mkdir()
    print(f"run dir {rd} (resume={resuming})", flush=True)

    model = build_model(variant).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]))
    n_params = count_params(model)

    t_load = time.time()
    if overfit:
        crops, batch, items, ts = overfit_batch(cfg, split["train"], device, with_flow)
        sampler = None
        val_scenes = []
    else:
        train_scenes = [SceneArrays(s, with_flow) for s in split["train"]]
        sampler = CropSampler(train_scenes, cfg["crop"], cfg["augment"], cfg["seed"])
        val_scenes = [SceneArrays(s, with_flow) for s in split[cfg["val_scenes"]]]
    print(f"data loaded in {time.time()-t_load:.1f}s", flush=True)

    step = 0
    if resuming:
        ck = torch.load(ckdir / "latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        torch.set_rng_state(ck["rng"]["torch"])
        if ck["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        if sampler is not None and ck["rng"]["numpy_sampler"] is not None:
            sampler.rng.bit_generator.state = ck["rng"]["numpy_sampler"]
        np.random.set_state(ck["rng"]["numpy_global"])
        print(f"resumed at step {step}", flush=True)

    train_log = rd / "train_log.csv"
    val_log = rd / "val_log.csv"
    model.train()
    t_start = time.time()
    wall_prev = 0.0
    wf = rd / "wall_time.json"
    if wf.exists():
        wall_prev = json.loads(wf.read_text())["train_wall_s"]

    if overfit:
        of = cfg["overfit"]
        c0 = crop_metrics(batch["feat"][:, 1, 3:6], batch["target"])  # C0 = H_2
        c0_psnr_mu = float(np.mean([m["psnr_mu"] for m in c0]))
        with torch.no_grad():
            p0, e0 = model_forward(model, variant, batch)
        snaps = {"pred_step0": p0.cpu().numpy()}
        psnr_rows = []
        m = crop_metrics(p0, batch["target"])
        psnr_rows.append({"step": 0, "psnr_mu": float(np.mean([x["psnr_mu"] for x in m])),
                          **{f"psnr_mu_crop{k}": x["psnr_mu"] for k, x in enumerate(m)}})
        loss_rows = []
        for step in range(1, cfg["n_updates"] + 1):
            opt.zero_grad(set_to_none=True)
            pred, _ = model_forward(model, variant, batch)
            loss = l1_mu(pred, batch["target"])
            loss.backward()
            gn = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None)))
            opt.step()
            if not np.isfinite(float(loss)):
                raise FloatingPointError(f"non-finite loss at {step}")
            if step % cfg["log_every"] == 0 or step == 1:
                loss_rows.append({"step": step, "loss": float(loss), "grad_norm": gn})
            if step % of["psnr_every"] == 0:
                with torch.no_grad():
                    pe, _ = model_forward(model, variant, batch)
                m = crop_metrics(pe, batch["target"])
                psnr_rows.append({"step": step, "psnr_mu": float(np.mean([x["psnr_mu"] for x in m])),
                                  **{f"psnr_mu_crop{k}": x["psnr_mu"] for k, x in enumerate(m)}})
                print(f"step {step} loss {float(loss):.5f} psnr_mu {psnr_rows[-1]['psnr_mu']:.3f} (C0 {c0_psnr_mu:.3f})",
                      flush=True)
        with torch.no_grad():
            pf, ef = model_forward(model, variant, batch)
        snaps["pred_final"] = pf.cpu().numpy()
        snaps["target"] = batch["target"].cpu().numpy()
        snaps["ldr"] = batch["feat"][:, :, :3].cpu().numpy()
        if ef is not None:
            snaps["gate1"] = ef[0].cpu().numpy()
            snaps["gate3"] = ef[1].cpu().numpy()
            snaps["cues"] = batch["cues"].cpu().numpy()
            snaps["ldr_w"] = batch["featw"][:, :, :3].cpu().numpy()
        np.savez_compressed(rd / "overfit_snapshots.npz", **snaps)
        append_csv(train_log, loss_rows)
        append_csv(rd / "psnr_log.csv", psnr_rows)
        final_m = crop_metrics(pf, batch["target"])
        res = {"variant": variant, "n_params": n_params, "n_updates": cfg["n_updates"], "crops": crops,
               "loss_first": loss_rows[0]["loss"], "loss_last": loss_rows[-1]["loss"],
               "psnr_mu_step0": psnr_rows[0]["psnr_mu"], "psnr_mu_final": float(np.mean([x["psnr_mu"] for x in final_m])),
               "psnr_mu_final_per_crop": [x["psnr_mu"] for x in final_m],
               "c0_psnr_mu": c0_psnr_mu, "c0_psnr_mu_per_crop": [x["psnr_mu"] for x in c0],
               "wall_s": time.time() - t_start}
        res["margin_over_c0_db"] = res["psnr_mu_final"] - c0_psnr_mu
        res["pass_5db"] = bool(res["margin_over_c0_db"] >= 5.0)
        res["loss_ratio_last_first"] = res["loss_last"] / res["loss_first"]
        (rd / "metrics.json").write_text(json.dumps(res, indent=2))
        save_ckpt(ckdir / "final.pt", model, opt, cfg["n_updates"], None, cfg)
        write_notes(rd, [
            f"overfit check ({variant}), not a generalization result; {len(crops)} fixed crops, batch {len(crops)}, no augmentation",
            f"crops: {crops}",
            f"{cfg['n_updates']} updates, lr {cfg['lr']}, fp32 (TF32 off); params {n_params}",
            f"loss {res['loss_first']:.5f} -> {res['loss_last']:.5f}",
            f"PSNR-mu on crops: step0 {res['psnr_mu_step0']:.3f} -> final {res['psnr_mu_final']:.3f} dB; C0 {c0_psnr_mu:.3f} dB; margin {res['margin_over_c0_db']:.3f} dB; pass(>=5 dB) {res['pass_5db']}",
            f"wall {res['wall_s']:.0f} s",
        ])
        (rd / "FINISHED").write_text("done\n")
        print(json.dumps(res, indent=2))
        return

    # ---------------------------------------------------------------- pilot training
    losses = []
    t_last = time.time()
    while step < cfg["n_updates"]:
        b = sampler.sample(cfg["batch"], device)
        opt.zero_grad(set_to_none=True)
        pred, _ = model_forward(model, variant, b)
        loss = l1_mu(pred, b["target"])
        loss.backward()
        opt.step()
        step += 1
        lv = float(loss)
        if not np.isfinite(lv):
            raise FloatingPointError(f"non-finite loss at step {step}")
        losses.append(lv)
        if step % cfg["log_every"] == 0:
            now = time.time()
            append_csv(train_log, [{"step": step, "loss": float(np.mean(losses)), "loss_last": lv,
                                    "it_per_s": cfg["log_every"] / (now - t_last),
                                    "wall_s": wall_prev + now - t_start}])
            t_last = now
            losses = []
        if step % cfg["eval_every"] == 0 or step == cfg["n_updates"]:
            wall = wall_prev + time.time() - t_start
            ck = ckdir / f"step_{step:06d}.pt"
            save_ckpt(ck, model, opt, step, sampler.rng, cfg)
            shutil.copyfile(ck, ckdir / "latest.tmp")
            os.replace(ckdir / "latest.tmp", ckdir / "latest.pt")
            t_v = time.time()
            rows = validate(model, variant, val_scenes, device)
            mean = {k: float(np.mean([r[k] for r in rows])) for k in METRIC_KEYS}
            rows = [{"step": step, "wall_s": wall, **r} for r in rows] + [{"step": step, "wall_s": wall, "scene": "mean", **mean}]
            append_csv(val_log, rows)
            wf.write_text(json.dumps({"train_wall_s": wall, "step": step}))
            print(f"step {step} val psnr_mu {mean['psnr_mu']:.3f} psnr_l {mean['psnr_l']:.3f} ssim_mu {mean['ssim_mu']:.4f} "
                  f"(val {time.time()-t_v:.0f}s, wall {wall/3600:.2f}h)", flush=True)
            t_last = time.time()

    import pandas as pd

    vl = pd.read_csv(val_log, dtype={"scene": str})
    vm = vl[vl.scene == "mean"].drop_duplicates("step", keep="last")
    best = vm.loc[vm.psnr_mu.idxmax()]
    res = {"variant": variant, "n_params": n_params, "n_updates": step, "best_step": int(best.step),
           "best_val": {k: float(best[k]) for k in METRIC_KEYS}, "train_wall_s": json.loads(wf.read_text())["train_wall_s"],
           "n_train_scenes": len(split["train"]), "approx_epochs": step * cfg["batch"] / len(split["train"])}
    (rd / "metrics.json").write_text(json.dumps(res, indent=2))
    write_notes(rd, [
        f"{variant} pilot; {step} updates, batch {cfg['batch']}, lr {cfg['lr']}, fp32 (TF32 off), seed {cfg['seed']}",
        f"params {n_params}; train wall {res['train_wall_s']/3600:.2f} h; approx epochs (N*8/#train) {res['approx_epochs']:.0f}",
        f"best val PSNR-mu {res['best_val']['psnr_mu']:.3f} at step {res['best_step']} (5 val scenes, full image)",
        "val_log.csv has per-scene + mean metrics every eval step; checkpoints/step_*.pt kept for matched-budget selection",
    ])
    (rd / "FINISHED").write_text("done\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
