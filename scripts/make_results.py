"""Build results/RESULTS.md from files written by the code (no hand-typed numbers).

python scripts/make_results.py
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
R = REPO / "results"
T = R / "tables"


def latest(exp: str, need: str = "metrics.json") -> Path | None:
    base = REPO / "runs" / exp
    if not base.exists():
        return None
    c = sorted(d for d in base.iterdir() if (d / need).exists())
    return c[-1] if c else None


def rel(p) -> str:
    return str(Path(p).relative_to(REPO)) if p else "-"


def jload(p):
    return json.loads(Path(p).read_text()) if p and Path(p).exists() else None


def csv_commented(p):
    p = Path(p)
    if not p.exists():
        return None, None
    lines = p.read_text().splitlines()
    head = lines[0][2:] if lines[0].startswith("#") else ""
    body = "\n".join(l for l in lines if not l.startswith("#"))
    return head, pd.read_csv(io.StringIO(body), dtype={"scene": str})


def f(x, n=3):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.{n}f}"


def main():
    rows, sections = [], []

    # ---- Exp 1
    plat = pd.read_csv(T / "platform.csv") if (T / "platform.csv").exists() else None
    if plat is not None and len(plat):
        ok = bool(plat.asserts_passed.all())
        key = "; ".join(f"{r.gpu_name} ({r.host}, job {r.slurm_job}): {r.train_it_per_s:.3f} it/s, peak {r.train_peak_mem_gb:.2f} GB, "
                        f"full-res fwd {r.fullres_forward_s:.3f} s / {r.fullres_peak_mem_gb:.2f} GB" for r in plat.itertuples())
        rows.append(("1 Platform", "done" if ok else "failed", ", ".join(plat.run_dir), key,
                     f"synthetic env check; M0 params {int(plat.n_params.iloc[0])}; TF32 off"))
        p0 = plat.iloc[0]
        sections.append("## Platform\n\n" + "\n".join([
            f"- Cluster: MSI (Slurm). Each run = one Slurm job with 1 GPU; the plan's local GPU ids are mapped to roles in launch/run_on_gpu.sh (gpu4/gpu5 refused). Physical GPU uuid recorded per run.",
            f"- {p0.cpu}; {p0.os}; Python {p0.python}; torch {p0.torch}; CUDA {p0.cuda}; cuDNN {p0.cudnn}; OpenCV {p0.opencv} (results/tables/platform.csv).",
            *[f"- Exp 1 run {r.run_dir}: {r.gpu_name} {r.gpu_mem_gb} GB, uuid {r.gpu_uuid}, partition {r.partition}, node {r.host}; asserts passed: {r.asserts_passed}" for r in plat.itertuples()],
            "- GPUs skipped: none were busy inside our allocations (Slurm gives exclusive GPUs).",
        ]))
    else:
        rows.append(("1 Platform", "not started", "-", "-", "-"))

    # ---- Exp 2
    e2 = latest("exp2")
    m2 = jload(e2 / "metrics.json") if e2 else None
    split = jload(REPO / "configs" / "split.json")
    if m2:
        s2 = m2["s2_stats"]
        key = (f"{m2['n_pairing_ok']}/{m2['n_scenes']} paired; best ref k counts {m2['best_k_counts']} (009 resolved to k=2, see notes); "
               f"t = 2^e (relative); s_2 median {s2['median']:.4f} [{s2['min']:.4f}, {s2['max']:.4f}]")
        rows.append(("2 Data", "done", rel(e2), key, f"split frozen: val {split['val']}" if split else "split missing"))
        c09 = jload(T / "scene009_reference_check.json")
        sections.append("## Data conventions (as verified)\n\n" + "\n".join([
            f"- {m2['n_scenes']} scenes, dtypes {m2['dtypes']}, shapes {m2['shapes']}; LDR normalised by 65535 (uint16 max).",
            f"- exposure.txt = log2 exposure; sets {m2['exposure_sets']}; t = 2^e are relative times (1,4,16 or 1,8,64).",
            f"- Pairing (sorted TIFF names <-> exposure rows; brightness increases with e over pixels unclipped in all frames): {m2['n_pairing_ok']}/{m2['n_scenes']} pass.",
            f"- Reference index: median per-scene median rel. error of H_gt ~ s_k L_k^2.2/t_k: {m2['median_med_rel_err']}; mean Pearson {m2['mean_pearson']}.",
            f"- Scene 009 flagged (k=3 by rel. err.); it shares bit-identical L_2/H_gt with 010 = {c09 and c09['L2_009_eq_L2_010']}/{c09 and c09['Hgt_009_eq_Hgt_010']}; log-gradient corr. k1/k2/k3 = "
            f"{f(c09 and c09['k1_logradient_corr'])}/{f(c09 and c09['k2_logradient_corr'])}/{f(c09 and c09['k3_logradient_corr'])} -> k=2 kept.",
            f"- s_2 (H_gt / (L_2^2.2/t_2)): {s2}. Target and inputs share units (s_2 ~ 1); nothing rescaled.",
            f"- Most saturated scene {m2['max_saturated_scene']} ({m2['max_saturated_pct']:.2f}% with m(L_2)>0.95).",
            f"- Split: NCC>{split['ncc_threshold']} on 48x72 L_2 thumbnails flagged {len(split['flagged_pairs'])} pairs in {len(split['groups'])} groups; val = {split['val']} ({len(split['train'])} train). Groups never split." if split else "",
            "- Our metric numbers are comparable only within this repo (SSIM implementation and normalisation differ across papers).",
        ]))
    else:
        rows.append(("2 Data", "not started", "-", "-", "-"))

    # ---- Exp 3
    h3, c3 = csv_commented(T / "classical_val.csv")
    if c3 is not None:
        mm = c3[c3.scene == "mean"].set_index("method")
        key = "; ".join(f"{k}: {mm.loc[k,'psnr_mu']:.2f} / {mm.loc[k,'psnr_l']:.2f} / {mm.loc[k,'ssim_mu']:.4f}" for k in ["C0", "C1", "C2", "C3"])
        rows.append(("3 Classical C0–C3", "done", rel(latest("classical-v1")), "PSNR-mu / PSNR-L / SSIM-mu: " + key,
                     f"C1 ghosts (fig_classical_ladder); fallback % C3 {mm.loc['C3','fallback_pct']:.3f}"))
    else:
        rows.append(("3 Classical C0–C3", "not started", "-", "-", "-"))

    # ---- Exp 4
    o4 = latest("overfit_M0")
    m4 = jload(o4 / "metrics.json") if o4 else None
    if m4:
        rows.append(("4 Overfit M0", "done (pass)" if m4["pass_5db"] else "done (FAIL)", rel(o4),
                     f"final PSNR-mu {m4['psnr_mu_final']:.2f} vs C0 {m4['c0_psnr_mu']:.2f} dB (+{m4['margin_over_c0_db']:.2f}); loss {m4['loss_first']:.4f} -> {m4['loss_last']:.4f}",
                     "pipeline check, 4 fixed crops"))
    else:
        rows.append(("4 Overfit M0", "not started", "-", "-", "-"))

    # ---- Exp 5
    p5 = latest("pilot_M0", "val_log.csv")
    ev5 = latest("pilot-eval-v1")
    rp = jload(T / "repro_check.json")
    if p5:
        tl = pd.read_csv(p5 / "train_log.csv")
        cfg = jload(p5 / "config.json")
        fin = (p5 / "FINISHED").exists()
        wall = jload(p5 / "wall_time.json")
        n_upd = int(tl.step.max())
        ntr = len(split["train"])
        txt = f"updates {n_upd} of {cfg['n_updates']}; wall {wall['train_wall_s']/3600:.2f} h; epochs ~ N*8/{ntr} = {n_upd*8/ntr:.0f}"
        if ev5:
            em = jload(ev5 / "metrics.json")
            s = em["selection"]["M0"]["step"]
            mv = em["mean"]["M0"]
            txt += f"; selected step {s}: PSNR-mu {mv['psnr_mu']:.2f}, PSNR-L {mv['psnr_l']:.2f}, SSIM-mu {mv['ssim_mu']:.4f}, SSIM-L {mv['ssim_l']:.4f}"
        if rp:
            txt += f"; repro check {'PASS' if rp['pass'] else 'FAIL'} (max |dPSNR| vs log {max(rp['max_abs_diff_vs_training_log_a'].values()):.2e}, outputs identical {all(rp['outputs_bit_identical_a_vs_b'].values())})"
        gpu = cfg["_gpu"]
        rows.append(("5 Pilot M0", "done" if fin and ev5 else "partial", f"{rel(p5)}; {rel(ev5)}", txt,
                     f"GPU {gpu.get('nvidia_smi_visible')} job {gpu.get('SLURM_JOB_ID')}"))
    else:
        rows.append(("5 Pilot M0", "not started", "-", "-", "-"))

    # ---- 6a/6b
    h6, c6 = csv_commented(T / "cue_diagnostics_val.csv")
    if c6 is not None:
        pp = c6[c6.scene == "pooled"]
        key = "; ".join(f"{r.frame}/{r.stratum}: err cons {r.mean_err_consistent:.4f} vs incons {r.mean_err_inconsistent:.4f}, AUROC {f(r.auroc_1_minus_r)} (n={r.n_pix})"
                        for r in pp.itertuples())
        rows.append(("6a/6b Cues", "done", rel(latest("cues-v1")), key, "estimated cues; see fig_reliability_maps.txt for a cue failure"))
    else:
        rows.append(("6a/6b Cues", "not started", "-", "-", "-"))

    # ---- 6c
    o3 = latest("overfit_M3")
    m6 = jload(o3 / "metrics.json") if o3 else None
    pt = (T / "pytest.txt").read_text().strip().splitlines()[-1] if (T / "pytest.txt").exists() else "pytest not recorded"
    if m6:
        rows.append(("6c Gate", "done", rel(o3), f"tests: {pt}; M3 overfit PSNR-mu {m6['psnr_mu_final']:.2f} (C0 {m6['c0_psnr_mu']:.2f}); params M3 {m6['n_params']}",
                     "gate implemented and verified to train on a fixed batch"))
    else:
        rows.append(("6c Gate", "not started", "-", f"tests: {pt}", "-"))

    # ---- 6d
    mb = jload(T / "matched_budget.json")
    hv, vv = csv_commented(T / "variants_val.csv")
    hd, dd = csv_commented(T / "variants_diffs.csv")
    if vv is not None and mb:
        mm = vv[vv.scene == "mean"].set_index("model")
        dm = dd.groupby("diff")[["psnr_mu", "psnr_l", "ssim_mu"]].mean()
        key = (f"N* = {mb['N_star']}; " + "; ".join(f"{k} (step {int(mm.loc[k,'step'])}): PSNR-mu {mm.loc[k,'psnr_mu']:.2f}" for k in mm.index)
               + "; mean diffs PSNR-mu: " + ", ".join(f"{k} {dm.loc[k,'psnr_mu']:+.3f}" for k in dm.index))
        rows.append(("6d M1–M3", "done (matched budget)", rel(latest("variants-eval-v1")), key, "preliminary, single seed, 5 validation scenes"))
    elif mb:
        rows.append(("6d M1–M3", "partial", "-", f"max steps reached {mb['max_step_reached']}; N* {mb['N_star']}", "preliminary"))
    else:
        rows.append(("6d M1–M3", "not started / running", "-", "-", "-"))

    out = ["# Results summary (factual only; every number traces to a file)", "",
           "Generated by scripts/make_results.py from files under runs/ and results/tables/.", "",
           "| Exp | Status | Run dir | Key numbers | Notes |", "|-----|--------|---------|-------------|-------|"]
    out += [f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in rows]
    out.append("")
    out += sections
    if h3:
        out += ["", "## Exp 3 table (mean rows)", "", f"Conventions: {h3}", "",
                c3[c3.scene == "mean"].drop(columns=["scene"]).to_markdown(index=False, floatfmt=".4f")]
    if vv is not None:
        out += ["", "## Exp 6d (preliminary, single seed, 5 validation scenes)", "", f"Conventions: {hv}", "",
                vv[vv.scene == "mean"].drop(columns=["scene"]).to_markdown(index=False, floatfmt=".4f"), "",
                "Per-scene differences (results/tables/variants_diffs.csv):", "",
                dd.to_markdown(index=False, floatfmt="+.4f")]
    out += ["", "## Figures produced (file -> what -> source run)", ""]
    for p in sorted((R / "figures").glob("*.png")):
        side = p.with_suffix(".txt")
        first = side.read_text().splitlines() if side.exists() else ["(no sidecar)"]
        src = next((l for l in first if "ource" in l), "")
        out.append(f"- results/figures/{p.name} -> {first[0]} {src}")
    out += ["", "## Problems encountered and what was tried (timestamped)", "",
            (REPO / "notes" / "problems.md").read_text().split("\n", 2)[-1].strip(), "",
            "## Not done / future work", "",
            "- Per plan: separate ablation of exposure vs FB cues; multiple seeds; Challenge123; the official test set (never downloaded).",
            ]
    (R / "RESULTS.md").write_text("\n".join(out) + "\n")
    print("\n".join(out[:16]))


if __name__ == "__main__":
    main()
