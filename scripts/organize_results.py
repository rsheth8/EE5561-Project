"""Copy every result into results/by_step/<step>/ with a README per step.

python scripts/organize_results.py

Originals in results/figures, results/tables and runs/ are left in place (the scripts
write there); this folder is a read-only, per-step view for the report. Rerun after any
result changes. Large files (checkpoints, full-resolution outputs) are not copied.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIG, TAB, RUNS = REPO / "results" / "figures", REPO / "results" / "tables", REPO / "runs"
OUT = REPO / "results" / "by_step"
RUN_FILES = ["config.json", "metrics.json", "metrics.csv", "notes.md", "stats.json", "train_log.csv", "val_log.csv",
             "psnr_log.csv", "wall_time.json", "vs_val_log.csv", "nvidia_smi.txt", "git_commit.txt"]

RAHIL_NOTE = (
    "> **Note for Rahil:** this step was done quickly, as a provisional version, only so that Zijian's later steps "
    "(Z2–Z8, Exp 4–6) could run on time. Feel free to replace it with your own implementation and results.\n"
    ">\n"
    "> If you replace it, Zijian's steps depend on these pieces, so please keep them or tell Zijian so the affected runs can be re-done:\n"
    "> - `hdr/io.py`, `hdr/data.py`: per-scene arrays (`ldr`, `exposure`, `radiance`, `features`, `target`; float32, middle-exposure reference, t = 2^e, LDR / 65535) and the cache in `cache/`. Used by `hdr/train.py`, `hdr/evaluate.py`, `hdr/diagnostics.py`, `scripts/precompute_flow.py`.\n"
    "> - `configs/split.json`: the 5 validation scenes used for checkpoint selection and every val number in Exp 5/6. Changing it means re-running Exp 5/6d evaluation (and ideally training).\n"
    "> - `hdr/metrics.py`, `hdr/display.py`: PSNR/SSIM (mu-law, mu = 5000) and the display mapping. Changing PSNR/SSIM or the display only needs re-running `hdr.evaluate` and the report scripts; but the training loss uses `mu_law_torch` from `hdr/metrics.py`, so changing the mu-law itself means retraining.\n"
    "> - C0 = H_2 is the pass line of the overfit checks (Exp 4, 6c); C1–C3 numbers appear next to M0 in `Z3-Z4_pilot_M0/tables/val_comparison.csv`; `ghost_crop.json` fixes the scene/crop reused by `fig_val_comparison`.\n"
    "> - C3 uses Zijian's alignment module (`hdr/alignment.py`, flow cache)."
)

STEPS = [
    {"dir": "S4-B1_platform", "title": "S4 + B1 (Exp 1) — Platform check", "owner": "Together (shared setup, guide B1)",
     "results_row": "1 Platform",
     "what": "Synthetic GPU check on the MSI A100s: forward/backward/update of M0, throughput (it/s), peak memory, full-resolution forward time. No HDR data involved; not an HDR result.",
     "figs": [], "tabs": ["platform.csv"], "runs": ["exp1/*"]},
    {"dir": "R1-R3_data_and_split", "note": RAHIL_NOTE, "title": "R1–R3 (Exp 2) — Dataset verification and validation split", "owner": "Rahil (guide R1–R3)",
     "results_row": "2 Data",
     "what": "File pairing (TIFF <-> exposure.txt), exposure convention t = 2^e, reference index and scale of H_gt, saturation statistics, scene inspections (001, 027, 036), related-scene check and frozen split (5 val scenes).",
     "figs": ["fig_saturation_mask", "related_scenes"],
     "tabs": ["dataset_inventory.csv", "ref_scale_diag.csv", "scene009_reference_check.json"],
     "extra": [("configs/split.json", "split.json")],
     "data_check": True, "runs": ["exp2/*", "split/*"]},
    {"dir": "R4_classical_C0-C3", "note": RAHIL_NOTE, "title": "R4 (Exp 3) — Classical merges C0–C3", "owner": "Rahil (guide R4)",
     "results_row": "3 Classical",
     "what": "C0 reference only; C1 hat-weighted merge without alignment; C2 reference-fallback deghosting; C3 = C2 on flow-aligned frames. Metric table on the 5 val scenes, ghost comparison, error maps, good case vs classified failure.",
     "figs": ["fig_classical_ladder", "fig_classical_error", "fig_classical_cases"],
     "tabs": ["classical_val.csv", "ghost_crop.json"], "extra": [("configs/classical.json", "classical.json")],
     "runs": ["classical-v1/*"]},
    {"dir": "Z5_alignment_warp", "title": "Z5 — Alignment module (DIS flow + backward warp)", "owner": "Zijian (guide Z5)",
     "results_row": None,
     "what": "Warp convention and checks: zero-flow identity, known translation (direction/sign), forward-backward consistency; flow cache for all scenes. Used by C3, M1–M3 and Exp 6. Flow method is OpenCV DIS (guide suggested e.g. Farneback).",
     "figs": ["fig_warp_check"], "tabs": ["warp_check.json"], "runs": ["flow/*"]},
    {"dir": "Z2-Z4_overfit_M0", "title": "Z2 + Z4 (Exp 4) — Overfit check (M0)", "owner": "Zijian (guide Z2, Z4)",
     "results_row": "4 Overfit",
     "what": "M0 trained on one fixed batch of 4 crops for 3000 updates. Pipeline/debugging check (pass = PSNR-mu at least 5 dB above C0 on the same crops); not a generalization result.",
     "figs": ["fig_overfit_curve", "fig_overfit_before_after"], "tabs": [],
     "extra": [("configs/overfit_M0.json", "overfit_M0.json"), ("configs/overfit_crops.json", "overfit_crops.json")],
     "runs": ["overfit_M0/*"]},
    {"dir": "Z3-Z4_pilot_M0", "title": "Z3 + Z4 + B2 (Exp 5) — Pilot training M0 + reproduction check", "owner": "Zijian (guide Z3, Z4)",
     "results_row": "5 Pilot",
     "what": "M0 trained 50k updates (single seed); checkpoint selected on val PSNR-mu; evaluated on the 5 val scenes at full resolution and compared with C0–C3. Reproduction check: two fresh evaluation processes vs the values logged in training.",
     "figs": ["fig_train_curve", "fig_val_comparison"], "tabs": ["val_comparison.csv", "repro_check.json"],
     "extra": [("configs/pilot_M0.json", "pilot_M0.json"), ("configs/pilot-eval.json", "pilot-eval.json")],
     "runs": ["pilot_M0/*", "pilot-eval-v1/*"]},
    {"dir": "Z6_reliability_cues", "title": "Z6 (Exp 6a/6b) — Reliability cue maps and whether cues predict alignment error",
     "owner": "Zijian (guide Z6)", "results_row": "6a/6b",
     "what": "Estimated cues (warp validity, forward-backward consistency, exposure cues q_ref / q_i^w) visualized on 2 val scenes, with a marked cue-failure example; statistics of warped-frame error vs FB consistency (AUROC), split by whether the reference is clipped.",
     "figs": ["fig_reliability_maps"], "tabs": ["cue_diagnostics_val.csv"], "runs": ["cues-v1/*"]},
    {"dir": "Z7_gate", "title": "Z7 (Exp 6c) — Gate implementation checks", "owner": "Zijian (guide Z7)",
     "results_row": "6c",
     "what": "Unit tests (shapes, gate = 0 where warp invalid, init 0.5, M2/M3 identical parameters, M2 ignores cue values, finite outputs) and M3 overfit check with gate maps. Verifies the gate trains; not evidence that it helps.",
     "figs": ["fig_overfit_M3_curve", "fig_gate_maps"], "tabs": ["pytest.txt"],
     "extra": [("configs/overfit_M3.json", "overfit_M3.json")], "runs": ["overfit_M3/*"]},
    {"dir": "Z8_controls_M0-M3", "title": "Z8 (Exp 6d) — Controls M0/M1/M2/M3 under a matched budget", "owner": "Zijian (guide Z8); interpretation together",
     "results_row": "6d",
     "what": "M0 original inputs; M1 aligned inputs; M2 original+aligned with gate and constant cues; M3 same gate with real cues. Same data, loss, 50k-update budget, checkpoint rule. Preliminary: single seed, 5 validation scenes.",
     "figs": ["fig_variants_curve"], "tabs": ["variants_val.csv", "variants_diffs.csv", "matched_budget.json"],
     "extra": [(f"configs/pilot_{v}.json", f"pilot_{v}.json") for v in ("M1", "M2", "M3")] + [("configs/variants-eval.json", "variants-eval.json")],
     "runs": ["pilot_M1/*", "pilot_M2/*", "pilot_M3/*", "variants-eval-v1/*"]},
]


def results_rows() -> dict[str, list[str]]:
    p = REPO / "results" / "RESULTS.md"
    rows = {}
    for line in p.read_text().splitlines() if p.exists() else []:
        if line.startswith("| ") and not line.startswith("| Exp") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
            rows[cells[0]] = cells
    return rows


def table_header(p: Path) -> str:
    if p.suffix == ".csv":
        first = p.read_text().splitlines()[0]
        return first[2:] if first.startswith("#") else "(no convention line)"
    if p.suffix == ".json":
        return "JSON; see file"
    return p.read_text().strip().splitlines()[-1]


def copy_run(src: Path, dst: Path) -> list[str]:
    dst.mkdir(parents=True, exist_ok=True)
    got = []
    for name in RUN_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
            got.append(name)
    for fig in sorted((src / "figures").glob("*")) if (src / "figures").exists() else []:
        (dst / "figures").mkdir(exist_ok=True)
        shutil.copy2(fig, dst / "figures" / fig.name)
        got.append(f"figures/{fig.name}")
    return got


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rr = results_rows()
    index = ["# Results by step", "",
             "Generated by `scripts/organize_results.py` (copies; originals stay in results/figures, results/tables, runs/).",
             "Owners follow the team guide's split (HDR_step_by_step_guide, Nov 23). Checkpoints and full-resolution outputs are not copied; their paths are listed in each step's run records.", "",
             "Rahil's steps (R1–R3, R4) were done quickly as provisional versions so that Zijian's steps could run; Rahil may replace them with his own results (see the note in those READMEs for what Zijian's steps depend on).", "",
             "Folder names use the guide's step labels (S = shared setup, R = Rahil, Z = Zijian, B = batch jobs, I = integration/report).", "",
             "| Folder | Step | Owner (guide) |", "|---|---|---|"]
    for s in STEPS:
        d = OUT / s["dir"]
        d.mkdir()
        lines = [f"# {s['title']}", "", f"**Owner:** {s['owner']}", ""]
        if s.get("note"):
            lines += [s["note"], ""]
        lines += [f"**What:** {s['what']}", ""]
        row = next((v for k, v in rr.items() if s["results_row"] and k.startswith(s["results_row"])), None)
        if row:
            lines += [f"**Status (from results/RESULTS.md):** {row[1]}", "", f"**Key numbers:** {row[3]}", ""]
            if row[4] and row[4] != "-":
                lines += [f"**Notes:** {row[4]}", ""]
        if s["figs"]:
            lines += ["## Figures", ""]
            (d / "figures").mkdir()
            for f in s["figs"]:
                for ext in (".png", ".txt"):
                    if (FIG / f"{f}{ext}").exists():
                        shutil.copy2(FIG / f"{f}{ext}", d / "figures" / f"{f}{ext}")
                cap = (FIG / f"{f}.txt").read_text().strip().splitlines() if (FIG / f"{f}.txt").exists() else ["(missing)"]
                lines += [f"- `figures/{f}.png` — " + " ".join(cap), ""]
        if s["tabs"] or s.get("extra"):
            lines += ["## Tables and configs", ""]
            (d / "tables").mkdir()
            for t in s["tabs"]:
                if (TAB / t).exists():
                    shutil.copy2(TAB / t, d / "tables" / t)
                    lines.append(f"- `tables/{t}` — {table_header(TAB / t)}")
                else:
                    lines.append(f"- `tables/{t}` — MISSING")
            for src, name in s.get("extra", []):
                if (REPO / src).exists():
                    shutil.copy2(REPO / src, d / "tables" / name)
                    lines.append(f"- `tables/{name}` — copy of `{src}`")
            lines.append("")
        if s.get("data_check"):
            lines += ["## Scene inspections (runs/data-check)", ""]
            for sd in sorted((RUNS / "data-check").glob("*/*")):
                tgt = d / "scene_inspections" / sd.parent.name
                copy_run(sd, tgt)
                why = json.loads((sd / "config.json").read_text()).get("why", "")
                lines.append(f"- `scene_inspections/{sd.parent.name}/figures/montage.png` — scene {sd.parent.name}; chosen because: {why}")
            lines.append("")
        lines += ["## Run records (small files copied from runs/)", ""]
        for pat in s["runs"]:
            for rd in sorted(RUNS.glob(pat)):
                if not rd.is_dir():
                    continue
                tgt = d / "run_records" / rd.parent.name / rd.name
                got = copy_run(rd, tgt)
                big = [x for x in ("checkpoints", "outputs") if (rd / x).exists()]
                extra = f"; large files left in place: {', '.join(f'runs/{rd.parent.name}/{rd.name}/{b}/' for b in big)}" if big else ""
                lines.append(f"- `run_records/{rd.parent.name}/{rd.name}/` ← `runs/{rd.parent.name}/{rd.name}` ({', '.join(got)}){extra}")
        lines.append("")
        (d / "README.md").write_text("\n".join(lines))
        index.append(f"| [{s['dir']}]({s['dir']}/README.md) | {s['title']} | {s['owner']}{' — provisional, replaceable' if s.get('note') else ''} |")
    index += ["", "Guide steps with no folder here: S1–S3, S5 (MSI/Jupyter setup, no result files); R5 (handoff note to Zijian, not written as a separate document); "
              "Z1 (learning HDR conventions, a personal task); B3 (code exchange); I1 (cross-reproduction between Rahil and Zijian, not done yet); "
              "I2–I4 (packaging, LaTeX report, final check, not done yet)."]
    index += ["", "Also: `results/RESULTS.md` (overall summary), `notes/problems.md` (timestamped problems log)."]
    shutil.copy2(REPO / "notes" / "problems.md", OUT / "problems_log.md")
    index[-1] += " A copy of the problems log is `problems_log.md` here."
    (OUT / "README.md").write_text("\n".join(index) + "\n")
    print((OUT / "README.md").read_text())


if __name__ == "__main__":
    main()
