# R1–R3 (Exp 2) — Dataset verification and validation split

**Owner:** Rahil (guide R1–R3)

> **Note for Rahil:** this step was done quickly, as a provisional version, only so that Zijian's later steps (Z2–Z8, Exp 4–6) could run on time. Feel free to replace it with your own implementation and results.
>
> If you replace it, Zijian's steps depend on these pieces, so please keep them or tell Zijian so the affected runs can be re-done:
> - `hdr/io.py`, `hdr/data.py`: per-scene arrays (`ldr`, `exposure`, `radiance`, `features`, `target`; float32, middle-exposure reference, t = 2^e, LDR / 65535) and the cache in `cache/`. Used by `hdr/train.py`, `hdr/evaluate.py`, `hdr/diagnostics.py`, `scripts/precompute_flow.py`.
> - `configs/split.json`: the 5 validation scenes used for checkpoint selection and every val number in Exp 5/6. Changing it means re-running Exp 5/6d evaluation (and ideally training).
> - `hdr/metrics.py`, `hdr/display.py`: PSNR/SSIM (mu-law, mu = 5000) and the display mapping. Changing PSNR/SSIM or the display only needs re-running `hdr.evaluate` and the report scripts; but the training loss uses `mu_law_torch` from `hdr/metrics.py`, so changing the mu-law itself means retraining.
> - C0 = H_2 is the pass line of the overfit checks (Exp 4, 6c); C1–C3 numbers appear next to M0 in `Z3-Z4_pilot_M0/tables/val_comparison.csv`; `ghost_crop.json` fixes the scene/crop reused by `fig_val_comparison`.
> - C3 uses Zijian's alignment module (`hdr/alignment.py`, flow cache).

**What:** File pairing (TIFF <-> exposure.txt), exposure convention t = 2^e, reference index and scale of H_gt, saturation statistics, scene inspections (001, 027, 036), related-scene check and frozen split (5 val scenes).

**Status (from results/RESULTS.md):** done

**Key numbers:** 74/74 paired; best ref k counts {'2': 73, '3': 1} (009 resolved to k=2, see notes); t = 2^e (relative); s_2 median 1.0154 [0.9982, 1.0696]

**Notes:** split frozen: val ['025', '045', '051', '056', '062']

## Figures

- `figures/fig_saturation_mask.png` — Scenes 027, 036 (training split), full frames, no crop. This is a CLIPPING mask computed from the reference exposure alone: m(L_2) = max over RGB; over-exposed m>0.95 (magenta), under-exposed m<0.05 (cyan). It is NOT a motion mask. scene 027: over-exposed 36.14%, under-exposed 0.00% of pixels scene 036: over-exposed 29.52%, under-exposed 0.00% of pixels Display: LDR shown as stored (normalised by dtype max, x255, rounded); overlay alpha 0.55. Data inspection only, no method/training. Produced by scripts/fig_saturation.py.

- `figures/related_scenes.png` — Related-scene check: middle exposure L_2, 48x72 grayscale thumbnails, z-normalised; NCC threshold 0.9. 32 pairs with NCC > 0.9. Pairs shown: 004-005 NCC=1.000; 004-030 NCC=0.951; 005-030 NCC=0.951; 007-031 NCC=0.953; 009-010 NCC=1.000; 009-032 NCC=0.960; 009-033 NCC=0.960; 010-032 NCC=0.960; 010-033 NCC=0.960; 011-012 NCC=1.000; 011-034 NCC=0.958; 012-034 NCC=0.958; 013-014 NCC=1.000; 013-035 NCC=0.942; 013-036 NCC=0.942; 014-035 NCC=0.942; 014-036 NCC=0.942; 018-019 NCC=0.969; 020-021 NCC=1.000; 022-037 NCC=0.975; 023-024 NCC=1.000; 028-038 NCC=0.983; 032-033 NCC=1.000; 035-036 NCC=1.000; 040-041 NCC=1.000; 040-042 NCC=1.000; 041-042 NCC=1.000; 052-053 NCC=1.000; 064-065 NCC=1.000; 066-067 NCC=1.000; 068-069 NCC=0.958; 071-072 NCC=1.000 Groups (never split): [["004", "005", "030"], ["007", "031"], ["009", "010", "032", "033"], ["011", "012", "034"], ["013", "014", "035", "036"], ["018", "019"], ["020", "021"], ["022", "037"], ["023", "024"], ["028", "038"], ["040", "041", "042"], ["052", "053"], ["064", "065"], ["066", "067"], ["068", "069"], ["071", "072"]] Display mapping: LDR shown as stored (normalised by dtype max, x255, rounded). Full frames, no crop. Not a training result. Source run: runs/split/20261123-143206

## Tables and configs

- `tables/dataset_inventory.csv` — (no convention line)
- `tables/ref_scale_diag.csv` — (no convention line)
- `tables/scene009_reference_check.json` — JSON; see file
- `tables/split.json` — copy of `configs/split.json`

## Scene inspections (runs/data-check)

- `scene_inspections/001/figures/montage.png` — scene 001; chosen because: first scene; exposure convention example (e = 0,2,4 -> t = 1,4,16)
- `scene_inspections/027/figures/montage.png` — scene 027; chosen because: largest saturated fraction: 36.14% of pixels with m(L_2)>0.95 (dataset_inventory.csv)
- `scene_inspections/036/figures/montage.png` — scene 036; chosen because: visibly large motion chosen by eye: subject shifted left in L_1, hands/box position differ across L_1-L_3; median DIS flow 30-39 px (runs/flow metrics.csv)

## Run records (small files copied from runs/)

- `run_records/exp2/20261123-143143/` ← `runs/exp2/20261123-143143` (config.json, metrics.json, metrics.csv, notes.md, git_commit.txt)
- `run_records/split/20261123-143206/` ← `runs/split/20261123-143206` (config.json, metrics.json, notes.md, git_commit.txt, figures/related_scenes.png)
