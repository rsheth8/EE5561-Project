# Z3 + Z4 + B2 (Exp 5) — Pilot training M0 + reproduction check

**Owner:** Zijian (guide Z3, Z4)

**What:** M0 trained 50k updates (single seed); checkpoint selected on val PSNR-mu; evaluated on the 5 val scenes at full resolution and compared with C0–C3. Reproduction check: two fresh evaluation processes vs the values logged in training.

**Status (from results/RESULTS.md):** done

**Key numbers:** updates 50000 of 50000; wall 6.20 h; epochs ~ N*8/69 = 5797; selected step 50000: PSNR-mu 42.46, PSNR-L 40.07, SSIM-mu 0.9892, SSIM-L 0.9841; repro check PASS (max |dPSNR| vs log 0.00e+00, outputs identical True)

**Notes:** GPU 0, NVIDIA A100-SXM4-40GB, GPU-e7f7c5b3-78c4-325a-9241-bd4954f3162e, 00000000:47:00.0, 40960 MiB job 7913724

## Figures

- `figures/fig_train_curve.png` — M0 pilot (finished), single seed 5561, batch 8, lr 1e-4, fp32; val = 5 validation scenes, full image, mu=5000. Left: training loss logged every 10 updates (mean of those 10) and its 50-point moving average. Right: mean val PSNR-mu every 2000 updates; C0/C3 levels from the classical run. Selected step 50000 = best val PSNR-mu (val-selected). Sources: runs/pilot_M0/20261123-150849/train_log.csv, val_log.csv; runs/classical-v1/20261123-145641/metrics.csv.

- `figures/fig_val_comparison.png` — Validation scene 062, crop y=640, x=557, 300x300 - same scene and crop as fig_classical_ladder.png. C1, C3 classical (no training); M0 = AHDRNet-style pilot, step 50000, val-selected, single seed; GT. Display: to_uint8(H) = round(255*T(clip(H,0,1))), T = mu-law, mu=5000, identical for every panel. PSNR-mu from results/tables/val_comparison.csv.

## Tables and configs

- `tables/val_comparison.csv` — 5 validation scenes, full image, no border excluded, mu = 5000; C0-C3 from runs/classical-v1/20261123-145641; M0 = pilot, val-selected step 50000, from runs/pilot-eval-v1/20261123-212943 (single seed)
- `tables/repro_check.json` — JSON; see file
- `tables/pilot_M0.json` — copy of `configs/pilot_M0.json`
- `tables/pilot-eval.json` — copy of `configs/pilot-eval.json`

## Run records (small files copied from runs/)

- `run_records/pilot_M0/20261123-150849/` ← `runs/pilot_M0/20261123-150849` (config.json, metrics.json, notes.md, train_log.csv, val_log.csv, wall_time.json, git_commit.txt); large files left in place: runs/pilot_M0/20261123-150849/checkpoints/
- `run_records/pilot-eval-v1/20261123-212943/` ← `runs/pilot-eval-v1/20261123-212943` (config.json, metrics.json, metrics.csv, notes.md, vs_val_log.csv, git_commit.txt); large files left in place: runs/pilot-eval-v1/20261123-212943/outputs/
- `run_records/pilot-eval-v1/20261123-213012/` ← `runs/pilot-eval-v1/20261123-213012` (config.json, metrics.json, metrics.csv, notes.md, vs_val_log.csv, git_commit.txt); large files left in place: runs/pilot-eval-v1/20261123-213012/outputs/
