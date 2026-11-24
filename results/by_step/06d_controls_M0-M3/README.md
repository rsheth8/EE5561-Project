# Exp 6d — Controls M0/M1/M2/M3 under a matched budget

**Owner:** Zijian (guide Z8); interpretation together

**What:** M0 original inputs; M1 aligned inputs; M2 original+aligned with gate and constant cues; M3 same gate with real cues. Same data, loss, 50k-update budget, checkpoint rule. Preliminary: single seed, 5 validation scenes.

**Status (from results/RESULTS.md):** done (matched budget)

**Key numbers:** N* = 50000; M0 (step 50000): PSNR-mu 42.46; M1 (step 50000): PSNR-mu 43.22; M2 (step 50000): PSNR-mu 44.13; M3 (step 50000): PSNR-mu 44.09; mean diffs PSNR-mu: M1-M0 +0.763, M2-M1 +0.913, M3-M2 -0.043

**Notes:** preliminary, single seed, 5 validation scenes

## Figures

- `figures/fig_variants_curve.png` — Validation PSNR-mu (5 val scenes, full image, mu=5000) vs updates for M0-M3; preliminary, single seed, 5 validation scenes. Matched-budget N* = 50000; selected steps {'M0': 50000, 'M1': 50000, 'M2': 50000, 'M3': 50000}. Sources: runs/pilot_M*/<ts>/val_log.csv, runs/variants-eval-v1/20261123-223749.

## Tables and configs

- `tables/variants_val.csv` — preliminary, single seed, 5 validation scenes; full image, mu = 5000; matched-budget rule N* = 50000 (largest eval step all four variants reached); each variant's checkpoint = best val PSNR-mu among steps <= N*: {'M0': 50000, 'M1': 50000, 'M2': 50000, 'M3': 50000}; source run runs/variants-eval-v1/20261123-223749
- `tables/variants_diffs.csv` — preliminary, single seed, 5 validation scenes; full image, mu = 5000; matched-budget rule N* = 50000 (largest eval step all four variants reached); each variant's checkpoint = best val PSNR-mu among steps <= N*: {'M0': 50000, 'M1': 50000, 'M2': 50000, 'M3': 50000}; source run runs/variants-eval-v1/20261123-223749
- `tables/matched_budget.json` — JSON; see file
- `tables/pilot_M1.json` — copy of `configs/pilot_M1.json`
- `tables/pilot_M2.json` — copy of `configs/pilot_M2.json`
- `tables/pilot_M3.json` — copy of `configs/pilot_M3.json`
- `tables/variants-eval.json` — copy of `configs/variants-eval.json`

## Run records (small files copied from runs/)

- `run_records/pilot_M1/20261123-152837/` ← `runs/pilot_M1/20261123-152837` (config.json, metrics.json, notes.md, train_log.csv, val_log.csv, wall_time.json, git_commit.txt); large files left in place: runs/pilot_M1/20261123-152837/checkpoints/
- `run_records/pilot_M2/20261123-152844/` ← `runs/pilot_M2/20261123-152844` (config.json, metrics.json, notes.md, train_log.csv, val_log.csv, wall_time.json, git_commit.txt); large files left in place: runs/pilot_M2/20261123-152844/checkpoints/
- `run_records/pilot_M3/20261123-153159/` ← `runs/pilot_M3/20261123-153159` (config.json, metrics.json, notes.md, train_log.csv, val_log.csv, wall_time.json, git_commit.txt); large files left in place: runs/pilot_M3/20261123-153159/checkpoints/
- `run_records/variants-eval-v1/20261123-223749/` ← `runs/variants-eval-v1/20261123-223749` (config.json, metrics.json, metrics.csv, notes.md, vs_val_log.csv, git_commit.txt); large files left in place: runs/variants-eval-v1/20261123-223749/outputs/
