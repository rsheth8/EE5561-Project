# Exp 4 — Overfit check (M0)

**Owner:** Zijian (guide Z2, Z4)

**What:** M0 trained on one fixed batch of 4 crops for 3000 updates. Pipeline/debugging check (pass = PSNR-mu at least 5 dB above C0 on the same crops); not a generalization result.

**Status (from results/RESULTS.md):** done (pass)

**Key numbers:** final PSNR-mu 43.45 vs C0 36.96 dB (+6.49); loss 0.3391 -> 0.0047

**Notes:** pipeline check, 4 fixed crops

## Figures

- `figures/fig_overfit_curve.png` — overfit check (M0), 3000 updates on one fixed batch of 4 crops; not a generalization result Crops: scene 001 y=725 x=417 256x256; scene 059 y=266 x=566 256x256; scene 061 y=554 x=425 256x256; scene 072 y=509 x=485 256x256. Loss logged every 10 updates; PSNR-mu (mean over the 4 crops) every 250 updates; C0 = H_2 on the same crops. Source run: runs/overfit_M0/20261123-145303; numbers from metrics.json, train_log.csv, psnr_log.csv.

- `figures/fig_overfit_before_after.png` — overfit check (M0), 3000 updates on one fixed batch of 4 crops; not a generalization result Crops shown: crop 0 scene 001 y=725 x=417; crop 1 scene 059 y=266 x=566. Inputs: LDR shown as stored (normalised by dtype max, x255, rounded). Outputs and GT: to_uint8(H) = round(255*T(clip(H,0,1))), T = mu-law, mu=5000. Per-crop final PSNR-mu [40.6, 42.96, 45.38, 44.88], C0 [32.42, 47.01, 38.96, 29.47]. Source run: runs/overfit_M0/20261123-145303

## Tables and configs

- `tables/overfit_M0.json` — copy of `configs/overfit_M0.json`
- `tables/overfit_crops.json` — copy of `configs/overfit_crops.json`

## Run records (small files copied from runs/)

- `run_records/overfit_M0/20261123-145303/` ← `runs/overfit_M0/20261123-145303` (config.json, metrics.json, notes.md, train_log.csv, psnr_log.csv, git_commit.txt); large files left in place: runs/overfit_M0/20261123-145303/checkpoints/
