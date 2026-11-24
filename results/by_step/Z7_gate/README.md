# Z7 (Exp 6c) — Gate implementation checks

**Owner:** Zijian (guide Z7)

**What:** Unit tests (shapes, gate = 0 where warp invalid, init 0.5, M2/M3 identical parameters, M2 ignores cue values, finite outputs) and M3 overfit check with gate maps. Verifies the gate trains; not evidence that it helps.

**Status (from results/RESULTS.md):** done

**Key numbers:** tests: ======================== 15 passed in 155.59s (0:02:35) ========================; M3 overfit PSNR-mu 43.75 (C0 36.96); params M3 1668357

**Notes:** gate implemented and verified to train on a fixed batch

## Figures

- `figures/fig_overfit_M3_curve.png` — overfit check (M3), 3000 updates on one fixed batch of 4 crops; not a generalization result Crops: scene 001 y=725 x=417 256x256; scene 059 y=266 x=566 256x256; scene 061 y=554 x=425 256x256; scene 072 y=509 x=485 256x256. Loss logged every 10 updates; PSNR-mu (mean over the 4 crops) every 250 updates; C0 = H_2 on the same crops. Source run: runs/overfit_M3/20261123-151540; numbers from metrics.json, train_log.csv, psnr_log.csv.

- `figures/fig_gate_maps.png` — Gate maps of M3 at the end of the overfit check (3000 updates, fixed batch of 4 crops); status: gate implemented and verified to train on a fixed batch. Crops shown: crop 0 scene 001 y=725 x=417; crop 1 scene 059 y=266 x=566. Panels: L_2 (LDR shown as stored (normalised by dtype max, x255, rounded)); g_1, g_3 (gate in [0,1]); v_i validity; r_i = exp(-e/2) FB cue; q_ref exposure cue; viridis 0..1. Cues are estimated (DIS flow, FB check), not ground-truth motion. Source run: runs/overfit_M3/20261123-151540

## Tables and configs

- `tables/pytest.txt` — ======================== 15 passed in 155.59s (0:02:35) ========================
- `tables/overfit_M3.json` — copy of `configs/overfit_M3.json`

## Run records (small files copied from runs/)

- `run_records/overfit_M3/20261123-151540/` ← `runs/overfit_M3/20261123-151540` (config.json, metrics.json, notes.md, train_log.csv, psnr_log.csv, git_commit.txt); large files left in place: runs/overfit_M3/20261123-151540/checkpoints/
