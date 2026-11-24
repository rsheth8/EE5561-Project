# Exp 3 — Classical merges C0–C3

**Owner:** Rahil (guide R4)

**What:** C0 reference only; C1 hat-weighted merge without alignment; C2 reference-fallback deghosting; C3 = C2 on flow-aligned frames. Metric table on the 5 val scenes, ghost comparison, error maps, good case vs classified failure.

**Status (from results/RESULTS.md):** done

**Key numbers:** PSNR-mu / PSNR-L / SSIM-mu: C0: 36.59 / 32.72 / 0.9777; C1: 22.51 / 29.51 / 0.7613; C2: 26.84 / 31.87 / 0.9003; C3: 39.21 / 42.07 / 0.9800

**Notes:** C1 ghosts (fig_classical_ladder); fallback % C3 0.104

## Figures

- `figures/fig_classical_ladder.png` — Validation scene 062 (largest mean|T(C1)-T(GT)| among the 5 val scenes: 025=0.0366, 045=0.0383, 051=0.0378, 056=0.0444, 062=0.0516). Top row full frame with crop box; bottom row crop y=640, x=557, size 300x300 (window with largest mean C1 error). Methods: C0 reference only; C1 hat-weighted merge (no alignment); C2 reference-fallback deghosting; C3 C2 on DIS-aligned frames. Not learned; no training. Display: to_uint8(H) = round(255*T(clip(H,0,1))), T = mu-law, mu=5000, same mapping for every panel. PSNR-mu in titles from runs/classical-v1/20261123-145641/metrics.csv.

- `figures/fig_classical_error.png` — Validation scene 062; error = channel mean of |T(H) - T(H_gt)|, T = mu-law (mu=5000) on clip(H,0,1). Shared colour scale 0..0.3 (inferno) for all panels; crop y=640, x=557, 300x300. Methods C0-C3 as in fig_classical_ladder; classical, no training.

- `figures/fig_classical_cases.png` — Good case: val scene 062 (highest C3 PSNR-mu 41.466 dB), crop y=258, x=896, 300x300 (its largest-C3-error window). Scene 062 C3 residual error is small and concentrated on thin foliage edges. Failure: val scene 025, crop y=359, x=367, 300x300 = the window with the largest mean |T(C3)-T(GT)| over all 5 val scenes (0.0162). Failure classification: ghost (left: building pattern from a misaligned non-reference frame) + alignment artifact (moving hand: colour noise). Left region: ~50% of pixels have m(L_2)>0.95, so the reference cannot veto the samples (C2/C3 rule) and 82-98% of them are FB-inconsistent; hand: reference unclipped (mean m(L_2)=0.36) but 15% (L_1) / 30% (L_3) FB-inconsistent warped samples pass the 0.25 disagreement test. Display: to_uint8(H) = round(255*T(clip(H,0,1))), T = mu-law, mu=5000; error maps shared scale 0..0.3. Classical methods, no training.

## Tables and configs

- `tables/classical_val.csv` — 5 validation scenes (025, 045, 051, 056, 062), full image, no border excluded, mu = 5000; PSNR-L/SSIM-L on clip(H,0,1), PSNR-mu/SSIM-mu on mu-law; source run runs/classical-v1/20261123-145641
- `tables/ghost_crop.json` — JSON; see file
- `tables/classical.json` — copy of `configs/classical.json`

## Run records (small files copied from runs/)

- `run_records/classical-v1/20261123-145641/` ← `runs/classical-v1/20261123-145641` (config.json, metrics.json, metrics.csv, notes.md, git_commit.txt); large files left in place: runs/classical-v1/20261123-145641/outputs/
