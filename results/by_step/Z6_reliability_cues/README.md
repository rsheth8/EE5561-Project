# Z6 (Exp 6a/6b) — Reliability cue maps and whether cues predict alignment error

**Owner:** Zijian (guide Z6)

**What:** Estimated cues (warp validity, forward-backward consistency, exposure cues q_ref / q_i^w) visualized on 2 val scenes, with a marked cue-failure example; statistics of warped-frame error vs FB consistency (AUROC), split by whether the reference is clipped.

**Status (from results/RESULTS.md):** done

**Key numbers:** L1/all: err cons 0.0095 vs incons 0.0110, AUROC 0.613 (n=5253541); L1/ref_clipped: err cons 0.0064 vs incons 0.0061, AUROC 0.441 (n=659941); L1/ref_unclipped: err cons 0.0098 vs incons 0.0127, AUROC 0.642 (n=4593600); L3/all: err cons 0.0100 vs incons 0.0308, AUROC 0.731 (n=4728323); L3/ref_clipped: err cons 0.2489 vs incons 0.2946, AUROC 0.878 (n=543); L3/ref_unclipped: err cons 0.0100 vs incons 0.0306, AUROC 0.730 (n=4727780)

**Notes:** estimated cues; see fig_reliability_maps.txt for a cue failure

## Figures

- `figures/fig_reliability_maps.png` — Validation scenes 062 (ghost scene of Exp 3) and 025 (most saturated val scene other than it); full frames. One row per non-reference frame. These are ESTIMATED reliability cues (DIS flow + forward-backward check, exposure cues), not ground-truth motion. Panels: exposure-matched pair as a red/cyan composite (8-bit gray, used only for flow); warped L_i^w; validity v_i; FB-inconsistency mask; q_ref; q_i^w; mu-law error of the warped frame vs H_gt. Cue failure example (consistent but wrong): scene 062, L_1: lime box y=370 x=544 96x96 holds the densest 'consistent but wrong' pixels (FB-consistent, valid, m(L_2)>0.95, err>0.05): 0.3% of the box; 68 such pixels in the frame | scene 062, L_3: lime box y=577 x=644 96x96 holds the densest 'consistent but wrong' pixels (FB-consistent, valid, m(L_2)>0.95, err>0.05): 80.9% of the box; 32555 such pixels in the frame | scene 025, L_1: lime box y=298 x=330 96x96 holds the densest 'consistent but wrong' pixels (FB-consistent, valid, m(L_2)>0.95, err>0.05): 4.0% of the box; 1143 such pixels in the frame | scene 025, L_3: lime box y=281 x=1043 96x96 holds the densest 'consistent but wrong' pixels (FB-consistent, valid, m(L_2)>0.95, err>0.05): 100.0% of the box; 87964 such pixels in the frame Display: LDR panels LDR shown as stored (normalised by dtype max, x255, rounded); masks/cues gray 0..1; error inferno 0..0.3. Not a learned result. Source run: runs/cues-v1/20261123-150109

## Tables and configs

- `tables/cue_diagnostics_val.csv` — 5 validation scenes (025, 045, 051, 056, 062); pixels with v_i=1 and q_i^w>0.2; err = mean_c |T(H_i^w)-T(H_gt)| (mu=5000); large error: err>0.05; consistent = FB check passed; AUROC of (1-r_i) for large error; strata by m(L_2)>0.95; source run runs/cues-v1/20261123-150109

## Run records (small files copied from runs/)

- `run_records/cues-v1/20261123-150109/` ← `runs/cues-v1/20261123-150109` (config.json, metrics.csv, notes.md, git_commit.txt, figures/fig_reliability_maps.png)
