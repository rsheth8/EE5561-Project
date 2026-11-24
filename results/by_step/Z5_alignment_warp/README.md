# Z5 — Alignment module (DIS flow + backward warp)

**Owner:** Zijian (guide Z5)

**What:** Warp convention and checks: zero-flow identity, known translation (direction/sign), forward-backward consistency; flow cache for all scenes. Used by C3, M1–M3 and Exp 6. Flow method is OpenCV DIS (guide suggested e.g. Farneback).

## Figures

- `figures/fig_warp_check.png` — Synthetic textured image (seeded random texture), 240x320, shifted by (+5, -3) px. Not dataset data. Method: DIS optical flow (PRESET_MEDIUM) ref->src, backward warp cv2.remap INTER_LINEAR, BORDER_CONSTANT 0. Median DIS flow in interior: (5.0011, -3.0007); mean |warp-ref| interior 0.00097. Validity v = 1 where the sampling map lies inside the source; v=0 strip: right 5 columns, top 3 rows. Display: images shown as stored values x255; error map shown with a 0..0.2 gray scale. Produced by tests/test_warp.py.

## Tables and configs

- `tables/warp_check.json` — JSON; see file

## Run records (small files copied from runs/)

- `run_records/flow/20261123-143742/` ← `runs/flow/20261123-143742` (config.json, metrics.json, metrics.csv, notes.md, git_commit.txt)
