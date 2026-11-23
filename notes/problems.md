# Problems encountered and what was tried (timestamps CST)

- 2026-11-23 14:02 — Training-archive download (curl) dropped at 1.89 GB ("transfer closed with 215433041 bytes remaining"); resumed with `curl -C -`; `unzip -t` then passed; sha256 in notes/download_sha256.txt.
- 2026-11-23 14:25 — Plan assumes a local 8-GPU box; this is MSI (Slurm, no GPU on login node). User chose: one Slurm job per GPU role on public `a100-4` (account ee5561), mhong fallback. launch/run_on_gpu.sh maps the plan's GPU ids to roles (gpu0..gpu7, gpu4/gpu5 refused); physical GPU uuid/node/job id recorded in each run's config.json.
- 2026-11-23 14:28 — First Exp 2 job on `agsmall` estimated to start Dec 1; cancelled and resubmitted on `msilarge` (started immediately).
- 2026-11-23 14:33 — Exp 2: scene 009 had best_k = 3 by median relative error (0.0398 vs 0.0464 for k=2). Investigated before continuing: 009 shares bit-identical L_2 and H_gt with 010 (k=2 clearly best there), log-gradient correlation k2 0.981 vs k1 0.846 / k3 0.767 -> reference k=2 kept (results/tables/scene009_reference_check.json). No constant changed.
- 2026-11-23 14:40 — Login node killed a CPU smoke test of training (memory cgroup); tested on GPU instead.
- 2026-11-23 14:45 — First overfit job failed after 4 s with no log: home directory quota exceeded (191 GB). Moved downloads/, raw/, cache/ (16 GB) to /projects/standard/ee5561/shared/zha00175_hdr and symlinked; checkpoints/ and outputs/ of runs are symlinks into the same project space. Archive and extracted files moved unchanged (mv).
- 2026-11-23 14:52 — tests/test_data.py cache-vs-raw feature equality failed by 6e-8: torch.pow on a non-contiguous (permuted) tensor rounds differently from the contiguous path. Test now compares in the same memory layout (bit-identical) and checks the permuted version within 1e-6.
- 2026-11-23 14:55 — fp32 means TF32 disabled for matmul and cuDNN in every run (hdr/torchutil.py); Exp 1 throughput is measured with TF32 off.
