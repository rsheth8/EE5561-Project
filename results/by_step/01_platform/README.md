# Exp 1 — Platform check

**Owner:** Together (shared setup, guide B1)

**What:** Synthetic GPU check on the MSI A100s: forward/backward/update of M0, throughput (it/s), peak memory, full-resolution forward time. No HDR data involved; not an HDR result.

**Status (from results/RESULTS.md):** done

**Key numbers:** NVIDIA A100-SXM4-40GB (aga23, job 7911791): 2.354 it/s, peak 12.81 GB, full-res fwd 0.272 s / 5.94 GB; NVIDIA A100-SXM4-40GB (aga21, job 7911790): 2.352 it/s, peak 12.81 GB, full-res fwd 0.272 s / 5.94 GB

**Notes:** synthetic env check; M0 params 1441283; TF32 off

## Tables and configs

- `tables/platform.csv` — (no convention line)

## Run records (small files copied from runs/)

- `run_records/exp1/20261123-143444/` ← `runs/exp1/20261123-143444` (config.json, metrics.json, notes.md, nvidia_smi.txt, git_commit.txt)
- `run_records/exp1/20261123-143446/` ← `runs/exp1/20261123-143446` (config.json, metrics.json, notes.md, nvidia_smi.txt, git_commit.txt)
