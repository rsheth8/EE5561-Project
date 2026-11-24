# EE5561-Project
EE 5561 Team HDR (Rahil Sheth, Zijian Zhang): ghost-free multi-exposure HDR on the Kalantari SIGGRAPH17 dataset — intermediate-report experiments Exp 1–6.

## Where things are

- `results/by_step/` — results grouped by the team guide's step labels (R = Rahil, Z = Zijian, S/B = shared). Start with `results/by_step/README.md`.
- `results/RESULTS.md` — overall summary; every number traces to a saved file.
- `release/` — the 6 checkpoints behind the reported numbers and M0–M3 validation output images.
- `hdr/` package, `scripts/`, `tests/`, `configs/` (frozen split: `configs/split.json`), `launch/run_on_gpu.sh` (Slurm launcher).
- `runs/` — per-run configs, logs and metrics (checkpoints and full-resolution outputs stay on MSI).
- `notes/problems.md` — timestamped problems log.

Not in the repo: the dataset (download from https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/, sha256 in `notes/download_sha256.txt`), caches, `.venv` (see `requirements.txt`).

Status: Rahil's steps (R1–R4) are provisional versions made so Zijian's steps could run; Rahil may replace them (see the notes in `results/by_step/R*/README.md`).
