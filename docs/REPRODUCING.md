# Verification and reproduction

## Start with saved records

Run from the repository root:

```bash
uv run scripts/verify_handoff.py
```

This standard-library command needs no GPU, dataset, credentials or model loading. It checks artifact SHA-256 values, the preserved evaluator's code manifest, frozen split/protocol hashes, checkpoint bytes against the four frozen hashes, all 120 unique test score records, 24 aggregate groups, the 20 saved validation score checks, report assets and local links in the new documentation. It prints descriptive results and exits nonzero if a check fails. It does not independently rescore test float arrays. After an intentional report revision, update its PDF, small Overleaf ZIP and corresponding report entries in the artifact manifest together; do not rehash experiment evidence to bypass a failure. The original transfer did not include checkpoint bytes; the copies under `release/checkpoints/` were already in this repository and can now be compared separately.

## Environments are part of the result

| Work | Recorded runtime |
| --- | --- |
| Original training/validation, MSI Slurm A100 | Python 3.11.13, torch 2.7.1+cu126, OpenCV 5.1.0 |
| Rahil regional analysis / frozen test, MSI OnDemand A100 | Python 3.11.13, torch 2.7.1+cu118, OpenCV 4.10.0, NumPy 1.26.4, scikit-image 0.24.0 |
| Local code checks | CPU smoke checks only; not reproduction of the MSI metric environment |

The root `requirements.txt` is the original Linux/CUDA environment snapshot. It includes Linux NVIDIA packages and is not a portable macOS installation recipe. Preserve it as provenance. Do not change the recorded versions or claim that installing a different local stack reproduces the trained run. Compare methods within the same evaluation partition; validation-to-test changes include runtime differences as well as different scenes.

For a lightweight CPU check on a local machine with uv and Python 3.11, the existing suite can run with portable packages:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --python 3.11 \
  --with 'torch==2.7.1' --with 'numpy==1.26.4' \
  --with 'opencv-python-headless==4.10.0.84' --with 'scikit-image==0.24.0' \
  --with matplotlib --with pandas --with pytest \
  python -m pytest -q
```

Raw/cache-dependent tests skip when the data is absent. That is not evidence that they passed. Full data/GPU checks belong on MSI in the established environment.

## Known MSI locations

These were recorded during the completed work. Verify access and existence before a rerun; the paths are not a claim of a current live MSI session.

- Rahil workspace: `/projects/standard/ee5561/shared/HDR/sheth047`
- Rahil training inputs: `/projects/standard/ee5561/shared/HDR/sheth047/raw/Training`
- Rahil prepared test inputs: `/projects/standard/ee5561/shared/HDR/sheth047/runs/test-inputs-v1`
- Rahil completed test run: `/projects/standard/ee5561/shared/HDR/sheth047/runs/frozen-test-v1` (float arrays under `predictions/`)
- Rahil regional run: `/projects/standard/ee5561/shared/HDR/sheth047/runs/rahil-analysis/validation-review-v2`
- Zijian shared project storage: `/projects/standard/ee5561/shared/zha00175_hdr`
- Validation float outputs: `/projects/standard/ee5561/shared/zha00175_hdr/outputs/variants-eval-v1/20261123-223749`
- Pilot checkpoints: `/projects/standard/ee5561/shared/zha00175_hdr/checkpoints/pilot_M*/<run>/step_050000.pt`; exact run names/hashes are in the frozen protocol.

See [released checkpoint instructions](../release/README.md) for the original validation evaluation. Set up raw data/cache and checkpoint placement first. That procedure is not needed to review the report.

## Frozen test rerun, only when deliberately requested

The completed evaluator is in [tools/frozen-test](../tools/frozen-test/README.md). Its helper manifest and checkpoint hashes must remain unchanged. Use an allocated GPU terminal, the recorded Rahil environment and a fresh output directory. For example, with the paths above verified:

```bash
uv run --no-sync --python /projects/standard/ee5561/shared/HDR/sheth047/.venv/bin/python \
  tools/frozen-test/evaluate_test.py \
  --input-manifest /projects/standard/ee5561/shared/HDR/sheth047/runs/test-inputs-v1/input_manifest.json \
  --checkpoints /projects/standard/ee5561/shared/zha00175_hdr/checkpoints \
  --out /projects/standard/ee5561/shared/HDR/sheth047/runs/frozen-test-reproduction-NEW
```

The evaluator validates its code, protocol, full 15-entry list, input hashes, overlap status and checkpoint hashes. It refuses reused output directories. It saves float predictions and figures separately; PNGs must never be substituted for metric arrays. There is no need to rerun this merely to obtain already completed test results.

The original `READ_THIS_FIRST.txt` contains preparation/download commands and predates successful completion. Preserve it as a record, but use [current results](../results/TEST_RESULTS.md) and [the handoff](HANDOFF.md) for task status. No training/checkpoint/protocol changes are part of this handoff.
