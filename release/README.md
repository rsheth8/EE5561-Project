# Released checkpoints and validation output images

Only the checkpoints needed to reproduce the reported numbers are here (about 110 MB). All other
checkpoints (every 2000 updates), the full-resolution outputs (.npy/.hdr), the raw data and the
caches stay on MSI under `/projects/standard/ee5561/shared/zha00175_hdr/`. File hashes are in `SHA256SUMS`.

## Checkpoints (`checkpoints/<experiment>/<run>/`)

| File | Model | Guide step | Use |
|---|---|---|---|
| `overfit_M0/20261123-145303/final.pt` | M0 | Z4 (Exp 4) | overfit check, 3000 updates on 4 fixed crops (not a generalization result) |
| `overfit_M3/20261123-151540/final.pt` | M3 | Z7 (Exp 6c) | gate overfit check, 3000 updates on 4 fixed crops |
| `pilot_M0/20261123-150849/step_050000.pt` | M0 | Z3/Z4 (Exp 5), Z8 (Exp 6d) | val-selected checkpoint of the M0 pilot |
| `pilot_M1/20261123-152837/step_050000.pt` | M1 | Z8 (Exp 6d) | val-selected, matched budget N* = 50000 |
| `pilot_M2/20261123-152844/step_050000.pt` | M2 | Z8 (Exp 6d) | val-selected, matched budget N* = 50000 |
| `pilot_M3/20261123-153159/step_050000.pt` | M3 | Z8 (Exp 6d) | val-selected, matched budget N* = 50000 |

Each file holds the model and optimizer states, the update number, RNG states and the config
(saved by `hdr/train.py`). The run directory with the same name under `runs/` has its config,
training/validation logs and notes.

### Reproducing the validation numbers (guide I1)

`hdr/evaluate.py` looks for `runs/<experiment>/<run>/checkpoints/step_XXXXXX.pt`. After getting the
data (R1) and building the cache, put the checkpoints there and evaluate:

```bash
for d in release/checkpoints/pilot_M*/*/; do
  e=$(basename $(dirname $d)); r=$(basename $d)
  mkdir -p runs/$e/$r/checkpoints && cp $d/*.pt runs/$e/$r/checkpoints/
done
.venv/bin/python -m hdr.evaluate --config configs/variants-eval.json --out runs/variants-eval-check
```

Expected: the same numbers as `results/tables/variants_val.csv` (the reproduction check in
`results/tables/repro_check.json` found exact agreement and bit-identical outputs on an A100 with
cuDNN deterministic mode).

## Validation output images (`val_outputs_png/<model>/<scene>.png`)

Full-resolution outputs of M0–M3 (the checkpoints above) on the 5 validation scenes
(025, 045, 051, 056, 062), copied from `runs/variants-eval-v1/20261123-223749/outputs/`.
Display mapping: `round(255 * T(clip(H, 0, 1)))`, T = mu-law with mu = 5000. These are display
images only; metrics are computed on the float outputs.
