# Preserved frozen test helper

This package already ran successfully on MSI. Its original `READ_THIS_FIRST.txt` describes the pre-evaluation workflow and is preserved as historical provenance, not current task status. See [the handoff](../../docs/HANDOFF.md) and [completed records](../../results/rahil/frozen-test/).

All files covered by `helper_sha256.json` are unchanged from Rahil's evaluated package. `frozen_hdr/` is an intentional snapshot of the relevant implementation, not a new training code path. The evaluator refuses code/checkpoint hash mismatches, incomplete test lists, unresolved overlap flags and reused output directories. It uses eval/inference mode without constructing an optimizer.

Read [reproduction notes](../../docs/REPRODUCING.md) before any rerun. No rerun is required to use existing completed results. Dataset and full float predictions are not included here.
