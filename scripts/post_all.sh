#!/bin/bash
# 6d post-processing: matched-budget selection, evaluation of M0-M3, tables, RESULTS.md.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY scripts/matched_budget.py select
$PY -m hdr.evaluate --config configs/variants-eval.json --out runs/variants-eval-v1
E=$(ls -d runs/variants-eval-v1/*/ | sort | tail -1)
$PY scripts/matched_budget.py table "$E"
$PY scripts/make_results.py
