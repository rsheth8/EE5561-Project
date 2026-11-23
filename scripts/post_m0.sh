#!/bin/bash
# Exp 5 post-processing: evaluate val-selected M0 twice (fresh processes), repro check, report.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY -m hdr.evaluate --config configs/pilot-eval.json --out runs/pilot-eval-v1
sleep 2
$PY -m hdr.evaluate --config configs/pilot-eval.json --out runs/pilot-eval-v1
A=$(ls -d runs/pilot-eval-v1/*/ | sort | tail -2 | head -1); B=$(ls -d runs/pilot-eval-v1/*/ | sort | tail -1)
$PY scripts/repro_check.py "$A" "$B" --name M0
P=$(ls -d runs/pilot_M0/*/ | sort | tail -1); C=$(ls -d runs/classical-v1/*/ | sort | tail -1)
$PY scripts/pilot_report.py --pilot "$P" --eval "$A" --classical "$C"
$PY scripts/make_results.py
