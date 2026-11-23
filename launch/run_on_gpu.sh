#!/bin/bash
# usage: launch/run_on_gpu.sh <role> <log_file> <hours> <command...>
#
# Slurm adaptation of EXPERIMENT_PLAN.md section 12. The plan's machine had local GPUs
# 0-7 (4 and 5 forbidden); on MSI every run gets exactly one GPU from Slurm instead, so
# the plan's GPU ids map to *roles* (one Slurm job each):
#   gpu0 -> exp1 / overfit (short jobs)   gpu1 -> M0 pilot   gpu2 -> M1 pilot
#   gpu3 -> M2 pilot   gpu6 -> M3 pilot   gpu7 -> exp1 check / eval / reruns
# Roles gpu4 / gpu5 are refused, mirroring the plan's rule. Inside Python the device is
# cuda:0; hdr.runlog records SLURM_JOB_ID, node and nvidia-smi (uuid, bus id) in config.json.
# Env overrides: PARTITION (default a100-4), ACCOUNT (default ee5561), GRES (gpu:a100:1).
set -euo pipefail
ALLOWED="gpu0 gpu1 gpu2 gpu3 gpu6 gpu7"
ROLE="$1"; LOG="$2"; HOURS="$3"; shift 3
if ! [[ " $ALLOWED " == *" $ROLE "* ]]; then
  echo "role $ROLE is not allowed (allowed: $ALLOWED)" >&2; exit 1
fi
PARTITION="${PARTITION:-a100-4}"
ACCOUNT="${ACCOUNT:-ee5561}"
GRES="${GRES:-gpu:a100:1}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$(dirname "$LOG")"
CMD="$(printf '%q ' "$@")"
sbatch ${DEPEND:+--dependency=$DEPEND} -A "$ACCOUNT" -p "$PARTITION" -N 1 -n 1 -c 8 --mem=96G --gres="$GRES" \
  -t "$(printf '%02d:00:00' "$HOURS")" -J "hdr-$ROLE" -o "$LOG" --open-mode=append \
  --wrap "cd $REPO && echo \"[\$(date)] role=$ROLE job=\$SLURM_JOB_ID node=\$(hostname) CVD=\$CUDA_VISIBLE_DEVICES\" && nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu --format=csv && export HDR_ROLE=$ROLE && $CMD"
