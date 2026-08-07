#!/bin/bash
# Measure --rematerialize-param-from-master-weight on Qwen3-30B-A3B, 8xH200.
# H100/1-node branch => TP4 / EP8 / colocate + optimizer-cpu-offload +
# use-precision-aware-optimizer (the HDO path) + --use-kl-loss (the ref tag),
# so one run exercises both new code paths.
#   usage: exp_1572_run.sh <off|on> [extra miles args...]
set -uo pipefail

MODE="$1"; shift || true
EXP=/scratch/exp1572
export MILES_SCRIPT_DATA_DIR=$EXP/datasets
export MILES_SCRIPT_MODEL_DIR=$EXP/models
export MILES_SCRIPT_OUTPUT_DIR=$EXP/out_$MODE
export MILES_SCRIPT_MEGATRON_PATH=/mirror/nemotron/megatron
mkdir -p "$MILES_SCRIPT_DATA_DIR" "$MILES_SCRIPT_MODEL_DIR" "$MILES_SCRIPT_OUTPUT_DIR" $EXP/logs

FLAGS="--num-rollout 3"
if [ "$MODE" = "on" ]; then
  FLAGS="$FLAGS --rematerialize-param-from-master-weight"
fi
FLAGS="$FLAGS $*"

LOG=$EXP/logs/qwen30b_$MODE.log
cd /mirror/miles-pr-1572
echo "=== $MODE | flags: $FLAGS | $(date -u +%FT%TZ) ===" | tee "$LOG"
python scripts/run_qwen3_30b_a3b.py \
    --num-gpus-per-node 8 \
    --hardware H100 \
    --no-enable-eval \
    --extra-args "$FLAGS" 2>&1 | tee -a "$LOG"
echo "=== exit=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a "$LOG"
