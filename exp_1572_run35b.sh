#!/bin/bash
# Qwen3.5-35B-A3B (MTP, spec v2), 8xH200: TP2 / CP2 / EP8 / colocate +
# optimizer-cpu-offload + use-precision-aware-optimizer (HDO path) +
# --use-kl-loss (ref tag).  usage: exp_1572_run35b.sh <off|on|check>
set -uo pipefail

MODE="$1"; shift || true
EXP=/scratch/exp1572
export MILES_SCRIPT_DATA_DIR=$EXP/datasets
export MILES_SCRIPT_MODEL_DIR=$EXP/models
export MILES_SCRIPT_OUTPUT_DIR=$EXP/out35b_$MODE
export MILES_SCRIPT_MEGATRON_PATH=/mirror/Megatron-LM
mkdir -p "$MILES_SCRIPT_DATA_DIR" "$MILES_SCRIPT_MODEL_DIR" "$MILES_SCRIPT_OUTPUT_DIR" $EXP/logs

FLAGS="--num-rollout 3"
case "$MODE" in
  on)    FLAGS="$FLAGS --rematerialize-param-from-master-weight" ;;
  check) FLAGS="$FLAGS --rematerialize-param-from-master-weight --check-rematerialize-param-from-master-weight" ;;
esac
FLAGS="$FLAGS $*"

LOG=$EXP/logs/qwen35b_$MODE.log
cd /mirror/miles-pr-1572
echo "=== 35b $MODE | flags: $FLAGS | $(date -u +%FT%TZ) ===" | tee "$LOG"
python scripts/run_qwen3_5_35b_a3b_mtp_cp2_ep8.py \
    --num-gpus-per-node 8 \
    --hardware H200 \
    --extra-args "$FLAGS" 2>&1 | tee -a "$LOG"
echo "=== exit=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a "$LOG"
