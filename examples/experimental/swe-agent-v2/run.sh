#!/bin/bash
# Agent V2 launcher: Miles <-> Harbor agent orchestration.
#
# Supports any task type (SWE-bench, Terminal-Bench, custom) via Harbor.
#
# Usage:
#   bash run.sh                                    # full training (default)
#   bash run.sh --mode debug                       # debug with smaller batch/rollout
#   bash run.sh --mode debug --num-gpus 2
#   bash run.sh --hf-checkpoint Qwen/Qwen3-4B --model-script qwen3-4B.sh
#
# Tested on 8x H200 (miles-maocheng container):
#   bash run.sh --mode debug --num-gpus 8 --ep 8 \
#     --prompt-data /root/mixed.jsonl \
#     --sglang-tool-call-parser glm47 \
#     --sglang-reasoning-parser glm45 \
#     --no-wait
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MILES_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ── Defaults ────────────────────────────────────────────────────────
MODE="train"
NUM_GPUS=4
TP=1; PP=1; EP=4

HF_CHECKPOINT="zai-org/GLM-4.7-Flash"
MODEL_SCRIPT="glm4.7-flash.sh"
REF_LOAD="/root/GLM-4.7-Flash_torch_dist"
LOAD_DIR=""
SAVE_DIR="/root/GLM-4.7-Flash_agent_v2/"
SAVE_INTERVAL=100

PROMPT_DATA="/root/swe_train.jsonl"
NUM_ROLLOUT=3000
ROLLOUT_BATCH_SIZE=8
N_SAMPLES=8
ROLLOUT_TEMP=0.8
MAX_RESP_LEN=8192
GLOBAL_BATCH=64
MAX_TOKENS_PER_GPU=2048

LR=1e-6
KL_LOSS_COEF=0.01
EPS_CLIP=0.2
EPS_CLIP_HIGH=0.28

AGENT_SERVER_URL="${AGENT_SERVER_URL:-${SWE_AGENT_URL:-http://agent_env:11000}}"
HARBOR_TASKS_DIR="${HARBOR_TASKS_DIR:-/root/harbor_tasks}"
ROUTER_EXTERNAL_HOST="${MILES_ROUTER_EXTERNAL_HOST:-$(hostname)}"
SGLANG_ROUTER_PORT=30000
SGLANG_TOOL_CALL_PARSER=""
SGLANG_REASONING_PARSER=""
NO_WAIT=""

# ── Parse arguments ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)               MODE="$2";               shift 2 ;;
    --num-gpus)           NUM_GPUS="$2";            shift 2 ;;
    --tp)                 TP="$2";                  shift 2 ;;
    --pp)                 PP="$2";                  shift 2 ;;
    --ep)                 EP="$2";                  shift 2 ;;
    --hf-checkpoint)      HF_CHECKPOINT="$2";       shift 2 ;;
    --model-script)       MODEL_SCRIPT="$2";        shift 2 ;;
    --ref-load)           REF_LOAD="$2";            shift 2 ;;
    --load-dir)           LOAD_DIR="$2";            shift 2 ;;
    --save-dir)           SAVE_DIR="$2";            shift 2 ;;
    --save-interval)      SAVE_INTERVAL="$2";       shift 2 ;;
    --prompt-data)        PROMPT_DATA="$2";         shift 2 ;;
    --num-rollout)        NUM_ROLLOUT="$2";         shift 2 ;;
    --rollout-batch-size) ROLLOUT_BATCH_SIZE="$2";  shift 2 ;;
    --n-samples-per-prompt) N_SAMPLES="$2";         shift 2 ;;
    --rollout-temperature)  ROLLOUT_TEMP="$2";      shift 2 ;;
    --rollout-max-response-len) MAX_RESP_LEN="$2";  shift 2 ;;
    --global-batch-size)  GLOBAL_BATCH="$2";        shift 2 ;;
    --max-tokens-per-gpu) MAX_TOKENS_PER_GPU="$2";  shift 2 ;;
    --lr)                 LR="$2";                  shift 2 ;;
    --kl-loss-coef)       KL_LOSS_COEF="$2";        shift 2 ;;
    --eps-clip)           EPS_CLIP="$2";            shift 2 ;;
    --eps-clip-high)      EPS_CLIP_HIGH="$2";       shift 2 ;;
    --agent-server-url)   AGENT_SERVER_URL="$2";    shift 2 ;;
    --harbor-tasks-dir)   HARBOR_TASKS_DIR="$2";    shift 2 ;;
    --router-external-host) ROUTER_EXTERNAL_HOST="$2"; shift 2 ;;
    --sglang-router-port) SGLANG_ROUTER_PORT="$2";  shift 2 ;;
    --sglang-tool-call-parser)  SGLANG_TOOL_CALL_PARSER="$2"; shift 2 ;;
    --sglang-reasoning-parser)  SGLANG_REASONING_PARSER="$2"; shift 2 ;;
    --no-wait)            NO_WAIT="--no-wait";      shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── Debug mode overrides ────────────────────────────────────────────
if [[ "$MODE" == "debug" ]]; then
  NUM_ROLLOUT=50
  ROLLOUT_BATCH_SIZE=4
  N_SAMPLES=4
  MAX_RESP_LEN=4096
  GLOBAL_BATCH=16
  MAX_TOKENS_PER_GPU=1024
fi

# ── Source model architecture ────────────────────────────────────────
MODEL_SCRIPT_PATH="$MILES_ROOT/scripts/models/$MODEL_SCRIPT"
if [[ ! -f "$MODEL_SCRIPT_PATH" ]]; then
  echo "ERROR: Model script not found: $MODEL_SCRIPT_PATH"
  exit 1
fi
source "$MODEL_SCRIPT_PATH"

# ── Cleanup stale processes ──────────────────────────────────────────
echo "Cleaning up stale processes..."
pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 3

# ── Start Ray ────────────────────────────────────────────────────────
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
echo "Starting Ray head node..."
ray start --head \
  --node-ip-address "$MASTER_ADDR" \
  --num-gpus "$NUM_GPUS" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --port=8899

# ── Build runtime env ────────────────────────────────────────────────
RUNTIME_ENV=$(python3 -c "
import json, sys
print(json.dumps({'env_vars': {
    'PYTHONPATH': '/root/Megatron-LM/:${SCRIPT_DIR}:${MILES_ROOT}',
    'CUDA_DEVICE_MAX_CONNECTIONS': '1',
    'MILES_EXPERIMENTAL_ROLLOUT_REFACTOR': '1',
    'AGENT_SERVER_URL': '${AGENT_SERVER_URL}',
    'AGENT_MODEL_NAME': '${AGENT_MODEL_NAME:-model}',
    'MILES_ROUTER_EXTERNAL_HOST': '${ROUTER_EXTERNAL_HOST}',
    'HARBOR_TASKS_DIR': '${HARBOR_TASKS_DIR}',
    'MILES_HOST_IP': '${MILES_HOST_IP:-$(hostname)}',
    'NCCL_NVLS_ENABLE': '0',
    'DEPRECATED_MEGATRON_COMPATIBLE': '1',
}}))
")

# ── Build checkpoint args ────────────────────────────────────────────
CKPT_ARGS=(
  --hf-checkpoint "$HF_CHECKPOINT"
  --ref-load "$REF_LOAD"
  --save "$SAVE_DIR"
  --save-interval "$SAVE_INTERVAL"
)
if [[ -n "$LOAD_DIR" ]]; then
  CKPT_ARGS+=(--load "$LOAD_DIR")
fi

# ── Build SGLang args ────────────────────────────────────────────────
SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static 0.5
  --use-miles-router
  --sglang-router-port "$SGLANG_ROUTER_PORT"
)
if [[ -n "$SGLANG_TOOL_CALL_PARSER" ]]; then
  SGLANG_ARGS+=(--sglang-tool-call-parser "$SGLANG_TOOL_CALL_PARSER")
fi
if [[ -n "$SGLANG_REASONING_PARSER" ]]; then
  SGLANG_ARGS+=(--sglang-reasoning-parser "$SGLANG_REASONING_PARSER")
fi

# ── W&B args ─────────────────────────────────────────────────────────
WANDB_ARGS=()
if [[ -n "${WANDB_KEY:-}" && "$MODE" != "debug" ]]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT:-miles-agent-v2}"
    --wandb-group "${WANDB_GROUP:-agent-v2}"
    --wandb-key "$WANDB_KEY"
  )
  if [[ -n "${WANDB_TEAM:-}" ]]; then
    WANDB_ARGS+=(--wandb-team "$WANDB_TEAM")
  fi
fi

# ── Submit ───────────────────────────────────────────────────────────
echo "Launching ${MODE} training..."
echo "  Agent server:  $AGENT_SERVER_URL"
echo "  Harbor tasks:  $HARBOR_TASKS_DIR"
echo "  Model:         $HF_CHECKPOINT"

ray job submit \
  --address="http://127.0.0.1:8265" \
  --runtime-env-json="$RUNTIME_ENV" \
  $NO_WAIT \
  -- python3 "$MILES_ROOT/train.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "$NUM_GPUS" \
  --colocate \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  --tensor-model-parallel-size "$TP" \
  --pipeline-model-parallel-size "$PP" \
  --context-parallel-size 1 \
  --expert-model-parallel-size "$EP" \
  --expert-tensor-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU" \
  --prompt-data "$PROMPT_DATA" \
  --input-key prompt \
  --metadata-key metadata \
  --rollout-shuffle \
  --num-rollout "$NUM_ROLLOUT" \
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
  --n-samples-per-prompt "$N_SAMPLES" \
  --rollout-temperature "$ROLLOUT_TEMP" \
  --rollout-max-response-len "$MAX_RESP_LEN" \
  --global-batch-size "$GLOBAL_BATCH" \
  --balance-data \
  --advantage-estimator grpo \
  --use-kl-loss \
  --kl-loss-coef "$KL_LOSS_COEF" \
  --kl-loss-type low_var_kl \
  --entropy-coef 0.0 \
  --eps-clip "$EPS_CLIP" \
  --eps-clip-high "$EPS_CLIP_HIGH" \
  --optimizer adam \
  --lr "$LR" \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  "${SGLANG_ARGS[@]}" \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate \
  --custom-agent-function-path swe_agent_function.run \
  --custom-rm-path generate.reward_func \
  --rollout-function-path generate.RolloutFn \
  --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted \
  "${WANDB_ARGS[@]}" \
  --generate-multi-samples
  # TODO: Remove --generate-multi-samples once TITO (Token In Token Out) is
  # workable. Without TITO, multi-turn token merging (merge_samples) fails
  # because BPE re-tokenization across turns produces inconsistent token
  # sequences, causing "b.tokens must start with a.tokens" assertion errors.
