#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Async GRPO training on SWE-Gym via Polar rollouts + Miles (Megatron backend),
# model = NVIDIA Nemotron-3-Nano-30B-A3B (BF16, hybrid MoE/Mamba).
#
# This is the Miles-native port of ProRL-Agent-Server's
# examples/swegym_slime_grpo/run.sh. It launches the Miles train.py job and
# points its rollout function / reward / custom config at the Miles `polar_*`
# bridge modules:
#   rollout : miles.rollout.polar_rollout.generate_rollout_polar_async
#   reward  : miles.rollout.polar_reward.custom_rm
#   config  : miles.rollout.polar_config:resolve_polar_slime_config  (via --custom-config-path)
#
# The Polar rollout server + gateway are assumed to be ALREADY RUNNING (a
# prerequisite — see README.md). This script only renders the Polar custom
# config, starts Ray, and submits the Miles training job.
#
# GPU split (recommended, fixed):
#   4 actor/trainer GPUs (Megatron; the ref model is COLOCATED on the actor
#     ranks — Miles has no --ref-num-nodes flag, see README for rationale)
#   4 inference GPUs -> 2 SGLang engines x TP=2 (--rollout-num-gpus-per-engine 2)
#   Total 8 x {H100, H200, B200}.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# This example dir is <repo-root>/miles/examples/polar_swegym_grpo, i.e. three
# levels below the repo root where train.py and .venv live.
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"    # == the miles repo root
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/tmp/polar_swegym_grpo}"
mkdir -p "${RUN_DIR}" "${PROJECT_ROOT}/logs"

# Python: prefer the miles venv, fall back to any python3 on PATH.
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi
PYTHON_BIN_DIR="$(cd -- "$(dirname -- "${PYTHON_BIN}")" &>/dev/null && pwd)"
export PATH="${PYTHON_BIN_DIR}:${PATH}"

detect_host_ip() {
    "${PYTHON_BIN}" - <<'PY'
import socket
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect(("8.8.8.8", 80))
    print(sock.getsockname()[0])
    sock.close()
except Exception:
    try:
        print(socket.gethostbyname(socket.gethostname()))
    except Exception:
        print("127.0.0.1")
PY
}

# ── Model & checkpoint paths ────────────────────────────────────────────────
# Actor weights load straight from the HF checkpoint via Miles AutoBridge.
HF_CHECKPOINT="${HF_CHECKPOINT:-/data/siraj/hf_models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
# Reference model is --use-kl-loss's frozen copy, loaded from a Megatron
# torch_dist checkpoint (see README "Conversion"). There is NO --ref-num-nodes
# flag in Miles: the ref model always lives on the actor ranks
# (miles/ray/placement_group.py: with_ref = kl_coef != 0 or use_kl_loss).
REF_LOAD="${REF_LOAD:-${PROJECT_ROOT}/tmp/checkpoints/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16_torch_dist}"
RUN_ID="${RUN_ID:-polar-swegym-grpo-$(date -u +%Y%m%dT%H%M%SZ)}"
SAVE_ROOT="${SAVE_ROOT:-${PROJECT_ROOT}/tmp/ckpt/polar_swegym_grpo_nemotron30bnano}"
SAVE_DIR="${SAVE_DIR:-${SAVE_ROOT}/${RUN_ID}}"
mkdir -p "$SAVE_DIR"
if [ ! -d "$HF_CHECKPOINT" ]; then
    echo "ERROR: HF checkpoint not found at $HF_CHECKPOINT"
    exit 1
fi
if [ ! -d "$REF_LOAD" ] || [ ! -f "$REF_LOAD/latest_checkpointed_iteration.txt" ]; then
    echo "WARNING: ref-load checkpoint not found at $REF_LOAD"
    echo "  Convert the HF checkpoint to a Megatron torch_dist dir before --use-kl-loss."
fi

# shellcheck source=./model_args.sh
source "${SCRIPT_DIR}/model_args.sh"

# First run has an empty SAVE_DIR — leave --load empty so the actor initializes
# from --hf-checkpoint; once the first save lands --load points at SAVE_DIR.
if [ -f "$SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
    LOAD_DIR="$SAVE_DIR"
else
    LOAD_DIR=""
fi

# ── Data ────────────────────────────────────────────────────────────────────
# Required JSONL shape (docker variant): each row has prompt / label / metadata,
# and metadata.registry_image holds the SWE-Gym Docker image for the task.
PROMPT_DATA="${PROMPT_DATA:-/data/siraj/xyne_s/xyne_qa_train.jsonl}"
if [ ! -f "$PROMPT_DATA" ]; then
    echo "ERROR: prompt data not found at $PROMPT_DATA"
    echo "  (docker-JSONL rows must carry metadata.registry_image)"
    exit 1
fi

# ── Runtime configs ─────────────────────────────────────────────────────────
# Host dir with the Node 22 + agent CLIs, mounted as /opt/node in task pods.
export AGENT_CLI_DIR="${AGENT_CLI_DIR:-${PROJECT_ROOT}/tmp/swegym_agent_cli/opt_node}"
SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-9000}"
POLAR_CONFIG_TEMPLATE="${POLAR_CONFIG_TEMPLATE:-${SCRIPT_DIR}/polar_config_docker.yaml}"
CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${RUN_DIR}/polar_config_custom.yaml}"

# Render the YAML template: expand the host-path/auth placeholders used by the
# templates (${AGENT_CLI_DIR} always; the XYNE_* trio for the xye-qa config).
# Literal ${...} inside the task template (e.g. ${VARS} in bash harness
# scripts) is left untouched as long as it is not one of the listed vars.
command -v envsubst >/dev/null || { echo "ERROR: envsubst not found (install gettext-base)"; exit 1; }
mkdir -p "$(dirname "$CUSTOM_CONFIG_PATH")"
envsubst '${ACCESS_TOKEN_SECRET} ${AGENT_CLI_DIR} ${JUSPAY_API_KEY} ${JWT_SECRET} ${XYNE_ACCESS_TOKEN} ${XYNE_API_KEY} ${XYNE_AUTH_EMAIL} ${XYNE_AUTH_ROLE} ${XYNE_AUTH_WORKSPACE} ${XYNE_AGENT_ID} ${XYNE_BASE_URL} ${XYNE_JUDGE_BASE_URL} ${XYNE_JUDGE_MAX_CONCURRENCY} ${XYNE_JUDGE_MAX_TOKENS} ${XYNE_JUDGE_MODEL} ${XYNE_RUNTIME_IMAGE} ${XYNE_SERVER_DIR}' \
    < "$POLAR_CONFIG_TEMPLATE" > "$CUSTOM_CONFIG_PATH"

# cuDNN lib path for the runtime LD_LIBRARY_PATH.
if [ -z "${CUDNN_LIB:-}" ]; then
    CUDNN_LIB="$("${PYTHON_BIN}" -c 'import nvidia.cudnn, os; print(os.path.join(list(nvidia.cudnn.__path__)[0], "lib"))' 2>/dev/null || true)"
fi
RUNTIME_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [ -n "${CUDNN_LIB}" ] && [ -d "${CUDNN_LIB}" ]; then
    RUNTIME_LD_LIBRARY_PATH="${CUDNN_LIB}:${RUNTIME_LD_LIBRARY_PATH}"
fi

echo "Using HF checkpoint:    ${HF_CHECKPOINT}"
echo "Using ref-load:         ${REF_LOAD}"
echo "Using save dir:         ${SAVE_DIR}"
echo "Using Polar config:     ${CUSTOM_CONFIG_PATH}"
echo "Using prompt data:      ${PROMPT_DATA}"
# SGLANG_ROUTER_HOST is intentionally left unset unless the user pins an
# external router; when empty, Miles auto-launches its own router on this port.
echo "SGLang router:          http://${SGLANG_ROUTER_HOST:-<auto>}:${SGLANG_ROUTER_PORT}"

# ── GPU split ───────────────────────────────────────────────────────────────
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"   # 4 train GPUs
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"                 # 4 inference GPUs
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"   # TP=2 -> 2 engines
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"              # 4 prompts x N samples
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"         # 64 trajectories/rollout
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-30000}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-50000}"

RAY_NUM_GPUS="${RAY_NUM_GPUS:-$((ACTOR_NUM_GPUS_PER_NODE + ROLLOUT_NUM_GPUS))}"
RAY_HEAD_IP="${RAY_HEAD_IP:-127.0.0.1}"

echo "=== Starting Ray on ${RAY_HEAD_IP} (${RAY_NUM_GPUS} GPUs) ==="
ray stop --force 2>/dev/null || true
sleep 1
ray start --head --node-ip-address "$RAY_HEAD_IP" --num-gpus "$RAY_NUM_GPUS" --disable-usage-stats

MEGATRON_DIR="${MEGATRON_DIR:-}"
RUNTIME_PYTHONPATH="${PROJECT_ROOT}"
if [ -n "${MEGATRON_DIR}" ]; then
    RUNTIME_PYTHONPATH="${MEGATRON_DIR}:${RUNTIME_PYTHONPATH}"
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${RUNTIME_PYTHONPATH}\",
    \"PATH\": \"${PYTHON_BIN_DIR}:${PATH}\",
    \"VIRTUAL_ENV\": \"${VIRTUAL_ENV:-${PROJECT_ROOT}/.venv}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"max_split_size_mb:2048,expandable_segments:True\",
    \"WANDB_DIR\": \"${PROJECT_ROOT}/logs\",
    \"LD_LIBRARY_PATH\": \"${RUNTIME_LD_LIBRARY_PATH}\"
  }
}"

# ── Argument groups (assembled into the train.py flag list) ──────────────────
# Every flag below is a real Miles CLI flag. Each one's definition site in
# miles/utils/arguments.py / the Megatron backend, plus the slime->Miles rename
# notes, is listed in README "Flag mapping".

CKPT_ARGS=(
    --hf-checkpoint "$HF_CHECKPOINT"
    --ref-load "$REF_LOAD"
    --save "$SAVE_DIR"
    --save-interval "${SAVE_INTERVAL:-10}"
)
if [ -n "$LOAD_DIR" ]; then
    CKPT_ARGS+=(--load "$LOAD_DIR")
fi

# Polar wiring — Miles-mapped flags (slime names -> Miles names in README).
POLAR_ARGS=(
    --rollout-function-path miles.rollout.polar_rollout.generate_rollout_polar_async
    --custom-rm-path miles.rollout.polar_reward.custom_rm
    --custom-reward-post-process-path miles.rollout.polar_reward.post_process_rewards
    --custom-config-path "$CUSTOM_CONFIG_PATH"
    --data-source-path miles.rollout.polar_data_source.CeilEpochRolloutDataSourceWithBuffer
)

DATA_ARGS=(
    --prompt-data "$PROMPT_DATA"
    --input-key prompt
    --label-key label
    --metadata-key metadata
    --rollout-shuffle
    --reward-key score
    --num-rollout "${NUM_ROLLOUT:-10}"
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
    --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-16000}"
    --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN:-32000}"
    --num-steps-per-rollout 1
)

# Actor parallelism — 4 GPUs; MoE served at EP1/ETP1.
PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 2
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
    --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-256}"
    --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES:-30}"
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --normalize-advantages
    --use-tis
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-1e-6}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend auto
)

# LoRA adapter (rank 16) + SGLang serving.
SGLANG_ARGS=(
    --lora-rank "${LORA_RANK:-16}"
    --target-modules "${LORA_TARGET_MODULES:-all-linear}"
    --sglang-lora-backend triton
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.8}"
    --sglang-context-length "$SGLANG_CONTEXT_LENGTH"
    --sglang-router-port "$SGLANG_ROUTER_PORT"
    --sglang-router-policy "${SGLANG_ROUTER_POLICY:-round_robin}"
)

TRACKING_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT:-polar-swegym-grpo}"
    --wandb-group "${WANDB_GROUP:-swegym-nemotron30bnano-async-grpo}"
)

echo "=== Launching train.py ==="
ray job submit --address="http://${RAY_HEAD_IP}:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- "${PYTHON_BIN}" "${PROJECT_ROOT}/train.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "$ACTOR_NUM_GPUS_PER_NODE" \
    --rollout-num-gpus "$ROLLOUT_NUM_GPUS" \
    --rollout-num-gpus-per-engine "$ROLLOUT_NUM_GPUS_PER_ENGINE" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${POLAR_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${TRACKING_ARGS[@]}"
