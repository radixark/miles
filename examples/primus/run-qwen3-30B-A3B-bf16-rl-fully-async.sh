#!/bin/bash
# Fully-async GRPO on Qwen3-30B-A3B (MoE) with Megatron configured and patched by Primus.
# Disaggregated single-node layout: train_async.py rejects --colocate.
# 4 train GPUs: TP=1 PP=1 CP=1 EP=4. 4 rollout GPUs: 2 engines x TP=2.
# DAPO-Math-17k train + AIME-2024 eval. rm-type math grades \\boxed{} on the
# full response (deepscaler requires </think>, which Qwen3 does not emit when
# enable_thinking=false). Megatron comes from the image; Miles, Primus and
# Primus-Turbo come from the checkouts below.
# PRIMUS_CONFIG=none runs the same recipe on plain Megatron for comparison.
set -euo pipefail

MILES_DIR=${MILES_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
MEGATRON_DIR=${MEGATRON_DIR:-/root/Megatron-LM}
PRIMUS_DIR=${PRIMUS_DIR:-$(dirname "${MILES_DIR}")/Primus}
SGLANG_DIR=${SGLANG_DIR:-/sgl-workspace/sglang/python}
MODEL_DIR=${MODEL_DIR:-/root/models}
DATA_DIR=${DATA_DIR:-/root/datasets}

TRAIN_GPUS=${TRAIN_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
NUM_GPUS=${NUM_GPUS:-$((TRAIN_GPUS + ROLLOUT_GPUS))}
PRIMUS_CONFIG=${PRIMUS_CONFIG:-${MILES_DIR}/examples/primus/qwen3-30B-A3B-bf16-rl.yaml}
HF_CKPT=${HF_CKPT:-${MODEL_DIR}/Qwen3-30B-A3B}

export PYTHONPATH="${MILES_DIR}:${MEGATRON_DIR}:${PRIMUS_DIR}:${SGLANG_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PRIMUS_PATH="${PRIMUS_DIR}"

export GPU_ARCHS=${GPU_ARCHS:-gfx950}
export CU_NUM=${CU_NUM:-256}
export PYTHONUNBUFFERED=1
export MASTER_ADDR=127.0.0.1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export no_proxy=127.0.0.1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PRIMUS_WORKSPACE=${PRIMUS_WORKSPACE:-/tmp/primus}

cd "${MILES_DIR}"

pkill -9 sglang || true
ray stop --force || true
sleep 3
ray start --head --node-ip-address 127.0.0.1 --num-gpus "${NUM_GPUS}" --disable-usage-stats

read -r -a MODEL_ARGS <<<"$(PYTHONPATH="${MILES_DIR}/scripts/models:${MILES_DIR}/miles/utils/external_utils" python3 -c 'import importlib; print(importlib.import_module("qwen3-30B-A3B").model_args())')"

PRIMUS_FLAG=(--primus-config "${PRIMUS_CONFIG}" --primus-path "${PRIMUS_DIR}")
if [[ "${PRIMUS_CONFIG}" == "none" ]]; then
  PRIMUS_FLAG=()
fi

# Primus-Turbo pins a FlyDSL release the rollout engines' aiter cannot use, so the
# trainers alone see TURBO_PYTHONPATH (e.g. a `pip install --target` of flydsl==0.2.4).
TRAIN_ENV_VARS=$(python3 -c 'import json, os; p = os.environ.get("TURBO_PYTHONPATH"); print(json.dumps({"PYTHONPATH": p + ":" + os.environ["PYTHONPATH"]} if p else {}))')

RUNTIME_ENV_JSON=$(python3 - <<'EOF'
import json, os
keys = [
    "PYTHONPATH", "PRIMUS_PATH", "PRIMUS_WORKSPACE", "LD_LIBRARY_PATH", "PATH", "GPU_ARCHS", "CU_NUM",
    "CUDA_DEVICE_MAX_CONNECTIONS", "MASTER_ADDR", "no_proxy",
    "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES", "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "MILES_ALLOW_UNSAFE_PRIMUS_TURBO",
]
print(json.dumps({"env_vars": {k: os.environ[k] for k in keys if os.environ.get(k)}}))
EOF
)

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train_async.py \
  "${MODEL_ARGS[@]}" \
  "${PRIMUS_FLAG[@]}" \
  --train-backend megatron \
  --hf-checkpoint "${HF_CKPT}" \
  --megatron-to-hf-mode bridge \
  --prompt-data "${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl" \
  --input-key prompt \
  --label-key label \
  --apply-chat-template \
  --apply-chat-template-kwargs '{"enable_thinking": false}' \
  --rollout-shuffle \
  --rm-type math \
  --num-rollout "${NUM_ROLLOUT:-100}" \
  --rollout-batch-size 4 \
  --n-samples-per-prompt 8 \
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-8192}" \
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN:-16384}" \
  --rollout-temperature 1.0 \
  --global-batch-size 32 \
  --eval-interval "${EVAL_INTERVAL:-5}" \
  --eval-prompt-data aime24 "${DATA_DIR}/aime-2024/aime-2024.jsonl" \
  --eval-input-key prompt \
  --eval-label-key label \
  --n-samples-per-eval-prompt 1 \
  --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-8192}" \
  --eval-top-k 1 \
  --advantage-estimator grpo \
  --kl-coef 0.00 \
  --entropy-coef 0.00 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --fully-async \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --optimizer-cpu-offload \
  --overlap-cpu-optimizer-d2h-h2d \
  --use-precision-aware-optimizer \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size "${TRAIN_GPUS}" \
  --expert-tensor-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --qkv-format bshd \
  --micro-batch-size 1 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend unfused \
  --rollout-num-gpus-per-engine 2 \
  --sglang-mem-fraction-static 0.5 \
  --sglang-cuda-graph-max-bs-decode "${SGLANG_CUDA_GRAPH_MAX_BS:-32}" \
  --calculate-per-token-loss \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${TRAIN_GPUS}" \
  --num-gpus-per-node "${NUM_GPUS}" \
  --rollout-num-gpus "${ROLLOUT_GPUS}" \
  --train-env-vars "${TRAIN_ENV_VARS}" \
  "$@"
