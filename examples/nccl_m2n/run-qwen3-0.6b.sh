#!/usr/bin/env bash
# Four allocated GPUs: two Megatron PP stages + one SGLang TP2 engine.
# See README.md for checkpoint preparation and starting the Ray head.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MILES_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# Preserve this example's externally managed Ray lifecycle.
MODEL_ARGS_TEXT="$(python3 "${MILES_ROOT}/miles/utils/external_utils/model_args_utils.py" qwen3-0.6B)"
read -r -a MODEL_ARGS <<< "${MODEL_ARGS_TEXT}"

: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the local Qwen3-0.6B directory}"
: "${TRAIN_CHECKPOINT:?Set TRAIN_CHECKPOINT to its converted torch_dist checkpoint}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
DATA_DIR="${DATA_DIR:-/root/datasets}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl}"
for input in "${HF_CHECKPOINT}/config.json" "${TRAIN_CHECKPOINT}/latest_checkpointed_iteration.txt" "${PROMPT_DATA}"; do
    if [[ ! -f "${input}" ]]; then
        echo "Missing input: ${input}" >&2
        exit 1
    fi
done
if [[ ! -d "${MEGATRON_PATH}/megatron" ]]; then
    echo "MEGATRON_PATH must point to the Megatron-LM checkout" >&2
    exit 1
fi

export PYTHONPATH="${MILES_ROOT}:${MEGATRON_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Pass source paths to Ray workers, not just the submission process.
RUNTIME_ENV_JSON="$(python3 -c 'import json, os; print(json.dumps({"env_vars": {k: os.environ[k] for k in ("PYTHONPATH", "CUDA_DEVICE_MAX_CONNECTIONS")}}))')"

cd "${MILES_ROOT}"
exec ray job submit --address="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" -- python3 "${MILES_ROOT}/train.py" \
    "${MODEL_ARGS[@]}" \
    --train-backend megatron --actor-num-nodes 1 --actor-num-gpus-per-node 2 \
    --num-gpus-per-node 4 --rollout-num-gpus 2 --rollout-num-gpus-per-engine 2 \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 2 \
    --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 \
    --hf-checkpoint "${HF_CHECKPOINT}" --load "${TRAIN_CHECKPOINT}" \
    --no-load-optim --no-load-rng --finetune --bf16 \
    --update-weight-transfer-mode nccl-m2n --m2n-pp-concurrency "${M2N_PP_CONCURRENCY:-2}" \
    --check-weight-update-equal \
    --prompt-data "${PROMPT_DATA}" --input-key prompt --label-key label \
    --apply-chat-template --apply-chat-template-kwargs '{"enable_thinking":false}' \
    --rm-type math --num-rollout 2 --rollout-batch-size 4 --n-samples-per-prompt 2 \
    --rollout-max-response-len 256 --rollout-temperature 1 \
    --global-batch-size 8 --micro-batch-size 1 \
    --advantage-estimator grpo --entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.28 \
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 \
    --attention-dropout 0 --hidden-dropout 0 --attention-backend flash \
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
    --sglang-mem-fraction-static 0.5 \
    "$@"
