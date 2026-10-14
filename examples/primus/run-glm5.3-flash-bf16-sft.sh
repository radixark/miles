#!/bin/bash
# GLM-5.3-Flash thinking-mode SFT with Miles + Primus (trainer-only).
set -euo pipefail

MILES_DIR=${MILES_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
MEGATRON_DIR=${MEGATRON_DIR:-/root/Megatron-LM}
PRIMUS_DIR=${PRIMUS_DIR:-$(dirname "${MILES_DIR}")/Primus}
HF_CKPT=${HF_CKPT:-/root/models/GLM-5.3-Flash-bf16}
PRIMUS_CONFIG=${PRIMUS_CONFIG:-${MILES_DIR}/examples/primus/glm5.3-flash-bf16-rl.yaml}
SFT_DATA=${SFT_DATA:?Set SFT_DATA to a verified messages JSONL/Parquet file}
SFT_OUTPUT=${SFT_OUTPUT:?Set SFT_OUTPUT to the checkpoint/output directory}
TRAIN_NODES=${TRAIN_NODES:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-16384}

export PYTHONPATH="${MILES_DIR}:${MEGATRON_DIR}:${PRIMUS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PRIMUS_PATH="${PRIMUS_DIR}"
export PRIMUS_WORKSPACE=${PRIMUS_WORKSPACE:-${SFT_OUTPUT}/primus}
export GLM5_TOKENIZER="${HF_CKPT}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# MI300X (gfx942) has 64 KB of LDS. The default DSA backward tiles target
# gfx950's 160 KB and fail at launch. These are the CDNA3 settings from
# dsa_bwd_v4_triton.py.
export PRIMUS_DSA_DKV_SAFE=1
export PRIMUS_DSA_BWD_BLOCK_H=32
export PRIMUS_DSA_BWD_NUM_STAGES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export HSA_NO_SCRATCH_RECLAIM=1

case "${TRAIN_NODES}" in
  3) PP_ARGS=(--pipeline-model-parallel-size 3) ;;
  4) PP_ARGS=(--pipeline-model-parallel-size 4 --decoder-first-pipeline-num-layers 12 --decoder-last-pipeline-num-layers 11) ;;
  *) echo "TRAIN_NODES must be 3 or 4" >&2; exit 1 ;;
esac

TRAIN_ENV_VARS=$(python3 - <<'EOF'
import json, os
env = {}
if os.environ.get("TRAIN_PYTORCH_ALLOC_CONF"):
    env["PYTORCH_ALLOC_CONF"] = os.environ["TRAIN_PYTORCH_ALLOC_CONF"]
if os.environ.get("TURBO_PYTHONPATH"):
    env["PYTHONPATH"] = os.environ["TURBO_PYTHONPATH"] + ":" + os.environ["PYTHONPATH"]
env.update({k: v for k, v in os.environ.items() if k.startswith(("TORCH_DISTRIBUTED_", "TORCH_NCCL_"))})
print(json.dumps(env))
EOF
)

RUNTIME_ENV_JSON=$(python3 - <<'EOF'
import json, os
prefixes = ("NCCL_", "GLOO_", "TP_SOCKET_", "RAY_EXPERIMENTAL_", "PRIMUS_", "MILES_")
keys = ["PYTHONPATH", "LD_LIBRARY_PATH", "PATH", "GLM5_TOKENIZER", "CUDA_DEVICE_MAX_CONNECTIONS",
        "HSA_NO_SCRATCH_RECLAIM", "MASTER_ADDR", "no_proxy", "PYTHONUNBUFFERED", "TRITON_CACHE_DIR"]
env = {k: v for k, v in os.environ.items() if k in keys or k.startswith(prefixes)}
print(json.dumps({"env_vars": env}))
EOF
)

mkdir -p "${SFT_OUTPUT}"
cd "${MILES_DIR}"
ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${MILES_DIR}/train_async.py" \
  --primus-config "${PRIMUS_CONFIG}" \
  --primus-path "${PRIMUS_DIR}" \
  --train-backend megatron \
  --hf-checkpoint "${HF_CKPT}" \
  --ref-load "${HF_CKPT}" \
  --megatron-to-hf-mode raw \
  --load "${SFT_OUTPUT}/checkpoints" \
  --save "${SFT_OUTPUT}/checkpoints" \
  --save-interval "${SAVE_INTERVAL:-100}" \
  --no-save-optim \
  --rollout-function-path miles.rollout.sft_rollout.generate_rollout \
  --prompt-data "${SFT_DATA}" \
  --input-key messages \
  --chat-template-path "${HF_CKPT}/chat_template.jinja" \
  --loss-mask-type glm \
  --rollout-shuffle \
  --num-epoch "${NUM_EPOCH:-2}" \
  --rollout-batch-size "${GLOBAL_BATCH_SIZE}" \
  --rollout-max-context-len "${MAX_TOKENS_PER_GPU}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --loss-type sft_loss \
  --calculate-per-token-loss \
  --disable-compute-advantages-and-returns \
  --debug-train-only \
  --optimizer adam \
  --lr "${LR:-1e-5}" \
  --lr-decay-style cosine \
  --min-lr "${MIN_LR:-1e-6}" \
  --lr-warmup-fraction 0.1 \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --optimizer-cpu-offload \
  --overlap-cpu-optimizer-d2h-h2d \
  --use-precision-aware-optimizer \
  --tensor-model-parallel-size 1 \
  "${PP_ARGS[@]}" \
  --context-parallel-size 1 \
  --expert-model-parallel-size 8 \
  --expert-tensor-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --qkv-format bshd \
  --micro-batch-size 1 \
  --no-gradient-accumulation-fusion \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --actor-num-nodes "${TRAIN_NODES}" \
  --actor-num-gpus-per-node 8 \
  --num-gpus-per-node 8 \
  --train-env-vars "${TRAIN_ENV_VARS}" \
  "$@"
