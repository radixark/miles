#!/bin/bash
# GRPO on GLM-5.3-Flash (BF16) with the model built, loaded and patched by Primus.
# Run inside the head container of an existing multi-node Ray cluster of 8 x MI355X nodes:
#   train:   TRAIN_NODES (3 or 4), TP=1 EP=8 and one pipeline stage per node, Adam states on CPU.
#   rollout: ROLLOUT_NODES SGLang engines x TP=8, disaggregated from the trainers.
# Every node needs the HF checkpoint at HF_CKPT and the ROCm GLM-5 SGLang patch applied.
# DAPO-Math-17k train + AIME-2024 eval; rm-type math grades \boxed{} on the full response.
# TIS is on: SGLang and the trainer differ by ~0.08 nats per sampled token (median 0.003,
# a heavy tail on low-probability tokens; SGLang alone varies 0.04-0.06 run to run).
set -euo pipefail

MILES_DIR=${MILES_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
MEGATRON_DIR=${MEGATRON_DIR:-/root/Megatron-LM}
PRIMUS_DIR=${PRIMUS_DIR:-$(dirname "${MILES_DIR}")/Primus}
SGLANG_DIR=${SGLANG_DIR:-/sgl-workspace/sglang/python}
DATA_DIR=${DATA_DIR:-/root/datasets}
HF_CKPT=${HF_CKPT:-/root/models/GLM-5.3-Flash-bf16}
PRIMUS_CONFIG=${PRIMUS_CONFIG:-${MILES_DIR}/examples/primus/glm5.3-flash-bf16-rl.yaml}

TRAIN_NODES=${TRAIN_NODES:-4}
ROLLOUT_NODES=${ROLLOUT_NODES:-2}
RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-8192}

export PYTHONPATH="${MILES_DIR}:${MEGATRON_DIR}:${PRIMUS_DIR}:${SGLANG_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PRIMUS_PATH="${PRIMUS_DIR}"
export PRIMUS_WORKSPACE=${PRIMUS_WORKSPACE:-/tmp/primus}
export GLM5_TOKENIZER="${HF_CKPT}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# SGLang on ROCm: no DeepGEMM / CUDA tilelang mHC kernels, and keep short-sequence
# prefill off the dense-MHA path (it assumes a RoPE key part; GLM-5 DSA is NoPE).
export SGLANG_USE_AITER=1
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0
export SGLANG_OPT_FUSE_MHC_POST_PRE=0
export SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0

# One pipeline stage per training node; EP=8 stays within a node's data-parallel ranks.
case "${TRAIN_NODES}" in
  3) PP_ARGS=(--pipeline-model-parallel-size 3) ;;
  4) PP_ARGS=(--pipeline-model-parallel-size 4 --decoder-first-pipeline-num-layers 12 --decoder-last-pipeline-num-layers 11) ;;
  *) echo "TRAIN_NODES must be 3 or 4 (45 layers, one pipeline stage per node)" >&2; exit 1 ;;
esac

cd "${MILES_DIR}"

export HSA_NO_SCRATCH_RECLAIM=1

# Primus-Turbo pins a FlyDSL release the rollout engines' aiter cannot use, so only the
# trainers see TURBO_PYTHONPATH (a `pip install --target` of flydsl==0.2.4).
TRAIN_ENV_VARS=$(python3 - <<'EOF'
import json, os
env = {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}
if os.environ.get("TURBO_PYTHONPATH"):
    env["PYTHONPATH"] = os.environ["TURBO_PYTHONPATH"] + ":" + os.environ["PYTHONPATH"]
print(json.dumps(env))
EOF
)

RUNTIME_ENV_JSON=$(python3 - <<'EOF'
import json, os
prefixes = ("SGLANG_", "NCCL_", "GLOO_", "TP_SOCKET_", "RAY_EXPERIMENTAL_", "PRIMUS_")
keys = ["PYTHONPATH", "LD_LIBRARY_PATH", "PATH", "GLM5_TOKENIZER", "CUDA_DEVICE_MAX_CONNECTIONS", "HSA_NO_SCRATCH_RECLAIM",
        "MASTER_ADDR", "no_proxy", "PYTHONUNBUFFERED", "TRITON_CACHE_DIR"]
env = {k: v for k, v in os.environ.items() if k in keys or k.startswith(prefixes)}
print(json.dumps({"env_vars": env}))
EOF
)

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${MILES_DIR}/train_async.py" \
  --primus-config "${PRIMUS_CONFIG}" \
  --primus-path "${PRIMUS_DIR}" \
  --train-backend megatron \
  --hf-checkpoint "${HF_CKPT}" \
  --ref-load "${HF_CKPT}" \
  --megatron-to-hf-mode raw \
  --prompt-data "${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl" \
  --input-key prompt \
  --label-key label \
  --apply-chat-template \
  --rollout-shuffle \
  --rm-type math \
  --num-rollout "${NUM_ROLLOUT:-100}" \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-8}" \
  --n-samples-per-prompt 8 \
  --global-batch-size "$(( ${ROLLOUT_BATCH_SIZE:-8} * 8 ))" \
  --rollout-max-response-len "${RESPONSE_LEN}" \
  --rollout-max-context-len "$(( RESPONSE_LEN + 2048 ))" \
  --rollout-temperature 1.0 \
  --eval-interval "${EVAL_INTERVAL:-10}" \
  --eval-prompt-data aime24 "${DATA_DIR}/aime-2024/aime-2024.jsonl" \
  --eval-input-key prompt \
  --eval-label-key label \
  --n-samples-per-eval-prompt 1 \
  --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-${RESPONSE_LEN}}" \
  --eval-top-k 1 \
  --advantage-estimator grpo \
  --kl-coef 0.00 \
  --entropy-coef 0.00 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --use-tis \
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
  --calculate-per-token-loss \
  --rollout-num-gpus-per-engine 8 \
  --sglang-router-policy round_robin \
  --sglang-mem-fraction-static 0.8 \
  --sglang-disable-cuda-graph \
  --sglang-disable-radix-cache \
  --sglang-max-running-requests 128 \
  --sglang-chunked-prefill-size 16384 \
  --actor-num-nodes "${TRAIN_NODES}" \
  --actor-num-gpus-per-node 8 \
  --num-gpus-per-node 8 \
  --rollout-num-gpus "$(( ROLLOUT_NODES * 8 ))" \
  --train-env-vars "${TRAIN_ENV_VARS}" \
  "$@"
