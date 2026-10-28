#!/bin/bash
# GRPO on GLM-5.3-Flash for a verifiable reasoning task (DAPO-Math-17k, AIME-2024).
#
# Rollout is the released FP8 checkpoint (zai-org/GLM-5.3-Flash): SGLang loads it,
# and weight updates re-quantize the BF16 trainer with that checkpoint's
# quantization_config, including modules_to_not_convert. The trainer loads the
# BF16 cast of the same weights through --ref-load.
#
# CP_SIZE>1 packs THD and splits each sequence across the CP ranks
# (--allgather-cp), with dynamic batching. CP folds onto the same GPUs as EP,
# so it does not add nodes. MAX_TOKENS_PER_GPU * CP_SIZE must cover the
# prompt plus the longest response.
#
# Run inside the head container of a Ray cluster. Defaults are the 4+2 node
# shape (one pipeline stage per trainer node, one SGLang engine per rollout node).
set -euo pipefail

MILES_DIR=${MILES_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
MEGATRON_DIR=${MEGATRON_DIR:-/root/Megatron-LM}
PRIMUS_DIR=${PRIMUS_DIR:-$(dirname "${MILES_DIR}")/Primus}
SGLANG_DIR=${SGLANG_DIR:-/sgl-workspace/sglang/python}
DATA_DIR=${DATA_DIR:-/root/datasets}
# Released FP8 checkpoint: SGLang's model path and the quant config for updates.
HF_CKPT=${HF_CKPT:-/root/models/GLM-5.3-Flash}
# BF16 cast of that checkpoint. Primus loads these weights into the trainer.
REF_CKPT=${REF_CKPT:-/root/models/GLM-5.3-Flash-bf16}
PRIMUS_CONFIG=${PRIMUS_CONFIG:-${MILES_DIR}/examples/primus/glm5.3-flash-bf16-rl.yaml}

TRAIN_NODES=${TRAIN_NODES:-4}
ROLLOUT_NODES=${ROLLOUT_NODES:-2}
RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-16384}
PROMPT_LEN=${ROLLOUT_MAX_PROMPT_LEN:-2048}
EVAL_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-${RESPONSE_LEN}}
SGLANG_CONTEXT_LEN=$(( RESPONSE_LEN + PROMPT_LEN ))
if [ "${EVAL_RESPONSE_LEN}" -gt "${RESPONSE_LEN}" ]; then
  SGLANG_CONTEXT_LEN=$(( EVAL_RESPONSE_LEN + PROMPT_LEN ))
fi

export PYTHONPATH="${MILES_DIR}:${MEGATRON_DIR}:${PRIMUS_DIR}:${SGLANG_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PRIMUS_PATH="${PRIMUS_DIR}"
export PRIMUS_WORKSPACE=${PRIMUS_WORKSPACE:-/tmp/primus}
export GLM5_TOKENIZER="${HF_CKPT}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
# 0: SGLang reads the released FP8 checkpoint. The first weight update still
# overwrites it with the trainer. Set 1 only when the engines cannot read the
# checkpoint and should allocate from its config alone.
export MILES_SGLANG_DUMMY_LOAD=${MILES_SGLANG_DUMMY_LOAD:-0}
# NFS checkpoint loads take 25-90 min when many nodes read at once.
export MILES_CELLS_READY_TIMEOUT_SECONDS=${MILES_CELLS_READY_TIMEOUT_SECONDS:-7200}

export SGLANG_USE_AITER=1
# rocm: TileLang DSA, AITER MoE, decode CUDA graphs. legacy: image SGLang, graphs off.
SGLANG_PROFILE=${SGLANG_PROFILE:-rocm}
if [ "${SGLANG_PROFILE}" = rocm ]; then
  export AITER_ONLINE_TUNE=0 HIP_FORCE_DEV_KERNARG=1
  SGLANG_PROFILE_ARGS=(--sglang-dsa-prefill-backend tilelang --sglang-dsa-decode-backend tilelang
    --sglang-moe-runner-backend aiter --sglang-cuda-graph-max-bs-decode 128
    # Match the trainer and the GB300 recipe. The model default is FP8 KV, which
    # the DSA backends warn against and which widens the train/rollout logprob gap.
    --sglang-kv-cache-dtype bfloat16
    # AITER custom all-reduce (with FP8 weights) intermittently yields NaN logits in long decodes,
    # which surface as token 0 with null logprobs. RCCL decodes at the same speed.
    --sglang-disable-custom-all-reduce
    # Every TP rank reads every shard; prefetch splits the shards across local ranks into the
    # shared page cache so NFS serves the checkpoint once per node instead of once per rank.
    --sglang-weight-loader-prefetch-checkpoints)
else
  export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
  export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
  export SGLANG_OPT_USE_TILELANG_MHC_POST=0
  export SGLANG_OPT_FUSE_MHC_POST_PRE=0
  export SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0
  SGLANG_PROFILE_ARGS=()
fi

case "${TRAIN_NODES}" in
  3) PP_ARGS=(--pipeline-model-parallel-size 3) ;;
  4) PP_ARGS=(--pipeline-model-parallel-size 4 --decoder-first-pipeline-num-layers 12 --decoder-last-pipeline-num-layers 11) ;;
  # 5 + 6x6 + 4 layers; the last stage also holds the LM head.
  8) PP_ARGS=(--pipeline-model-parallel-size 8 --decoder-first-pipeline-num-layers 5 --decoder-last-pipeline-num-layers 4) ;;
  *) echo "TRAIN_NODES must be 3, 4 or 8 (45 layers, one pipeline stage per node)" >&2; exit 1 ;;
esac

# CP_SIZE>1 packs THD. CP and EP share the GPUs on each trainer node.
CP_SIZE=${CP_SIZE:-8}
if [ "${CP_SIZE}" -gt 1 ]; then
  MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-4096}
  PACKED_LEN=$(( MAX_TOKENS_PER_GPU * CP_SIZE ))
  [ "${PACKED_LEN}" -ge "${SGLANG_CONTEXT_LEN}" ] || {
    echo "MAX_TOKENS_PER_GPU*CP_SIZE (${PACKED_LEN}) < context ${SGLANG_CONTEXT_LEN}" >&2
    exit 1
  }
  export K3P_KDA_CONV=${K3P_KDA_CONV:-fla} PRIMUS_DSA_BWD_BLOCK_H=${PRIMUS_DSA_BWD_BLOCK_H:-32}
  SEQ_ARGS=(--context-parallel-size "${CP_SIZE}" --qkv-format thd --allgather-cp
    --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --primus-override seq_length=${PACKED_LEN} max_position_embeddings=${PACKED_LEN})
else
  SEQ_ARGS=(--context-parallel-size 1 --qkv-format bshd)
fi

cd "${MILES_DIR}"

CUDA_GRAPH_ARGS=(--sglang-disable-cuda-graph)
[ "${SGLANG_CUDA_GRAPH:-0}" = 1 ] || [ "${SGLANG_PROFILE}" = rocm ] && CUDA_GRAPH_ARGS=()

export HSA_NO_SCRATCH_RECLAIM=1

TRAIN_ENV_VARS=$(python3 - <<'EOF'
import json, os
# expandable_segments breaks lazily-initialized RCCL communicators on ROCm 10 / gfx942:
# the first collective on each new communicator returns its output buffer unwritten.
env = {}
if os.environ.get("TRAIN_PYTORCH_ALLOC_CONF"):
    env["PYTORCH_ALLOC_CONF"] = os.environ["TRAIN_PYTORCH_ALLOC_CONF"]
if os.environ.get("TRAIN_HSA_NO_SCRATCH_RECLAIM"):
    env["HSA_NO_SCRATCH_RECLAIM"] = os.environ["TRAIN_HSA_NO_SCRATCH_RECLAIM"]
if os.environ.get("TURBO_PYTHONPATH"):
    env["PYTHONPATH"] = os.environ["TURBO_PYTHONPATH"] + ":" + os.environ["PYTHONPATH"]
for key in ("K3P_KDA_CONV", "PRIMUS_DSA_DKV_SAFE", "PRIMUS_DSA_BWD_BLOCK_H", "PRIMUS_DSA_BWD_NUM_STAGES"):
    if os.environ.get(key):
        env[key] = os.environ[key]
env.update({k: v for k, v in os.environ.items() if k.startswith(("TORCH_DISTRIBUTED_", "TORCH_NCCL_"))})
print(json.dumps(env))
EOF
)

RUNTIME_ENV_JSON=$(python3 - <<'EOF'
import json, os
prefixes = ("SGLANG_", "NCCL_", "GLOO_", "TP_SOCKET_", "RAY_EXPERIMENTAL_", "PRIMUS_", "MILES_", "AITER_", "K3P_")
keys = ["PYTHONPATH", "LD_LIBRARY_PATH", "PATH", "GLM5_TOKENIZER", "CUDA_DEVICE_MAX_CONNECTIONS", "HSA_NO_SCRATCH_RECLAIM",
        "HIP_FORCE_DEV_KERNARG",
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
  --ref-load "${REF_CKPT}" \
  --megatron-to-hf-mode raw \
  --prompt-data "${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl" \
  --input-key prompt \
  --label-key label \
  --apply-chat-template \
  --rollout-shuffle \
  --rm-type math \
  --num-rollout "${NUM_ROLLOUT:-20}" \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-8}" \
  --n-samples-per-prompt 8 \
  --global-batch-size "$(( ${ROLLOUT_BATCH_SIZE:-8} * 8 ))" \
  --rollout-max-response-len "${RESPONSE_LEN}" \
  --rollout-max-prompt-len "${PROMPT_LEN}" \
  --rollout-max-context-len "${SGLANG_CONTEXT_LEN}" \
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}" \
  --eval-interval "${EVAL_INTERVAL:-10}" \
  --eval-prompt-data aime24 "${DATA_DIR}/aime-2024/aime-2024.jsonl" \
  --eval-input-key prompt \
  --eval-label-key label \
  --n-samples-per-eval-prompt 1 \
  --eval-max-response-len "${EVAL_RESPONSE_LEN}" \
  --eval-top-k 1 \
  --advantage-estimator grpo \
  --use-rollout-routing-replay \
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
  --no-save-optim \
  --no-load-optim \
  --tensor-model-parallel-size 1 \
  "${PP_ARGS[@]}" \
  "${SEQ_ARGS[@]}" \
  --expert-model-parallel-size 8 \
  --expert-tensor-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --micro-batch-size 1 \
  --no-gradient-accumulation-fusion \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --calculate-per-token-loss \
  --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES:-30}" \
  --rollout-num-gpus-per-engine 8 \
  --sglang-router-policy round_robin \
  --sglang-mem-fraction-static 0.7 \
  "${CUDA_GRAPH_ARGS[@]}" \
  "${SGLANG_PROFILE_ARGS[@]}" \
  --sglang-disable-radix-cache \
  --sglang-max-running-requests 128 \
  --sglang-chunked-prefill-size 16384 \
  --sglang-context-length "${SGLANG_CONTEXT_LEN}" \
  --actor-num-nodes "${TRAIN_NODES}" \
  --actor-num-gpus-per-node 8 \
  --num-gpus-per-node 8 \
  --rollout-num-gpus "$(( ROLLOUT_NODES * 8 ))" \
  --train-env-vars "${TRAIN_ENV_VARS}" \
  "$@"
