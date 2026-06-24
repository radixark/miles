#!/bin/bash
# Single-node (8x H200) RL smoke test for Gemma-4 26B-A4B-it MoE, TP8/EP8.
# Needs the radixark/Megatron-Bridge gemma4 branch and transformers>=5.5.0.

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/gemma-4-26b-a4b-it.sh"

MODELS_DIR=${MODELS_DIR:-/storage/models}
DATASETS_DIR=${DATASETS_DIR:-/cluster_public/miles_data/datasets}
# LLM_CKPT: checkpoint view whose config.json sets architectures=["Gemma4ForCausalLM"]
# so AutoBridge/sglang take the text-LLM path.
LLM_CKPT=${LLM_CKPT:-/storage/models/google/gemma-4-26B-A4B-it}

CKPT_ARGS=(
   --hf-checkpoint $LLM_CKPT
   --ref-load $LLM_CKPT
   --save $MODELS_DIR/google/gemma-4-26B-A4B-it_miles
   --save-interval 20
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data $DATASETS_DIR/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type gemma_math
   --num-rollout 10
   --rollout-batch-size 4
   --n-samples-per-prompt 2
   --rollout-max-response-len 256
   --rollout-temperature 1
   --global-batch-size 8
   --balance-data
)

EVAL_ARGS=(
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --micro-batch-size 1
   --max-tokens-per-gpu 1024
   --log-probs-chunk-size 128
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   # --optimizer-cpu-offload
   # --overlap-cpu-optimizer-d2h-h2d
   # --use-precision-aware-optimizer
)

WANDB_ARGS=()

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.55
   # triton: Gemma-4 global head_dim=512 exceeds FlashAttention's 256 cap
   --sglang-attention-backend triton
   --sglang-moe-runner-backend triton
   --sglang-disable-custom-all-reduce
   --sglang-disable-cuda-graph
   --sglang-disable-overlap-schedule
   --sglang-disable-radix-cache
   --no-offload-train
   --no-offload-rollout
   --use-miles-router
   --use-rollout-routing-replay
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --no-gradient-accumulation-fusion
   --no-check-for-nan-in-loss-and-grad
   --attention-softmax-in-fp32
   --attention-backend unfused
   --qkv-format bshd
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --colocate \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
