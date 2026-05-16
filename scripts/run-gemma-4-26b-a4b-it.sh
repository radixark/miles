#!/bin/bash
# Single-node (8x H200) RL smoke test for Google Gemma 4 26B-A4B-it MoE.
# Mirrors scripts/run-nemotron-3-nano-30b-a3b.sh. Parallelism is TP=4xPP=1xEP=8
# (TP*DP=8 == ETP*EP=8 so Megatron's expert-parallel grouping is valid).
#
# Requires the radixark/Megatron-Bridge `zhichen/gemma4-on-bridge` branch
# installed (brings in Gemma4Bridge + Gemma4VLBridge, FusedExpertMapping,
# ABSENT_PROJECTION). Also requires transformers>=5.5.0 (the stock miles
# image ships 5.3.0; upgrading does not break sglang's import path).
#
# Model: google/gemma-4-26B-A4B-it (VLM repo; the AutoBridge resolves it as
# Gemma4VLModelProvider — the LM portion is what RL trains). Stage the
# checkpoint under $MODELS_DIR/google/gemma-4-26B-A4B-it.

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
# LLM_CKPT is a symlinked view of $MODELS_DIR/google/gemma-4-26B-A4B-it with a
# rewritten config.json that promotes text_config to top level and sets
# architectures=["Gemma4ForCausalLM"], so AutoBridge picks the LLM bridge and
# sglang dispatches to its native Gemma4ForCausalLM model implementation
# (instead of the generic transformers adapter, which can't drive the model).
LLM_CKPT=${LLM_CKPT:-/cluster_personal/zhichen/gemma4_test/gemma-4-26B-A4B-it-llm}

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
   --rm-type deepscaler
   --num-rollout 10
   --rollout-batch-size 32
   --n-samples-per-prompt 4
   --rollout-max-response-len 1024
   --rollout-temperature 1
   --global-batch-size 128
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
   --use-dynamic-batch-size
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

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=()

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.7
   # Gemma 4 has head_dim=512 on global layers, above the FlashAttention
   # cap of 256. Triton attention backend handles any head_dim.
   --sglang-attention-backend triton
   --use-miles-router
   # --use-rollout-routing-replay  # disabled while we debug update_weights illegal memory access
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend auto
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
