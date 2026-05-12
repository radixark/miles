#!/bin/bash
# Multi-LoRA dynamic-lifecycle dev run.
#
# Drives a hard-coded register/deregister schedule via
# train_multi_lora_dynamic.py to exercise the online add/remove path:
#   wait → +dapo → +gsm8k → -dapo → -gsm8k+wait → +gsm8k → +dapo → end
#
# Each productive phase runs 3 rollouts; the final phase runs to
# --num-rollout. With 18 total productive rollouts, the final phase gets 6.

set -ex

export GPUS_PER_NODE=8

pkill sglang || true
ray stop --force || true
sleep 3

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source scripts/models/qwen3-4B.sh

ray start --head --node-ip-address 127.0.0.1 --num-gpus $GPUS_PER_NODE --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1"
     }
   }' \
   -- python3 examples/multi_lora/train_multi_lora_dynamic.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node $GPUS_PER_NODE \
   --colocate \
   --calculate-per-token-loss \
   --use-miles-router \
   ${MODEL_ARGS[@]} \
   \
   --hf-checkpoint /root/Qwen3-4B/ \
   --megatron-to-hf-mode bridge \
   \
   --lora-rank 32 \
   --lora-alpha 32 \
   --lora-dropout 0.0 \
   --target-modules "all-linear" \
   --multi-lora-dir "${SCRIPT_DIR}/adapters" \
   --multi-lora-n-adapters 4 \
   --multi-lora-idle-poll-s 5 \
   --sglang-lora-backend triton \
   \
   --prompt-data /root/gsm8k/train.parquet \
   --input-key messages \
   --label-key label \
   --apply-chat-template \
   --rollout-shuffle \
   --num-rollout 50 \
   --rollout-batch-size 32 \
   --n-samples-per-prompt 8 \
   --rollout-max-response-len 4096 \
   --rollout-temperature 1 \
   --global-batch-size 256 \
   \
   --save /tmp/multi_lora_dev2_save \
   --save-interval 1 \
   \
   --advantage-estimator grpo \
   --kl-loss-coef 0.00 \
   --kl-coef 0.00 \
   --entropy-coef 0.00 \
   --eps-clip 0.2 \
   --eps-clip-high 0.28 \
   \
   --optimizer adam \
   --lr 1e-5 \
   --lr-decay-style constant \
   --weight-decay 0.1 \
   --adam-beta1 0.9 \
   --adam-beta2 0.98 \
   \
   --tensor-model-parallel-size 2 \
   --sequence-parallel \
   --pipeline-model-parallel-size 1 \
   --context-parallel-size 1 \
   --expert-model-parallel-size 1 \
   --expert-tensor-parallel-size 1 \
   --use-dynamic-batch-size \
   --max-tokens-per-gpu 9216 \
   \
   --rollout-num-gpus-per-engine 1 \
   --sglang-mem-fraction-static 0.4 \
   \
   --attention-dropout 0.0 \
   --hidden-dropout 0.0 \
   --accumulate-allreduce-grads-in-fp32 \
   --attention-softmax-in-fp32 \
   --attention-backend flash \
   \
   --use-wandb \
   --wandb-host https://wandb.ai/ \
   --wandb-team osmosis-staging \
   --wandb-project miles-multilora \
   --wandb-group qwen3-4B-dev2-dynamic
