#!/bin/bash
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
   -- python3 examples/multi_lora/train_multi_lora.py \
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
   --multi-lora-n-adapters 4 \
   --multi-lora-idle-poll-s 5 \
   --multi-lora-adapter "dapo_math" "examples/multi_lora/adapters/dapo_math/adapter.yaml" \
   --multi-lora-adapter "gsm8k" "examples/multi_lora/adapters/gsm8k/adapter.yaml" \
   --multi-lora-disable-service-mode \
   --sglang-lora-backend triton \
   \
   --apply-chat-template \
   --rollout-shuffle \
   --num-rollout 50 \
   --rollout-batch-size 32 \
   --n-samples-per-prompt 8 \
   --rollout-max-response-len 4096 \
   --rollout-temperature 1 \
   --global-batch-size 256 \
   \
   --save /tmp/multi_lora \
   --save-interval 5 \
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
   --wandb-team staging \
   --wandb-project miles-multilora \
   --wandb-group qwen3-4B

