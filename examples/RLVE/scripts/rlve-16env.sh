#!/bin/bash
# RLVE Training Script for Miles deployment
# Adapted from RLVE num-environment=16.sh for Qwen2.5-0.5B on miles-gmi-tinker

set -ex

# Configuration
MODEL_PATH=${MODEL_PATH:-"/data/models/Qwen2.5-0.5B-Instruct"}
SAVE_PATH=${SAVE_PATH:-"/tmp/rlve-checkpoints"}
WANDB_PROJECT=${WANDB_PROJECT:-"miles-rlve-test"}
RUN_NAME=${RUN_NAME:-"rlve-16env-qwen0.5b"}

# 16 environments from the original RLVE paper
ENVIRONMENT_LIST="Division EuclidGame GCDOne_Counting HamiltonianPath LampChanging LargestConvexPolygon Multiplication PCPPermutation Path_NoGoingBack_Counting SAT ShortestPath Sorting SpiralMatrix SubsequenceReversalLNDS UndamagedSubmatrixCounting WYRLevelingGround"

# Will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Add RLVE Gym to PYTHONPATH
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}/../:${PYTHONPATH}"

# Model args for Qwen2.5-0.5B-Instruct
# (scaled down from 7B for testing)
MODEL_ARGS=(
   --swiglu
   --num-layers 24
   --hidden-size 896
   --ffn-hidden-size 4864
   --num-attention-heads 14
   --group-query-attention
   --num-query-groups 2
   --use-rotary-position-embeddings
   --disable-bias-linear
   --add-qkv-bias
   --normalization "RMSNorm"
   --norm-epsilon 1e-06
   --rotary-base 1000000
   --vocab-size 151936
   --untie-embeddings-and-output-weights
)

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_PATH}
   --save ${SAVE_PATH}
   --save-interval 10
)

ROLLOUT_ARGS=(
   --disable-rollout-global-dataset
   --rlve
   --environment-list ${ENVIRONMENT_LIST}

   --custom-prompt-preprocessor TinyZero
   --answer-marker-type "<answer></answer>"

   --rm-type rlve
   --reward-key reward

   --num-rollout 100
   --rollout-batch-size 8        # Scaled down for 0.5B
   --n-samples-per-prompt 4      # Scaled down
   --rollout-max-response-len 1024
   --rollout-temperature 1.0

   --num-steps-per-rollout 1
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 1  # Smaller model, less parallelism needed
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 2048
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project "${WANDB_PROJECT}"
   --wandb-group "${RUN_NAME}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# Check if Ray is already running, otherwise start it
if ! ray status &>/dev/null; then
    echo "Starting Ray head node..."
    ray start --head --num-gpus 8 --disable-usage-stats
fi

# Run training
python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
