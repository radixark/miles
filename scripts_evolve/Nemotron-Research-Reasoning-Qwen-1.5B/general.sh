#!/bin/bash
# General training script with configurable parameters
# Usage: ./scripts_evolve/${MODEL_NAME}/general.sh WANDB_PROJECT RUN_NAME INITIAL_PROGRAM EVALUATOR_FILE CONFIG_YAML SAVE_PATH IS_TRAINING LAZY_OUTPUT_PENALTY REWARD_PROCESS_TYPE SEED

if [ $# -ne 10 ]; then
  echo "Usage: $0 WANDB_PROJECT RUN_NAME INITIAL_PROGRAM EVALUATOR_FILE CONFIG_YAML SAVE_PATH IS_TRAINING LAZY_OUTPUT_PENALTY REWARD_PROCESS_TYPE SEED"
  echo ""
  echo "Required parameters:"
  echo "  WANDB_PROJECT        - Weights & Biases project name"
  echo "  RUN_NAME             - Experiment run name"
  echo "  INITIAL_PROGRAM      - Path to initial program file"
  echo "  EVALUATOR_FILE       - Path to evaluator file"
  echo "  CONFIG_YAML          - Path to config YAML file"
  echo "  SAVE_PATH            - Save directory path"
  echo "  IS_TRAINING          - True for training, False for inference-only"
  echo "  LAZY_OUTPUT_PENALTY  - Lazy output penalty level (1 or 2)"
  echo "  REWARD_PROCESS_TYPE  - Reward processing type (original_reward, rl_normalized_reward, etc.)"
  echo "  SEED                 - Random seed for reproducibility"
  exit 1
fi

WANDB_PROJECT=$1
RUN_NAME=$2
INITIAL_PROGRAM=$3
EVALUATOR_FILE=$4
CONFIG_YAML=$5
SAVE_PATH=$6
IS_TRAINING=$7
LAZY_OUTPUT_PENALTY=$8
REWARD_PROCESS_TYPE=$9
SEED=${10}

SAVE_SHM_DIR="${SAVE_PATH}/shm"
CKPT_DIR="${SAVE_PATH}/${RUN_NAME}"
RECORD_PATH="${SAVE_PATH}/${RUN_NAME}/records"
MODEL_NAME="Nemotron-Research-Reasoning-Qwen-1.5B"

# Determine debug-rollout-only mode based on IS_TRAINING
if [ "$IS_TRAINING" = "False" ] || [ "$IS_TRAINING" = "false" ]; then
    DEBUG_ROLLOUT_ONLY="--debug-rollout-only"
    echo "Inference-only mode enabled (IS_TRAINING=$IS_TRAINING)"
else
    DEBUG_ROLLOUT_ONLY=""
    echo "Normal training mode (IS_TRAINING=$IS_TRAINING)"
fi

# Create checkpoint directory
mkdir -p "${CKPT_DIR}"

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
# pkill -9 python
sleep 3
pkill -9 ray
# pkill -9 python

set -ex

export PYTHONBUFFERED=16
export TOKENIZERS_PARALLELISM=false

# NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l || echo 0)
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
echo "NVLINK_COUNT: $NVLINK_COUNT"
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source scripts/models/deepseek-r1-distill-qwen-1.5B.sh

CKPT_ARGS=(
   --hf-checkpoint "${SAVE_SHM_DIR}/${MODEL_NAME}"
   --ref-load "${SAVE_SHM_DIR}/${MODEL_NAME}_torch_dist"
   --load "${CKPT_DIR}/"
   --save "${CKPT_DIR}/"
   --save-interval 10
)

ROLLOUT_ARGS=(
  --disable-rollout-global-dataset
  --evolving-gym
  --evolving-gym-initial-program "${INITIAL_PROGRAM}"
  --evolving-gym-evaluator-file "${EVALUATOR_FILE}"
  --evolving-gym-config-path "${CONFIG_YAML}"
  --evolving-gym-max-concurrent-evals 128
  --evolving-gym-log-prompts
  --evolving-gym-record
  --evolving-gym-record-dir "${RECORD_PATH}"
  --evolving-gym-lazy-output-penalty-level "${LAZY_OUTPUT_PENALTY}"
  --evolving-gym-seed ${SEED}
  --evolving-gym-reward-process-type "${REWARD_PROCESS_TYPE}"

  --apply-chat-template

  --rm-type evolving-gym
  --reward-key reward

  --num-rollout 10000
  --rollout-batch-size 32
  --n-samples-per-prompt 16
  --rollout-max-response-len 16384
  --rollout-temperature 1.0

  --over-sampling-batch-size 32
  # --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
  --partial-rollout

  --num-steps-per-rollout 1
  --wandb-always-use-train-step
  --balance-data
)


PERF_ARGS=(
  --tensor-model-parallel-size 2
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1

  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1

  --use-dynamic-batch-size
  --max-tokens-per-gpu 9216
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
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

WANDB_ARGS=(
  --use-wandb
  --wandb-team ${WANDB_ENTITY}
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${RUN_NAME}"
  --wandb-key "${WANDB_API_KEY}"
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
  ${DEBUG_ROLLOUT_ONLY}
  --seed ${SEED}
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

# Start Ray (training/inference separation: don't use --colocate; use train_async.py)
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


# Disable Triton
export TRITON_DISABLE=1



export FAST_MOUNT=$SAVE_PATH/fast_mount
export HF_DATASETS_CACHE=$FAST_MOUNT/hf/datasets
export DATASETS_CACHE=$HF_DATASETS_CACHE
export DATASETS_TMPDIR=$FAST_MOUNT/tmp
export PYARROW_TMP_DIR=$FAST_MOUNT/tmp

mkdir -p "$HF_DATASETS_CACHE" "$DATASETS_TMPDIR"
echo "[disk] HF_DATASETS_CACHE=$HF_DATASETS_CACHE TMPDIR=$TMPDIR"


# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="$(cat <<JSON
{
  "env_vars": {
    "PYTHONPATH": "/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "HF_HOME": "${HF_HOME}",
    "HUGGINGFACE_HUB_CACHE": "${HUGGINGFACE_HUB_CACHE}",
    "TRANSFORMERS_CACHE": "${TRANSFORMERS_CACHE}",
    "HF_DATASETS_CACHE": "${HF_DATASETS_CACHE}",
    "DATASETS_CACHE": "${DATASETS_CACHE}",
    "DATASETS_TMPDIR": "${DATASETS_TMPDIR}",
    "PYARROW_TMP_DIR": "${PYARROW_TMP_DIR}",
    "TMPDIR": "${TMPDIR}",
    "WANDB_CACHE_DIR": "${WANDB_CACHE_DIR}",
    "WANDB_DIR": "${WANDB_DIR}",
    "WANDB_GROUP": "${RUN_NAME}",
    "TRITON_DISABLE": "1"
  }
}
JSON
)"

echo "RUNTIME_ENV_JSON = $RUNTIME_ENV_JSON"

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
  ${DISTRIBUTED_ARGS[@]} \
  ${WANDB_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${SGLANG_ARGS[@]} \
  ${MISC_ARGS[@]}
