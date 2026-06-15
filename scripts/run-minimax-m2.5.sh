#!/bin/bash
# Launcher for MiniMax-M2.5 RL training, aligned with THUDM/slime#1929.
#
# ============================================================================
# Prerequisites (run once inside the docker container 9e16408bd232):
# ----------------------------------------------------------------------------
#   docker exec -it 9e16408bd232 bash
#   cd /home/yangchengyi/data/miles
#
#   # Optional: HF -> torch_dist for a local Megatron ckpt layout.
#   # Weight naming is handled by miles_plugins.mbridge.minimax_m2.MiniMaxM2Bridge.
#   # ============================================================================
#   bash scripts/run-minimax-m2.5.sh

# Cleanup leftover servers from a previous run
pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3
pkill -9 ray || true
pkill -9 python || true

set -ex

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/minimax-m2.5.sh"

# Host -> container shared mount of large assets.
BASE_FOLDER=${BASE_FOLDER:-/home/yangchengyi/data/models}
DATA_FOLDER=${DATA_FOLDER:-/home/yangchengyi/data/datasets}
HF_CHECKPOINT=${HF_CHECKPOINT:-${BASE_FOLDER}/MiniMax-M2.5}
REF_LOAD=${REF_LOAD:-${BASE_FOLDER}/MiniMax-M2.5_torch_dist}
TRAIN_CKPT=${TRAIN_CKPT:-${BASE_FOLDER}/MiniMax-M2.5_miles/}
AIME_DATA=${AIME_DATA:-${DATA_FOLDER}/aime-2024/aime-2024.jsonl}
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-16}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}
TP=${TP:-2}
PP=${PP:-2}
CP=${CP:-1}
EP=${EP:-4}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-16}
SGLANG_EP_SIZE=${SGLANG_EP_SIZE:-16}

CKPT_ARGS=(
   --hf-checkpoint ${HF_CHECKPOINT}
   --ref-load ${REF_LOAD}
   --load ${TRAIN_CKPT}
   --save ${TRAIN_CKPT}
   --save-interval 20
   --megatron-to-hf-mode raw
   --model-name minimax_m2
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_FOLDER}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 128
   --n-samples-per-prompt 8
   --rollout-max-response-len 32768
   --rollout-temperature 1

   --over-sampling-batch-size 256
   --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

   --num-steps-per-rollout 4
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime ${AIME_DATA}
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 32768
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TP}
   --sequence-parallel
   --pipeline-model-parallel-size ${PP}
   --context-parallel-size ${CP}
   --expert-model-parallel-size ${EP}
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
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

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project miles-dev
   # --wandb-group minimax-m2.5
   # --wandb-key ${WANDB_KEY}
)

TB_ARGS=(
   --use-tensorboard
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
   --sglang-mem-fraction-static 0.7
   --sglang-ep-size ${SGLANG_EP_SIZE}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# Master node IP for ray (single-node defaults to localhost)
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"no_proxy\": \"localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes ${ACTOR_NUM_NODES} \
   --actor-num-gpus-per-node ${ACTOR_NUM_GPUS_PER_NODE} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${TB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
