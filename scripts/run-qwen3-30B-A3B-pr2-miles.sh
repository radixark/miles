#!/bin/bash
# Predictive Routing Replay (PR²) on Qwen3-30B-A3B with Miles enhancements.
#
# Same algorithm core as run-qwen3-30B-A3B-pr2-paper.sh, additionally turning
# on the Miles stabilization layer (depth-aware layer-scale gating + boundary-
# margin loss reweighting). Empirically more stable than paper-faithful PR² on
# this setup.

# Base directory where checkpoints + datasets live. Override to point at
# your own data layout (same pattern as scripts/run-qwen3-235B-A22B.sh).
BASE_FOLDER=${BASE_FOLDER:-/root}

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-30B-A3B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${BASE_FOLDER}/Qwen3-30B-A3B-Base
   --ref-load      ${BASE_FOLDER}/Qwen3-30B-A3B-Base_torch_dist/
   --load          ${BASE_FOLDER}/Qwen3-30B-A3B-Base_miles_pr2_miles
   --save          ${BASE_FOLDER}/Qwen3-30B-A3B-Base_miles_pr2_miles
   --save-interval 10
)

ROLLOUT_ARGS=(
   --prompt-data ${BASE_FOLDER}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type dapo
   --num-rollout 150
   --rollout-batch-size 32
   --n-samples-per-prompt 16
   --num-steps-per-rollout 2
   --rollout-max-prompt-len 1024
   --rollout-max-response-len 1024
   --rollout-temperature 1.0

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data aime24 ${BASE_FOLDER}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-prompt-len 1024
   --eval-max-response-len 1024
   --eval-temperature 1
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 2e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project miles-pr2
   # --wandb-group qwen3-30B-A3B-pr2-miles
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.5
   --sglang-max-running-requests 512
   --sglang-server-concurrency 512
)

# ============================================================================
#       Predictive Routing Replay — paper core + Miles stabilization
# ============================================================================
PR2_ARGS=(
   --use-miles-router
   --use-routing-replay

   --enable-predictive-routing-replay
   --bias-predictor-loss-type kl-post
   --bias-predictor-lr-mult   1e2
   --predictive-storage-dtype fp32

   # Token-budget caps to keep cached predictive tensors inside CPU offload
   # (sized for max_response_len=1024 + 16 samples per prompt).
   --predictive-downsample-batch-size     2
   --predictive-downsample-max-len-limit  4096
   --predictive-max-total-tokens          2048

   # Miles stabilization layer: depth-aware layer-scale gating + boundary-
   # margin loss reweighting. See docs/predictive-routing-replay.md §4.2.
   --predictive-layer-scale-schedule sqrt_decay
   --predictive-layer-scale-min      0.5
   --predictive-boundary-loss-max-weight 2.0
   --predictive-boundary-loss-min-margin 1e-3
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend auto
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

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
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   --keep-old-actor \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${PR2_ARGS[@]} \
   ${MISC_ARGS[@]}
