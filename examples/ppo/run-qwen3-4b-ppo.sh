#!/bin/bash

# PPO (actor + critic) with the Megatron backend on a single node.
#
# Unlike GRPO, which derives its baseline from a group of samples per prompt, PPO trains a
# separate value model (the critic) and turns rewards into advantages with GAE. The critic
# shares the actor's train GPUs, so a PPO run needs no extra GPUs over the GRPO equivalent --
# it trades GPU memory (a second model) for not needing a large --n-samples-per-prompt.
#
# The parallelism, GPU count and PPO flags here follow tests/e2e/megatron/test_qwen3_4B_ppo.py,
# which runs in CI. See README.md for which values are CI-verified and which are conventional
# starting points you should tune.

set -ex

export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# actor world size = NUM_GPUS = TP * PP * CP below. The critic inherits this same shape.
NUM_GPUS=4

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MODEL_ARGS_LINE="$(python3 "${SCRIPT_DIR}/../../miles/utils/external_utils/model_args_utils.py" "qwen3-4B")" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-4B
   --ref-load /root/Qwen3-4B_torch_dist
   --load /root/Qwen3-4B_miles/
   --save /root/Qwen3-4B_miles/
   --save-interval 20
   # --critic-load and --critic-lr default to --load and --lr.
   # --critic-save defaults to --save with a '_critic' suffix, i.e. /root/Qwen3-4B_miles_critic,
   # a sibling dir so the two models do not clobber each other's iteration tracker.
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 0.8

   --global-batch-size 32
   --balance-data
)

RM_ARGS=(
   --rm-type deepscaler
)

EVAL_ARGS=(
   # --eval-interval 20
   # --eval-prompt-data aime24 /root/aime-2024/aime-2024.jsonl
   # --n-samples-per-eval-prompt 1
   # --eval-max-response-len 16384
   # --eval-top-k 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 2
   --context-parallel-size 2

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

PPO_ARGS=(
   # Selecting ppo is what creates the critic; everything else below is tuning.
   --advantage-estimator ppo

   --critic-lr 1e-5             # critic usually wants a larger lr than the actor
   --num-critic-only-steps 1    # value-function warmup: actor frozen for this many rollout steps
   --normalize-advantages

   # Reward-level KL (--kl-coef) is rejected with ppo: the critic trains before the actor and
   # never sees ref log probs, so its value targets would silently omit that penalty.
   # Use loss-level KL instead.
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type k1
   --kl-coef 0.00

   --entropy-coef 0.00
   --eps-clip 0.2
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
   # --use-wandb
   # --wandb-project miles-dev
   # --wandb-group qwen3-4B-ppo
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.8
   --sglang-max-running-requests 512
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${PPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}

####clear after training
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
