#!/bin/bash

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
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"



SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

RUN_ID=${RUN_ID:-"run_$(date +%Y%m%d_%H%M%S)"}
LOAD_SAVE_PATH="/root/shared_data/${RUN_ID}/checkpoints"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-4B
   --load /root/Qwen3-4B
   --ref-load /root/Qwen3-4B
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --rm-type deepscaler
   --num-rollout 100
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 0.8
   --global-batch-size 64
)

GRPO_ARGS=(
   --use-kl-loss
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
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
   --wandb-project miles-dev-mcore-fsdp
   --wandb-group qwen3-4B-fsdp-1130-ref
   --wandb-key ${WANDB_API_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.75
   --sglang-decode-log-interval 1000
   --sglang-chunked-prefill-size 4096
   --sglang-attention-backend fa3
)

TRAIN_BACKEND_ARGS=(
   --train-backend fsdp
   --update-weight-buffer-size 536870912
   --gradient-checkpointing
   --attn-implementation flash_attention_3
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
)

PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

MISC_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node 8
   --colocate
   --use-fault-tolerance
   --dump-details /root/shared_data/qwen3-4B-fsdp-1116-noref/dump_details
   # --fsdp-cpu-offload
)

CUSTOM_ARGS=(
   # Enable importance sampling, details refer to the comments of compute_mis_weights in mis.py
   --use-rs
   # Aggregation level for importance sampling weights:
   # token: per-token
   # sequence: product over tokens
   # geometric: geometric mean
   --tis-level token
   --rs-level token
   # Handling mode for IS weights:
   # truncate: cap to upper bound, TIS
   # mask: zero outside [lower, upper], MIS
   # clip: clip to [lower, upper], CIS
   --tis-mode truncate
   # For clip mode, the lower bound of the IS weights.
   # For truncate mode, it will not be used.
   # If not set, it will be set to 1.0 / mis_upper_bound
   # For Geometry level, the lower bound should be 0.9999
   --tis-lower-bound 0.5
   # For truncate or clip mode, the upper bound of the IS weights
   # For Geometry level, the upper bound should be 1.0001
   --tis-upper-bound 2.0
   # Lower and upper bound for rejection sampling.
   # If not set, it will be same as tis_lower_bound and tis_upper_bound.

   # Per-token veto threshold. If any token ratio < this, zero the entire sequence weight, the sequences won't have gradient
   # Note: float number must be written with dot e.g. 1.0e-4, not 1e-4
   --rs-veto-threshold 1.0e-4
   # Batch normalization: normalize IS weights to mean=1.0 across entire batch
   # This reduces variance in gradient updates
   --tis-batch-normalize
   --custom-tis-function-path examples.infra_features.train_infer_mismatch_helper.mis.compute_mis_weights_fsdp
)

# launch the master node of ray in container - 8 GPUs for training
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats


RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"


ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${TRAIN_BACKEND_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
