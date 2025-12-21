#!/bin/bash
#
# Ray-free SFT Training Script
#
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# FIXME(f.srambical): this is hardcoded for now
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE}
NUM_NODES=${SLURM_JOB_NUM_NODES}
NODE_RANK=${SLURM_NODEID}
MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)}

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

RUN_ID=${RUN_ID:-"run_$(date +%Y%m%d_%H%M%S)"}
LOAD_SAVE_PATH="/fast/project/HFMI_SynergyUnit/tab_model/huggingface/shared_data/${RUN_ID}/checkpoints"

CKPT_ARGS=(
   --hf-checkpoint /fast/project/HFMI_SynergyUnit/tab_model/huggingface/Qwen3-0.6B
   --load /fast/project/HFMI_SynergyUnit/tab_model/huggingface/Qwen3-0.6B
   --ref-load /fast/project/HFMI_SynergyUnit/tab_model/huggingface/Qwen3-0.6B
)

SFT_ARGS=(
   --rollout-function-path miles.rollout.sft_rollout.generate_rollout
   --prompt-data /fast/project/HFMI_SynergyUnit/tab_model/huggingface/nemo_hf_part_jsonl_4k_tokens.parquet
   --input-key messages
   --apply-chat-template
   --rollout-shuffle
   --num-epoch 3
   --rollout-batch-size 16
   --global-batch-size 16

   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --num-rollout 2000
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style WSD
   --lr-wsd-decay-style linear
   --lr-warmup-iters 100
   --lr-decay-iters 2000
   --lr-wsd-decay-iters 500
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project crowd-pilot-miles
   --wandb-team instant-uv
   --wandb-group qwen3-0.6b-sft-torchrun
)

TRAIN_BACKEND_ARGS=(
   --train-backend fsdp
   --update-weight-buffer-size 536870912
   --gradient-checkpointing
   --attn-implementation flash_attention_3
)

PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

MISC_ARGS=(
   --rollout-max-context-len 8192
   --rollout-max-prompt-len 8000
   --rollout-max-response-len 8192
   --dump-details /fast/project/HFMI_SynergyUnit/tab_model/huggingface/shared_data/qwen3-600M-fsdp-1116-noref/dump_details
)

torchrun \
    --nproc_per_node=${GPUS_PER_NODE} \
    --nnodes=${NUM_NODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT:-29500} \
    train_sft.py \
    ${CKPT_ARGS[@]} \
    ${SFT_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${WANDB_ARGS[@]} \
    ${TRAIN_BACKEND_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${MISC_ARGS[@]}
