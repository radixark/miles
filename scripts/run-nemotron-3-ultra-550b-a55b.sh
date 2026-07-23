#!/bin/bash

# 16-node (16x 8x H200, 128 GPUs) launcher for Nemotron-3-Ultra-550B-A55B GRPO RL.
# Colocate (training + SGLang rollout share GPUs).
# Usage on each pod:
#   head:   bash run-nemotron-3-ultra-550b-a55b.sh head   <head_pod_ip>
#   worker: bash run-nemotron-3-ultra-550b-a55b.sh worker <head_pod_ip>
#
# Prereq: run scripts/convert-nemotron-3-ultra-550b-hf-to-dist.sh once to produce
# the torch_dist checkpoint ($DIST below); --load uses it to skip the slow
# per-run 512-expert HF bridge. --hf-checkpoint still points at HF for the
# tokenizer + the SGLang rollout engine.

ROLE=${1:?Usage: $0 <head|worker> <head_pod_ip>}
HEAD_IP=${2:?Usage: $0 <head|worker> <head_pod_ip>}

cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

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
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Worker just joins the head's ray cluster and blocks.
if [[ "$ROLE" == "worker" ]]; then
    for i in $(seq 1 60); do
        if nc -z "$HEAD_IP" 6379 2>/dev/null; then break; fi
        echo "waiting for head $HEAD_IP:6379 ..."
        sleep 5
    done
    ray start --address="${HEAD_IP}:6379" --num-gpus=8 --disable-usage-stats --block
    exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/nemotron-3-ultra-550b-a55b.sh"

MODELS_DIR=${MODELS_DIR:-/cluster_public/miles_data/models}
DATASETS_DIR=${DATASETS_DIR:-/cluster_public/miles_data/datasets}
# Public HF repo (https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16).
# Override HF=<local path> to use a pre-downloaded copy on shared storage.
HF=${HF:-nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16}
DIST=${DIST:-$MODELS_DIR/nemotron-3-ultra-550b-a55b_torch_dist}

CKPT_ARGS=(
   --hf-checkpoint $HF          # tokenizer + SGLang rollout
   --load $DIST                 # Megatron native load (skip per-run HF bridge)
   --ref-load $DIST
   --save $MODELS_DIR/nemotron-3-ultra-550b-a55b_miles
   --save-interval 50
   --no-save-optim              # weights-only
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data $DATASETS_DIR/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 30
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 128
   --balance-data
)

PERF_ARGS=(
   # TP8 PP4 EP32 ETP1 (DP4) = 128. Mamba n_groups=8 needs attention/mamba TP<=8;
   # see SGLANG_ARGS (DP-attention) for the rollout side.
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 4
   --context-parallel-size 1
   --expert-model-parallel-size 32
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

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project miles-nemotron-ultra
   # --wandb-group nemotron-3-ultra-550b-a55b
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   # 550B (~1.1TB bf16) does not fit one 8-GPU engine. Use a 32-GPU engine with
   # EP=32 for experts + DP-attention (dp=4) so attention/mamba run at
   # attn_tp = 32/4 = 8 (divides Mamba n_groups=8).
   --rollout-num-gpus-per-engine 32
   --sglang-ep-size 32
   --sglang-dp-size 4
   --sglang-enable-dp-attention
   --sglang-mem-fraction-static 0.7
   # Rollout routing-replay (--use-miles-router --use-rollout-routing-replay) is
   # NOT yet enabled for the 108-layer Ultra: the routing capturer shape needs a
   # fix for per-layer top-22 routing under DP-attention. Train/rollout logprob
   # diff is still ~0.01 without it.
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend auto
)

# launch the master node of ray in container
export MASTER_ADDR=${HEAD_IP}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# wait for all workers to join so the cluster has 128 GPUs before submitting
echo "Waiting for ray cluster to have 128 GPUs..."
for i in $(seq 1 240); do
    if ray status 2>/dev/null | grep -q '128.0 GPU'; then
        echo "[ray] cluster ready: 128 GPUs"
        break
    fi
    sleep 5
done
ray status

# SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK: the nemotron DP-attention path uses
# existing kernels; skip the blanket sgl-kernel version guard.
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 16 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 128 \
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
