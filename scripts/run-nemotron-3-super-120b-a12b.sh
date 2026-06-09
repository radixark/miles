#!/bin/bash

# Two-node (2x 8x H200, 16 GPUs) launcher for Nemotron-3-Super-120B-A12B.
# Usage on each pod:
#   head:   bash run-nemotron-3-super-120b-a12b.sh head   <head_pod_ip>
#   worker: bash run-nemotron-3-super-120b-a12b.sh worker <head_pod_ip>

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

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
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
source "${SCRIPT_DIR}/models/nemotron-3-super-120b-a12b.sh"

MODELS_DIR=${MODELS_DIR:-/cluster_public/miles_data/models}
DATASETS_DIR=${DATASETS_DIR:-/cluster_public/miles_data/datasets}

CKPT_ARGS=(
   --hf-checkpoint $MODELS_DIR/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
   --ref-load $MODELS_DIR/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
   --save $MODELS_DIR/nemotron-3-super-120b-a12b_miles
   --save-interval 20
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data $DATASETS_DIR/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 10
   --rollout-batch-size 32
   --n-samples-per-prompt 4
   --rollout-max-response-len 1024
   --rollout-temperature 1

   --global-batch-size 128
   --balance-data
)

EVAL_ARGS=(
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
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
   # --wandb-project miles-dev
   # --wandb-group nemotron-3-super-120b-a12b
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   # Replay the exact rollout routing during training forward so
   # train logprobs match rollout logprobs (needed for MoE).
   --use-miles-router
   --use-rollout-routing-replay
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend auto
)

# launch the master node of ray in container
export MASTER_ADDR=${HEAD_IP}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# wait for the worker to join so the cluster has 16 GPUs before submitting
echo "Waiting for ray cluster to have 16 GPUs..."
for i in $(seq 1 120); do
    if ray status 2>/dev/null | grep -q '16.0 GPU'; then
        echo "[ray] cluster ready: 16 GPUs"
        break
    fi
    sleep 5
done
ray status

# Build the runtime environment JSON with proper variable substitution
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
   --actor-num-nodes 2 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 16 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
