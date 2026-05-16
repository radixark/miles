#!/bin/bash
# Two-node (2x 8x H200, 16 GPUs) RL smoke test for Nemotron-3-Super-120B-A12B MoE.
# Variant of scripts/run-nemotron-3-super-120b-a12b.sh scaled to 2 nodes.
# Parallelism is TP=4xPP=2xEP=8 (DP=2): TP+PP stay intra-node via NVLink and
# DP=2 fans rollouts/grads across the two nodes. Memory is just inside the
# per-node H200 envelope when colocating sglang rollout with the actor.
#
# Usage (run on each pod, head first or in parallel):
#   head:   bash run-nemotron-3-super-120b-a12b-2node.sh head   <head_pod_ip>
#   worker: bash run-nemotron-3-super-120b-a12b-2node.sh worker <head_pod_ip>
# The worker process blocks while joined to the ray cluster; the head submits
# the training job and tails its logs.

set -ex

ROLE=${1:?Usage: $0 <head|worker> <head_pod_ip>}
HEAD_IP=${2:?Usage: $0 <head|worker> <head_pod_ip>}

cd "$(dirname -- "${BASH_SOURCE[0]}")/.."  # repo root: train.py lives here

pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3
pkill -9 ray || true
pkill -9 python || true

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK"

if [[ "$ROLE" == "worker" ]]; then
  # Block until the head's ray GCS is reachable, then join and stay attached.
  for i in $(seq 1 60); do
    if nc -z "$HEAD_IP" 6379 2>/dev/null; then break; fi
    echo "waiting for head $HEAD_IP:6379 ..."
    sleep 5
  done
  ray start --address="${HEAD_IP}:6379" --num-gpus=8 --disable-usage-stats --block
  exit 0
fi

# Head from here on.
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

WANDB_ARGS=()

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   # Replay the exact rollout routing during training forward so
   # train logprobs match rollout logprobs (needed for MoE).
   --use-miles-router
   --use-rollout-routing-replay
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend auto
)

export MASTER_ADDR=${HEAD_IP}
ray start --head --node-ip-address ${HEAD_IP} --num-gpus 8 \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

echo "Waiting for ray cluster to have 16 GPUs..."
for i in $(seq 1 120); do
  if ray status 2>/dev/null | grep -q '16.0 GPU'; then
    echo "[ray] cluster ready: 16 GPUs"
    break
  fi
  sleep 5
done
ray status

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
   --colocate \
   --actor-num-nodes 2 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 16 \
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
