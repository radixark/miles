#!/usr/bin/env bash
# Inkling-Small (276B) full-parameter GRPO on 4 nodes x 8 H200:
# TP4 SP PP8 EP4, ctx 4096 / response 2048, lr 6e-6, rollout 64 prompts x 8 samples.
# Assumes staged HF weights + torch_dist checkpoint and a running Ray cluster
# (see docs/models/thinkingmachines/inkling-small.md).
set -euxo pipefail
cd "$(dirname "$0")/.."

# Multi-node: point the launcher at the existing Ray cluster.
export MILES_SCRIPT_EXTERNAL_RAY=1
export MASTER_ADDR=${MASTER_ADDR:?set to the head node IP}
# Pick the NIC that carries your east-west traffic (bond0 on the H200 fleet).
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond0}

MODEL_DIR=${MODEL_DIR:-/root/models}
HF_CKPT=${HF_CKPT:-${MODEL_DIR}/Inkling-Small}
TORCH_DIST=${TORCH_DIST:-${MODEL_DIR}/Inkling-Small_torch_dist}

python3 scripts/run_inkling.py train \
  --model-name Inkling-Small --train-mode full --task dapo_math \
  --num-nodes 4 --num-gpus-per-node 8 \
  --hf-checkpoint "$HF_CKPT" \
  --torch-dist "$TORCH_DIST" \
  --data-dir "${DATA_DIR:-/root/datasets}" \
  --lr 6e-6 \
  --rollout-batch-size 64 --global-batch-size 128 \
  --sglang-context-length 4096 --rollout-max-response-len 2048 \
  --megatron-path "${MEGATRON_PATH:-/root/Megatron-LM}" \
  --extra-args "--offload-train-target cpu --sglang-mem-fraction-static 0.65 \
    --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer" \
  "$@"
