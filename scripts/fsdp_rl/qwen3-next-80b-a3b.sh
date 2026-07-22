#!/bin/bash
# Qwen3-Next-80B-A3B GatedDeltaNet MoE (single node + offload)
# GPUs: 1x8=8  (CPU offload on; sglang TP 2/engine; sglang_mem/max_tokens from common.sh)
export RUN_ID=qwen3-next-80b-a3b
export MODEL=Qwen3-Next-80B-A3B
export NNODES=1 GPUS_PER_NODE=8 ROLLOUT_GPUS_PER_ENGINE=2
source "$(dirname "$0")/common.sh"
