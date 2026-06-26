#!/bin/bash
# Qwen3-30B-A3B MoE (qwen3_moe)
# GPUs: 1x8=8  (CPU offload on; sglang TP 2/engine; sglang_mem/max_tokens from common.sh)
export RUN_ID=qwen3-30b-a3b
export MODEL=Qwen3-30B-A3B
export NNODES=1 GPUS_PER_NODE=8 ROLLOUT_GPUS_PER_ENGINE=2
source "$(dirname "$0")/common.sh"
