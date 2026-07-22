#!/bin/bash
# Qwen3-4B dense (validated end-to-end)
# GPUs: 1x8=8  (CPU offload on; sglang TP 1/engine; sglang_mem/max_tokens from common.sh)
export RUN_ID=qwen3-4b
export MODEL=Qwen3-4B
export NNODES=1 GPUS_PER_NODE=8 ROLLOUT_GPUS_PER_ENGINE=1
source "$(dirname "$0")/common.sh"
