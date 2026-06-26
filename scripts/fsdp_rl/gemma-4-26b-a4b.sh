#!/bin/bash
# Gemma-4-26B-A4B MoE
# GPUs: 1x8=8  (CPU offload on; sglang TP 2/engine; sglang_mem/max_tokens from common.sh)
export RUN_ID=gemma-4-26b-a4b
export MODEL=gemma-4-26b-a4b-it
export NNODES=1 GPUS_PER_NODE=8 ROLLOUT_GPUS_PER_ENGINE=2
source "$(dirname "$0")/common.sh"
