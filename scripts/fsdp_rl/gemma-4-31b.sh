#!/bin/bash
# Gemma-4-31B dense
# GPUs: 1x8=8  (CPU offload on; sglang TP 2/engine; sglang_mem/max_tokens from common.sh)
export RUN_ID=gemma-4-31b
export MODEL=gemma-4-31b-it
export NNODES=1 GPUS_PER_NODE=8 ROLLOUT_GPUS_PER_ENGINE=2
source "$(dirname "$0")/common.sh"
