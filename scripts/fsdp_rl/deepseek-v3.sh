#!/bin/bash
# DeepSeek-V3 671B MoE -- HUGE, multi-node; pure-FSDP aggressive (no EP/PP)
# GPUs: 8x8=64  (CPU offload on; sglang TP 8/engine; sglang_mem/max_tokens from common.sh)
export RUN_ID=deepseek-v3
export MODEL=DeepSeek-V3
export NNODES=8 GPUS_PER_NODE=8 ROLLOUT_GPUS_PER_ENGINE=8
source "$(dirname "$0")/common.sh"
