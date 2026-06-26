#!/bin/bash
# Kimi-K2.5 ~1T MoE -- HUGE, multi-node; pure-FSDP impractical, sized as a template
# GPUs: 16x8=128  (CPU offload on; sglang TP 8/engine; sglang_mem/max_tokens from common.sh)
export RUN_ID=kimi-k2.5
export MODEL=Kimi-K2.5
export NNODES=16 GPUS_PER_NODE=8 ROLLOUT_GPUS_PER_ENGINE=8
source "$(dirname "$0")/common.sh"
