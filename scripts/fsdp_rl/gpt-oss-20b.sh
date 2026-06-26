#!/bin/bash
# gpt-oss-20B MoE
# GPUs: 1x8=8  (CPU offload on; sglang TP 1/engine; sglang_mem/max_tokens from common.sh)
export RUN_ID=gpt-oss-20b
export MODEL=gpt-oss-20b-bf16
export NNODES=1 GPUS_PER_NODE=8 ROLLOUT_GPUS_PER_ENGINE=1
source "$(dirname "$0")/common.sh"
