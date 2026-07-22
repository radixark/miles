#!/bin/bash
# GLM-4.7-Flash MoE (glm4_moe_lite, fp32-master)
# GPUs: 1x8=8  (CPU offload on; sglang TP 2/engine; sglang_mem/max_tokens from common.sh)
export RUN_ID=glm4.7-flash
export MODEL=GLM-4.7-Flash
export NNODES=1 GPUS_PER_NODE=8 ROLLOUT_GPUS_PER_ENGINE=2
source "$(dirname "$0")/common.sh"
