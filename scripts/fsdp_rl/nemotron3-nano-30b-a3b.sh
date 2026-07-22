#!/bin/bash
# Nemotron-3-Nano-30B-A3B hybrid MoE (nemotron_h)
# GPUs: 1x8=8  (CPU offload on; sglang TP 2/engine; sglang_mem/max_tokens from common.sh)
export RUN_ID=nemotron3-nano-30b-a3b
export MODEL=NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
export NNODES=1 GPUS_PER_NODE=8 ROLLOUT_GPUS_PER_ENGINE=2
source "$(dirname "$0")/common.sh"
