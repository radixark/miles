#!/bin/bash
# Nemotron-3-Nano-4B dense (Mamba2 hybrid)
# GPUs: 1x8=8  (CPU offload on; sglang TP 1/engine; sglang_mem/max_tokens from common.sh)
export RUN_ID=nemotron3-nano-4b
export MODEL=NVIDIA-Nemotron-3-Nano-4B-BF16
export NNODES=1 GPUS_PER_NODE=8 ROLLOUT_GPUS_PER_ENGINE=1
source "$(dirname "$0")/common.sh"
