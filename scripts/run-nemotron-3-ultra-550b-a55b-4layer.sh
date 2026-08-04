#!/bin/bash
# Single-node (8x H200) smoke test for the 4-layer Nemotron-3-Ultra slice.
#
# Same code path as run-nemotron-3-ultra-550b-a55b.sh (AutoBridge + miles
# NemotronHBridge MoE/latent shim, colocated SGLang rollout), sized down to one
# node so the Megatron -> SGLang weight sync can be verified end to end with
# --check-weight-update-equal.
#
# Prereq: build the slice once (see scripts/models/nemotron-3-ultra-550b-a55b-4layer.sh):
#   python cluster_scripts/debug_tool_set/checkpoint/prune_nemotron_h.py \
#       --src <full HF ckpt> --dst $MODELS_DIR/Nemotron-3-Ultra-4layer --layers 0,1,7,8
#
# Usage:
#   bash scripts/run-nemotron-3-ultra-550b-a55b-4layer.sh

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/nemotron-3-ultra-550b-a55b-4layer.sh"
cd "${SCRIPT_DIR}/.."

MODELS_DIR=${MODELS_DIR:-/scratch/nemotron_data/models}
DATASETS_DIR=${DATASETS_DIR:-/scratch/nemotron_data/datasets}
HF=${HF:-$MODELS_DIR/Nemotron-3-Ultra-4layer}
MEGATRON_PATH=${MEGATRON_PATH:-/mirror/nemotron/megatron}

CKPT_ARGS=(
   --hf-checkpoint $HF
   --ref-load $HF
   --save $MODELS_DIR/nemotron-3-ultra-4layer_miles
   --save-interval 10000        # never inside a smoke test
   --no-save-optim
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data $DATASETS_DIR/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 2
   --rollout-batch-size 8
   --n-samples-per-prompt 2
   --rollout-max-response-len 256
   --rollout-temperature 1
   --global-batch-size 16
   --balance-data
)

PERF_ARGS=(
   # 8 GPUs. Mamba n_groups=8 caps attention/mamba TP at 8; TP=1 keeps the
   # 4-layer slice trivially valid and leaves all 8 ranks for expert parallel
   # (512 experts / EP8 = 64 per rank).
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 1024
   --log-probs-chunk-size 128
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-ep-size 8
   # DP-attention with dp=2 -> attn_tp = 8/2 = 4, which divides Mamba n_groups=8.
   --sglang-dp-size 2
   --sglang-enable-dp-attention
   --sglang-mem-fraction-static 0.6
)

CHECK_ARGS=(
   # Snapshot SGLang's loaded weights, poison them, then assert the Megatron ->
   # SGLang sync restores every tensor exactly. This is what the run is for.
   --check-weight-update-equal
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend auto
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK: the nemotron DP-attention path uses
# existing kernels; skip the blanket sgl-kernel version guard.
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --colocate \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${CHECK_ARGS[@]} \
   ${MISC_ARGS[@]}
