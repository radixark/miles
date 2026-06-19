#!/bin/bash
# ============================================================================
# GLM-5.1 (6-layer toy) GRPO LoRA — single node, 4x H200, colocated.
#
# GLM-5.1 is MoE + MLA + DSA (DeepSeek Sparse Attention). LoRA trains via the
# Megatron-Bridge path (--megatron-to-hf-mode bridge); the GLM-5.1 "dsa"
# experimental-attention-variant spec is registered for that path by miles
# (bridge_lora_helpers.py monkey-patch), so NO --spec is needed here.
#
# Verified e2e (rollout -> train -> save): TRAIN EXIT 0 + PEFT adapter saved.
# Modeled on examples/lora/run-qwen3-4B-megatron-lora.sh (single-node skeleton)
# and run-kimi-k25-megatron-lora.sh (MoE+MLA LoRA conventions).
#
# Two GLM-5.1/DSA specifics baked in below:
#   * --target-modules = explicit list EXCLUDING the 3 DSA indexer modules
#     (wq_b/wk/weights_proj); the indexer stays a code capability, not trained here.
#   * --qkv-format bshd (+ --micro-batch-size 1, NO --use-dynamic-batch-size):
#     megatron-core's DSA core-attention needs a 4D (bshd) query; the default
#     thd packing yields a 3D query -> "not enough values to unpack".
# ============================================================================

# Clean up stale processes from a previous run.
# (Use ray stop + name-based pkill; do NOT `pkill -f sglang.launch_server` —
#  that pattern also matches a wrapping launch shell and self-kills it.)
pkill -9 sglang 2>/dev/null
sleep 2
ray stop --force 2>/dev/null
sleep 3

set -ex

export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export HF_HOME=${HF_HOME:-/cluster-storage/models}
export PYTHONBUFFERED=16

# 6-layer GLM-5.1 checkpoint (HF cache snapshot) + where to save the LoRA adapter.
HF_CHECKPOINT=${HF_CHECKPOINT:-/cluster-storage/models/models--jybsuper--GLM-5.1-6layer/snapshots/1ea546e4990647bc651e94953a57dfaa9eedb576}
SAVE=${SAVE:-/personal/checkpoints/glm51-lora-e2e}
rm -rf "$SAVE"

# MODEL_ARGS for GLM-5.1 6-layer (3 dense + 3 MoE) come from the per-model registry.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/glm5-744B-A40B_6layer.sh"

CKPT_ARGS=(
   --hf-checkpoint "$HF_CHECKPOINT"
   --megatron-to-hf-mode bridge
)

LORA_ARGS=(
   --lora-rank 16
   --lora-alpha 32
   --lora-dropout 0.0
   # explicit list: standard attn + MLA + MLP/MoE, EXCLUDING the DSA indexer (wq_b/wk/weights_proj)
   --target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,q_a_proj,kv_a_proj_with_mqa,q_b_proj,kv_b_proj"
   # --- optional MoE expert-LoRA fidelity (see run-kimi-k25-megatron-lora.sh); verify on GLM-5.1: ---
   # --experts-shared-outer-loras
   # --lora-base-cpu-backup
   # --sglang-lora-backend triton            # kimi: "!!! must for moe-lora !!!"
   # --sglang-lora-use-virtual-experts
)

ROLLOUT_ARGS=(
   --prompt-data /root/datasets/gsm8k/train.parquet
   --input-key messages
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 1
   --rollout-batch-size 4
   --n-samples-per-prompt 4
   --rollout-max-response-len 256
   --rollout-temperature 1.0
   --global-batch-size 16
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# 4-GPU layout: TP4 x EP4 (DP1). bshd (4D query) is REQUIRED for DSA and forbids
# --use-dynamic-batch-size, hence --micro-batch-size.
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 4
   --expert-tensor-parallel-size 1
   --qkv-format bshd
   --micro-batch-size 1
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2          # rollout tp=2
   --sglang-mem-fraction-static 0.5
   --sglang-cuda-graph-max-bs 64
   --sglang-moe-runner-backend triton
   --sglang-disable-shared-experts-fusion
   --sglang-reasoning-parser glm45
   --sglang-tool-call-parser glm47
)

WANDB_ARGS=(
   --use-wandb
   --wandb-host https://wandb.ai/
   --wandb-project miles-glm51-lora
   --wandb-group glm5.1-6layer-lora
)

SAVE_ARGS=(
   --save-interval 1
   --save "$SAVE"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --calculate-per-token-loss
   --use-miles-router
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "$GPUS_PER_NODE" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "HF_HOME": "/cluster-storage/models",
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "$GPUS_PER_NODE" \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${LORA_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${SAVE_ARGS[@]} \
   ${MISC_ARGS[@]}
