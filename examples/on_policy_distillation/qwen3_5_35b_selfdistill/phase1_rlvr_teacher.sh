#!/bin/bash
# =============================================================================
# Phase 1: RLVR-train a Qwen3.5-35B-A3B *teacher* on DAPO-math (single 8xH200 node)
# =============================================================================
# Trains the base model with GRPO + format-agnostic correctness reward so it becomes
# measurably BETTER and MORE CONCISE than the base. This trained checkpoint is the
# teacher distilled into the base student in Phase 2.
#
#   Single node, 8 GPUs:  TP2 / PP1 / CP2 / EP8 / ETP1  (world=8, DP2)
#   lr 1e-5 is intentional: it makes the teacher DIVERGE from base fast enough to
#   matter within ~5-10 steps. lr 1e-6 (few steps) barely moves the weights, leaving
#   the teacher ~= base, which makes the Phase-2 OPD reverse-KL ~0 (inert).
#
# Usage:  bash phase1_rlvr_teacher.sh
# Env (override as needed):
#   MODEL_DIR   dir holding Qwen3.5-35B-A3B and Qwen3.5-35B-A3B_torch_dist
#   DATA_DIR    dir holding dapo_train.jsonl / dapo_eval.jsonl (see make_split.py)
#   OUTPUT_DIR  writable, *node-local-persistent* dir for checkpoints (see README)
#   EXAMPLE_DIR this directory (for rm.py + eval config on PYTHONPATH)
# =============================================================================
set -ex
export PYTHONUNBUFFERED=16

MODEL_DIR=${MODEL_DIR:-/cluster_public/miles_data/models}
DATA_DIR=${DATA_DIR:-/node_public/maocheng-qwen35/data}
OUTPUT_DIR=${OUTPUT_DIR:-/node_public/maocheng-qwen35/ckpt-teacher}
EXAMPLE_DIR=${EXAMPLE_DIR:-$(cd "$(dirname "$0")" && pwd)}
MILES_DIR=${MILES_DIR:-/root/miles}
RAY_ADDRESS=${RAY_ADDRESS:-http://127.0.0.1:8265}
mkdir -p "${OUTPUT_DIR}"

# Render the eval config (substitutes ${DATA_DIR}).
EVAL_CONFIG="${OUTPUT_DIR}/eval_dapo_heldout.yaml"
DATA_DIR="${DATA_DIR}" envsubst < "${EXAMPLE_DIR}/eval_dapo_heldout.yaml" > "${EVAL_CONFIG}"

# Qwen3.5-35B-A3B architecture (no scripts/models/*.sh ships for it).
MODEL_ARGS=(
   --spec miles_plugins.models.qwen3_5 get_qwen3_5_spec
   --disable-bias-linear --qk-layernorm --group-query-attention
   --num-attention-heads 16 --num-query-groups 2 --kv-channels 256
   --num-layers 40 --hidden-size 2048 --ffn-hidden-size 512
   --normalization RMSNorm --apply-layernorm-1p --position-embedding-type rope
   --norm-epsilon 1e-6 --rotary-percent 0.25 --swiglu
   --untie-embeddings-and-output-weights --vocab-size 248320 --rotary-base 10000000
   --moe-ffn-hidden-size 512 --moe-shared-expert-intermediate-size 512
   --moe-router-score-function softmax --moe-token-dispatcher-type alltoall
   --moe-router-topk 8
   --moe-layer-freq "[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]"
   --num-experts 256 --moe-grouped-gemm --moe-token-drop-policy probs --moe-router-dtype fp32
   --moe-permute-fusion --moe-aux-loss-coeff 0 --attention-output-gate --moe-shared-expert-gate
   --mtp-num-layers 1
)
CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.5-35B-A3B
   --ref-load ${MODEL_DIR}/Qwen3.5-35B-A3B_torch_dist
   --load ${OUTPUT_DIR} --save ${OUTPUT_DIR} --save-interval 5
)
ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}/dapo_train.jsonl --input-key prompt --label-key label
   --apply-chat-template --rollout-shuffle
   --num-rollout 20 --rollout-batch-size 32 --n-samples-per-prompt 8
   --rollout-max-response-len 24576 --rollout-temperature 1 --num-steps-per-rollout 1
   --over-sampling-batch-size 32 --global-batch-size 256 --balance-data
)
# Format-agnostic correctness reward -> rollout/raw_reward == accuracy.
RM_ARGS=( --custom-rm-path examples.on_policy_distillation.qwen3_5_35b_selfdistill.rm.reward_func )
EVAL_ARGS=( --eval-interval 5 --eval-config ${EVAL_CONFIG} )
GRPO_ARGS=(
   --advantage-estimator grpo --kl-loss-type low_var_kl --entropy-coef 0.00
   --eps-clip 0.2 --eps-clip-high 0.28 --use-tis
)
OPTIMIZER_ARGS=(
   --optimizer adam --lr 1e-5 --lr-decay-style constant --weight-decay 0.1
   --adam-beta1 0.9 --adam-beta2 0.98
   --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer
)
PERF_ARGS=(
   --tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1
   --context-parallel-size 2 --expert-model-parallel-size 8 --expert-tensor-parallel-size 1
   --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
   --use-dynamic-batch-size --max-tokens-per-gpu 16384 --log-probs-chunk-size 4096
)
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8 --sglang-mem-fraction-static 0.8 --sglang-ep-size 8
   --sglang-watchdog-timeout 1800 --sglang-enable-metrics --sglang-attention-backend fa3
   --sglang-cuda-graph-bs 1 2 4 8 16 32 --use-rollout-routing-replay
   --sglang-mamba-scheduler-strategy extra_buffer
)
MISC_ARGS=(
   --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32 --attention-backend flash
)
WANDB_ARGS=( --use-wandb --wandb-project miles-opd --wandb-group qwen3.5-35b-rlvr-teacher )

RUNTIME_ENV_JSON="{\"env_vars\": {\"PYTHONPATH\": \"${MILES_DIR}:/root/Megatron-LM/\", \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\", \"WANDB_API_KEY\": \"${WANDB_API_KEY}\"}}"

cd "${MILES_DIR}"
ray job submit --address="${RAY_ADDRESS}" --submission-id qwen3.5-rlvr-teacher --no-wait \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${MILES_DIR}/train.py \
   --actor-num-nodes 1 --actor-num-gpus-per-node 8 --num-gpus-per-node 8 --colocate \
   ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} ${OPTIMIZER_ARGS[@]} ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} ${PERF_ARGS[@]} ${EVAL_ARGS[@]} ${SGLANG_ARGS[@]} ${MISC_ARGS[@]} ${RM_ARGS[@]}
echo "Submitted Phase-1 RLVR teacher training (submission-id qwen3.5-rlvr-teacher)."
echo "Watch rollout/raw_reward climb and eval/dapo_heldout rise above the base ~0.83."
