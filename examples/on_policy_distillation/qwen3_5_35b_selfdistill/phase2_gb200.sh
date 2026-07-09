#!/bin/bash
# =============================================================================
# Phase 2: On-policy distillation of the Phase-1 teacher into the BASE student
# =============================================================================
# Student = base Qwen3.5-35B-A3B.  Teacher = the Phase-1 RLVR checkpoint (in-process,
# --opd-type megatron). The student trains only on its own rollouts; the teacher's
# token-level reverse-KL is folded into the GRPO advantages.
#
# Two modes (MODE env var):
#   pure      (default) — training task reward = 0; ONLY the teacher reverse-KL drives
#                         learning. Cleanest attribution: any change is distillation.
#                         eval/dapo_heldout still measures real accuracy.
#   grounded  — keep the correctness reward (rollout/raw_reward == accuracy, climbs)
#                         AND add the teacher reverse-KL on top.
#
# Memory note: we do NOT pass --use-kl-loss, so the reference model is NOT loaded
# (with_ref = use_kl_loss or kl_coef!=0). That keeps only student + teacher in memory
# (2 x 35B, ~124/143 GB per GPU). Adding --use-kl-loss would load a 3rd model and risk OOM.
# The teacher's reverse-KL is the regularizer.
#
# Usage:  TEACHER_LOAD=/path/to/teacher_ckpt_parent bash phase2_opd_selfdistill.sh
# IMPORTANT: --opd-teacher-load must point at the checkpoint PARENT directory (the one
# containing latest_checkpointed_iteration.txt), NOT an iter_XXXXXXX subdir. Pointing at
# the subdir prints "could not find metadata file" and silently falls back to base ->
# teacher == student -> opd_reverse_kl ~= 0 (inert).
# =============================================================================
set -ex
export PYTHONUNBUFFERED=16

MODE=${MODE:-pure}
# GB200 tiling: this cluster is 4 GPUs/node, so world=8 = 2 nodes x 4 GPUs.
# (Upstream phase2_opd_selfdistill.sh assumes a single 8xH200 node.) The parallel
# dims (TP2/PP1/CP2/EP8/ETP1, DP4) are unchanged; only the node tiling differs.
# Both nodes must share one NVLink (MNNVL) domain -- launch with MSC_RACK pinned.
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
MODEL_DIR=${MODEL_DIR:-/cluster_public/miles_data/models}
DATA_DIR=${DATA_DIR:-/node_public/maocheng-qwen35/data}
OUTPUT_DIR=${OUTPUT_DIR:-/node_public/maocheng-qwen35/ckpt-opd-${MODE}}
TEACHER_LOAD=${TEACHER_LOAD:-/node_public/maocheng-qwen35/ckpt-teacher}   # parent dir!
EXAMPLE_DIR=${EXAMPLE_DIR:-$(cd "$(dirname "$0")" && pwd)}
MILES_DIR=${MILES_DIR:-/workspace/miles}
RAY_ADDRESS=${RAY_ADDRESS:-http://127.0.0.1:8265}
OPD_KL_COEF=${OPD_KL_COEF:-0.2}
MOE_FLEX_DISPATCHER_BACKEND=${MOE_FLEX_DISPATCHER_BACKEND:-hybridep}
NUM_ROLLOUT=${NUM_ROLLOUT:-12}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE:-32}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-24576}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-16384}
SAVE_INTERVAL=${SAVE_INTERVAL:-5}
EVAL_INTERVAL=${EVAL_INTERVAL:-5}
SKIP_EVAL_BEFORE_TRAIN=${SKIP_EVAL_BEFORE_TRAIN:-0}
read -r -a SGLANG_CUDA_GRAPH_BS_ARRAY <<< "${SGLANG_CUDA_GRAPH_BS:-1 2 4}"
mkdir -p "${OUTPUT_DIR}"

EVAL_CONFIG="${OUTPUT_DIR}/eval_dapo_heldout.yaml"
DATA_DIR="${DATA_DIR}" envsubst < "${EXAMPLE_DIR}/eval_dapo_heldout.yaml" > "${EVAL_CONFIG}"

if [ "${MODE}" = "pure" ]; then
   RM_FUNC=examples.on_policy_distillation.qwen3_5_35b_selfdistill.rm.reward_func_pure_opd
else
   RM_FUNC=examples.on_policy_distillation.qwen3_5_35b_selfdistill.rm.reward_func
fi

MODEL_ARGS=(
   --spec miles_plugins.models.qwen3_5 get_qwen3_5_spec
   --disable-bias-linear --qk-layernorm --group-query-attention
   --num-attention-heads 16 --num-query-groups 2 --kv-channels 256
   --num-layers 40 --hidden-size 2048 --ffn-hidden-size 512
   --normalization RMSNorm --apply-layernorm-1p --position-embedding-type rope
   --norm-epsilon 1e-6 --rotary-percent 0.25 --swiglu
   --untie-embeddings-and-output-weights --vocab-size 248320 --rotary-base 10000000
   --moe-ffn-hidden-size 512 --moe-shared-expert-intermediate-size 512
   --moe-router-score-function softmax --moe-token-dispatcher-type flex
   --moe-flex-dispatcher-backend ${MOE_FLEX_DISPATCHER_BACKEND}
   --moe-router-topk 8
   --moe-layer-freq "[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]"
   --num-experts 256 --moe-grouped-gemm --moe-token-drop-policy probs --moe-router-dtype fp32
   --moe-permute-fusion --moe-aux-loss-coeff 0 --attention-output-gate --moe-shared-expert-gate
   --mtp-num-layers 1
)
CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.5-35B-A3B
   --ref-load ${MODEL_DIR}/Qwen3.5-35B-A3B_torch_dist --load ${OUTPUT_DIR}
)
if [ "${SAVE_INTERVAL}" != "none" ]; then
   CKPT_ARGS+=(--save ${OUTPUT_DIR} --save-interval ${SAVE_INTERVAL})
fi
OPD_ARGS=(
   --use-opd --opd-type megatron --opd-teacher-load ${TEACHER_LOAD} --opd-kl-coef ${OPD_KL_COEF}
)
ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}/dapo_train.jsonl --input-key prompt --label-key label
   --apply-chat-template --rollout-shuffle
   --num-rollout ${NUM_ROLLOUT} --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
   --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN}
   --rollout-temperature 1 --num-steps-per-rollout 1
   --over-sampling-batch-size ${OVER_SAMPLING_BATCH_SIZE}
   --global-batch-size ${GLOBAL_BATCH_SIZE} --balance-data
)
RM_ARGS=( --custom-rm-path ${RM_FUNC} )
EVAL_ARGS=( --eval-interval ${EVAL_INTERVAL} --eval-config ${EVAL_CONFIG} )
if [ "${SKIP_EVAL_BEFORE_TRAIN}" = "1" ]; then
   EVAL_ARGS+=(--skip-eval-before-train)
fi
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
   --use-dynamic-batch-size --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU} --log-probs-chunk-size 4096
)
# Blackwell (GB200 sm100 / B300 sm103) backends, per scripts/run_qwen3_5_35b_a3b_mtp_cp2_ep8.py:
# - moe-runner-backend flashinfer_cutlass: the default triton fused-MoE mis-shards
#   routed experts on the megatron->sglang weight sync (w13 reshape 64 vs 2048).
# - attention-backend trtllm_mha: FA3 is SM<=90 only; flashinfer/fa3 don't fit here.
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8 --sglang-mem-fraction-static 0.7 --sglang-ep-size 8
   --sglang-watchdog-timeout 1800 --sglang-enable-metrics
   --sglang-moe-runner-backend flashinfer_cutlass --sglang-attention-backend trtllm_mha
   --sglang-cuda-graph-bs "${SGLANG_CUDA_GRAPH_BS_ARRAY[@]}" --use-rollout-routing-replay
   --sglang-mamba-scheduler-strategy extra_buffer
)
MISC_ARGS=(
   --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32 --attention-backend flash
)
WANDB_ARGS=( --use-wandb --wandb-project miles-opd --wandb-group qwen3.5-35b-opd-${MODE} )

# PROMETHEUS_PORT override: the contractor k8s namespace has a `prometheus`
# Service, so kube auto-injects PROMETHEUS_PORT=tcp://<ip>:9090 into every pod.
# miles' add_prometheus_arguments does int(os.environ["PROMETHEUS_PORT"]) and
# crashes on the URL; force the plain port back via the ray runtime env.
# NCCL_NVLS_ENABLE=0: on multi-node GB200 the NVLS (NVLink SHARP) bind fails
# during cross-node ncclCommInitRank ("unhandled cuda error" / CUDA 999), which
# kills the EP8 sglang engine spanning both nodes. MNNVL stays enabled (pod env)
# for the NVLink fabric itself; only NVLS multicast is disabled.
TRACE_RUNTIME_ENV=""
if [ -n "${TMS_CUMEM_TRACE_LIB:-}" ]; then
   TRACE_RUNTIME_ENV=", \"TMS_CUMEM_TRACE_LIB\": \"${TMS_CUMEM_TRACE_LIB}\", \"CUMEM_TRACE_ALL\": \"${CUMEM_TRACE_ALL:-}\""
fi
if [ -n "${TORCH_SHM_UNLINK_COMPAT_LIB:-}" ]; then
   TRACE_RUNTIME_ENV="${TRACE_RUNTIME_ENV}, \"TORCH_SHM_UNLINK_COMPAT_LIB\": \"${TORCH_SHM_UNLINK_COMPAT_LIB}\""
fi
RUNTIME_ENV_JSON="{\"env_vars\": {\"PYTHONPATH\": \"${MILES_DIR}:/root/Megatron-LM/\", \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\", \"WANDB_API_KEY\": \"${WANDB_API_KEY}\", \"PROMETHEUS_PORT\": \"9090\", \"NCCL_NVLS_ENABLE\": \"0\"${TRACE_RUNTIME_ENV}}}"

cd "${MILES_DIR}"
ray job submit --address="${RAY_ADDRESS}" --submission-id qwen3.5-opd-${MODE} --no-wait \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${MILES_DIR}/train.py \
   --actor-num-nodes ${ACTOR_NUM_NODES} --actor-num-gpus-per-node ${GPUS_PER_NODE} --num-gpus-per-node ${GPUS_PER_NODE} --colocate \
   ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${OPD_ARGS[@]} ${ROLLOUT_ARGS[@]} ${OPTIMIZER_ARGS[@]} ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} ${PERF_ARGS[@]} ${EVAL_ARGS[@]} ${SGLANG_ARGS[@]} ${MISC_ARGS[@]} ${RM_ARGS[@]}
echo "Submitted Phase-2 OPD (${MODE}) self-distillation (submission-id qwen3.5-opd-${MODE})."
echo "Watch opd_reverse_kl (>>0 means the teacher differs from the student) and"
echo "eval/dapo_heldout (student moving toward the teacher's accuracy, with shorter responses)."
