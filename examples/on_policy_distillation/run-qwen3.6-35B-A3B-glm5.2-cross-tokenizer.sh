#!/bin/bash
#
# Cross-tokenizer on-policy distillation: GLM5.2 teacher -> Qwen3.6-35B-A3B student.
#
# This is the STUDENT TRAINER launcher. The GLM5.2 teacher runs as a SEPARATE
# SGLang server (a different node / container / Pod), because a 744B-A40B teacher
# and a 35B-A3B student do not co-fit on a single 8-GPU node. Point RM_URL at that
# server. See examples/on_policy_distillation/k8s/cross-tokenizer-opd.yaml for the
# two-node K8s topology, and the teacher-launch snippet at the bottom of this file.
#
# Required env (no model ids are hard-coded — set these to real HF ids or local paths):
#   STUDENT_MODEL          HF id or local dir of the Qwen3.6-35B-A3B student (e.g. /models/qwen3.6-35B-A3B)
#   TEACHER_TOKENIZER      HF id or local dir of the GLM5.2 tokenizer (the cross-tokenizer switch)
#   RM_URL                 GLM5.2 SGLang server generate endpoint (e.g. http://glm52-teacher:30000/generate)
# Optional env:
#   STUDENT_TORCH_DIST     torch_dist checkpoint dir for the student (built here if missing)
#   DATA_PATH              prompt jsonl (default /root/dapo-math-17k/dapo-math-17k.jsonl)
#   OPD_KL_COEF            distillation strength (default 1.0)
#   MEGATRON_PATH          Megatron-LM path (default /root/Megatron-LM)
#
# usage: bash examples/on_policy_distillation/run-qwen3.6-35B-A3B-glm5.2-cross-tokenizer.sh

set -ex

# for rerun the task
pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3

export PYTHONBUFFERED=16

STUDENT_MODEL="${STUDENT_MODEL:?set STUDENT_MODEL to the Qwen3.6-35B-A3B HF id or local path}"
TEACHER_TOKENIZER="${TEACHER_TOKENIZER:?set TEACHER_TOKENIZER to the GLM5.2 tokenizer HF id or local path}"
RM_URL="${RM_URL:?set RM_URL to the GLM5.2 SGLang server, e.g. http://glm52-teacher:30000/generate}"
STUDENT_TORCH_DIST="${STUDENT_TORCH_DIST:-${STUDENT_MODEL%/}_torch_dist}"
DATA_PATH="${DATA_PATH:-/root/dapo-math-17k/dapo-math-17k.jsonl}"
OPD_KL_COEF="${OPD_KL_COEF:-1.0}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
source "${REPO_ROOT}/scripts/models/qwen3.6-35B-A3B.sh"

# Wait for the GLM5.2 teacher server to be ready.
TEACHER_HEALTH="${RM_URL%/generate}/health_generate"
echo "Waiting for the GLM5.2 teacher server at ${TEACHER_HEALTH} ..."
until curl -sf "${TEACHER_HEALTH}" >/dev/null; do
    echo "  teacher not ready yet; retrying in 10s..."
    sleep 10
done
echo "GLM5.2 teacher server is up."

# Convert the student HF checkpoint to Megatron torch_dist if not already present.
if [ ! -d "${STUDENT_TORCH_DIST}" ]; then
    echo "Building student torch_dist at ${STUDENT_TORCH_DIST} ..."
    PYTHONPATH="${MEGATRON_PATH}" python3 tools/convert_hf_to_torch_dist.py \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint "${STUDENT_MODEL}" \
        --save "${STUDENT_TORCH_DIST}"
fi

CKPT_ARGS=(
   --hf-checkpoint "${STUDENT_MODEL}"
   --ref-load "${STUDENT_TORCH_DIST}"
   --load /root/Qwen3.6-35B-A3B_miles/
   --save /root/Qwen3.6-35B-A3B_miles/
   --save-interval 50
)

ROLLOUT_ARGS=(
   --prompt-data "${DATA_PATH}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 1000
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1.0

   --global-batch-size 128
   --balance-data
)

EVAL_ARGS=(
   # --eval-interval 50
   # --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
   # --n-samples-per-eval-prompt 16
   # --eval-max-response-len 16384
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

# Cross-tokenizer OPD: pure distillation (advantage estimator is grpo but task
# reward is 0; the entire learning signal is the per-token reverse-KL penalty
# stored in opd_reverse_kl by the cross-tokenizer reward path).
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef "${OPD_KL_COEF}"
   --opd-log-prob-top-k 0
   --opd-teacher-tokenizer "${TEACHER_TOKENIZER}"
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

RM_ARGS=(
   --custom-rm-path miles.rollout.cross_tokenizer_opd.reward_func
   --custom-reward-post-process-path miles.rollout.cross_tokenizer_opd.post_process_rewards
   --rm-url "${RM_URL}"
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

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project miles-opd
   # --wandb-group qwen3.6-35B-A3B-glm5.2-cross-tokenizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-ep-size 8
   --sglang-max-running-requests 512
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --moe-token-dispatcher-type flex
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${RM_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"

# clear after training
pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true

# ---------------------------------------------------------------------------
# Reference: how the GLM5.2 teacher server is launched on its own 8xH200 node
# (done by the `teacher` container in the K8s manifest, NOT by this script):
#
#   python3 -m sglang.launch_server \
#       --model-path "${TEACHER_MODEL}" \
#       --tp 8 \
#       --host 0.0.0.0 --port 30000 \
#       --mem-fraction-static 0.85 \
#       --chunked-prefill-size 8192 \
#       --trust-remote-code
#
# The trainer only needs --rm-url=http://<teacher-host>:30000/generate and the
# teacher tokenizer via --opd-teacher-tokenizer. The teacher is queried for
# per-token logprobs only (max_new_tokens=0), so it never generates.
# ---------------------------------------------------------------------------
