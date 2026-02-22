#!/bin/bash
#
# Precision debugging script for Qwen3 Next (small model).
#
# Usage:
#   # 1. Run baseline (e.g. on main branch):
#   bash scripts/dump-small-next.sh baseline
#
#   # 2. Switch to your branch and run target:
#   bash scripts/dump-small-next.sh target
#
#   # 3. Compare:
#   bash scripts/compare-dumps.sh baseline target
#
# Custom dump dir:
#   DUMPER_BASE_DIR=/root/shared/dumps bash scripts/dump-small-next.sh baseline
#

set -euo pipefail

TAG="${1:?Usage: $0 <tag>  (e.g. 'baseline' or 'target')}"
DUMPER_BASE_DIR="${DUMPER_BASE_DIR:-/tmp/dumper}"
DUMP_DIR="${DUMPER_BASE_DIR}/${TAG}"

echo "=== Dumper precision run: tag=${TAG}, dump_dir=${DUMP_DIR} ==="

# Clean previous dump for this tag
rm -rf "${DUMP_DIR}"
mkdir -p "${DUMP_DIR}"

# ---- cleanup stale processes ----
pkill -9 sglang || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray || true
pkill -9 python || true
sleep 3
pkill -9 ray || true
pkill -9 python || true

set -x

BASE_FOLDER=/root/shared
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/small-next.sh"

CKPT_ARGS=(
   --hf-checkpoint ${BASE_FOLDER}/Qwen3-Next-80B-A3B-Thinking-8L
   --ref-load ${BASE_FOLDER}/Qwen3-Next-80B-A3B-Thinking_partial_torch_dist
)

ROLLOUT_ARGS=(
   --prompt-data ${BASE_FOLDER}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   # use fixed seed for reproducibility across runs
   --seed 42
   --rm-type deepscaler
   --num-rollout 1
   --rollout-batch-size 8
   --n-samples-per-prompt 2
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 16
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --micro-batch-size 1
   --qkv-format bshd
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator gspo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 4e-4
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
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.4
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --check-weight-update-equal
)

# ---- dumper args: the key part ----
DUMPER_ARGS=(
   --dumper-enable
   --dumper-dir "${DUMP_DIR}"
   # dump all three phases: inference, fwd_only, fwd_bwd
   # add per-phase overrides here if needed, e.g.:
   # --dumper-inference enable=true
   # --dumper-fwd-only enable=true
   # --dumper-fwd-bwd enable=true
)

# ---- launch ----
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${DUMPER_ARGS[@]}

echo ""
echo "=== Dump complete: ${DUMP_DIR} ==="
echo "Phases dumped:"
for phase in inference fwd_only fwd_bwd; do
   dir="${DUMP_DIR}/${phase}"
   if [ -d "$dir" ]; then
      count=$(find "$dir" -name '*.pt' | wc -l)
      echo "  ${phase}: ${count} tensors in ${dir}"
   else
      echo "  ${phase}: (no output)"
   fi
done
