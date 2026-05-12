#!/bin/bash
# Launcher for MiniMax-M2.5 RL training (mbridge HF weights + raw Megatron build, MTP from scratch).
#
# ============================================================================
# Prerequisites (run once inside the docker container 9e16408bd232):
# ----------------------------------------------------------------------------
#   docker exec -it 9e16408bd232 bash
#   cd /home/yangchengyi/data/miles
#
#   # 1) FP8 -> bf16 dequant (offline GPU job; outputs new HF dir).
#   python tools/fp8_cast_bf16.py \
#       --input-fp8-hf-path  /home/yangchengyi/data/models/MiniMax-M2.5 \
#       --output-bf16-hf-path /home/yangchengyi/data/models/MiniMax-M2.5-bf16
#
#   # 2) Optional: HF -> torch_dist for a local Megatron ckpt layout.
#   #    Weight naming is handled by miles_plugins.mbridge.minimax_m2.MinimaxM2Bridge.
#   # ============================================================================
#   bash scripts/run-minimax-m2.5.sh

# Cleanup leftover servers from a previous run
pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3
pkill -9 ray || true
pkill -9 python || true

set -ex

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/minimax-m2.5.sh"

# Host -> container shared mount of large assets.
BASE_FOLDER=${BASE_FOLDER:-/home/yangchengyi/data/models}
DATA_FOLDER=${DATA_FOLDER:-/home/yangchengyi/data/datasets}

CKPT_ARGS=(
   # --hf-checkpoint: bf16 HF directory produced by tools/fp8_cast_bf16.py
   --hf-checkpoint ${BASE_FOLDER}/MiniMax-M2.5-bf16
   # --ref-load: same HF tree for SGLang / reward (raw training still uses mbridge for weight sync).
   --ref-load      ${BASE_FOLDER}/MiniMax-M2.5-bf16
   # --load/--save: miles' own training checkpoint dir (initially empty)
   --load          ${BASE_FOLDER}/MiniMax-M2.5_miles/
   --save          ${BASE_FOLDER}/MiniMax-M2.5_miles/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_FOLDER}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime ${DATA_FOLDER}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

# TP=4 / EP=8 / CP=2 -- aligned with origin/feat/minimax_m2_1202's run_minimax_m2.py.
# PerLayerRMSNorm is TP-aware (all_gather + slice) so TP>1 is correct.
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
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

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project miles-dev
   # --wandb-group minimax-m2.5
   # --wandb-key ${WANDB_KEY}
)

# SGLang rollout: EAGLE speculative decoding ON (M2.5's MTP heads drive it).
# Note: with --enable-mtp-training the MTP head is updated alongside the trunk,
# and the bridge round-trips its weights into the SGLang FP8 quantizer.
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-ep-size 8

   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)

   # MTP / EAGLE
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 2
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 3

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

# Master node IP for ray (single-node defaults to localhost)
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

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
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${MODEL_ARGS_MTP_TRAIN[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
