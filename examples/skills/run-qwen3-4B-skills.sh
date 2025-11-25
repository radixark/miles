#!/bin/bash

# Example launcher that reuses the Qwen3-4B recipe but delegates evaluation to an
# external Nemo Skills server via the --eval-delegate-* knobs.

# Clean up any stale processes from a previous run.
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

python examples/skills/skills_eval_server.py \
  --host 0.0.0.0 \
  --port 9050 \
  --output-root /root/shared/skills-eval \
  --config-dir examples/skills \
  --cluster local_cluster &

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
source "${REPO_ROOT}/scripts/models/qwen3-4B.sh"

# ---------------------------------------------------------------------------
# Skills delegate configuration. Override through environment variables, e.g.
#   SKILLS_EVAL_URL=http://skills-host:9000/evaluate \
#   SKILLS_EVAL_HEADERS_JSON='{"Authorization":"Bearer ..."}' \
#   SKILLS_EVAL_EXTRA_JSON='{"tasks":["aime_2024","amc_2024"],"priority":"low"}'
# ---------------------------------------------------------------------------
SKILLS_EVAL_URL=${SKILLS_EVAL_URL:-"http://127.0.0.1:9050/evaluate"}
SKILLS_EVAL_TIMEOUT_SECS=${SKILLS_EVAL_TIMEOUT_SECS:-7200}
SKILLS_EVAL_MAX_RETRIES=${SKILLS_EVAL_MAX_RETRIES:-5}
SKILLS_EVAL_HEADERS_JSON=${SKILLS_EVAL_HEADERS_JSON:-'{}'}
SKILLS_EVAL_EXTRA_JSON=${SKILLS_EVAL_EXTRA_JSON:-'{"benchmarks":["aime24:0"]}'}

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-4B
   --ref-load /root/Qwen3-4B_torch_dist
   # --load /root/shared/Qwen3-4B_slime/
   --save /root/shared/Qwen3-4B_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 0.8

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 8192
   --eval-top-p 0.7
)

SKILLS_EVAL_ARGS=(
   --eval-delegate-url "${SKILLS_EVAL_URL}"
   --eval-delegate-timeout-secs "${SKILLS_EVAL_TIMEOUT_SECS}"
   --eval-delegate-max-retries "${SKILLS_EVAL_MAX_RETRIES}"
   --eval-delegate-extra "${SKILLS_EVAL_EXTRA_JSON}"
   --eval-delegate-headers "${SKILLS_EVAL_HEADERS_JSON}"
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
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
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-skills
   --wandb-group qwen3-4b-skills
   --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export CUDA_VISIBLE_DEVICES=6,7
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 2 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

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
   --actor-num-gpus-per-node 2 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SKILLS_EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
