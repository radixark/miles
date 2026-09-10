#!/usr/bin/env bash
# e2e_test.sh: start the multi-LoRA Tinker gateway on a few local GPUs (Qwen3-0.6B), then run
# one tenant through dependent steps: rollout -> forward/backward -> optim_step -> publish -> next rollout.
# Knobs are environment variables; see e2e/lib.sh for the shared plumbing and teardown rules.
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
MODEL=${MODEL:-/personal/models/Qwen3-0.6B}
MODEL_TYPE=${MODEL_TYPE:-qwen3-0.6B}          # scripts/models/<type>.py
GPUS=${GPUS:-4,5,6,7}                          # physical GPUs this run may use
TRAIN_GPUS=${TRAIN_GPUS:-2}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}
GPUS_PER_ENGINE=${GPUS_PER_ENGINE:-1}
N_ADAPTERS=${N_ADAPTERS:-auto}                 # a count, or auto (the measured slot capacity)
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-2048}
# head-room the probe leaves untouched; on a 0.6B model this is what keeps `auto` at a few dozen
# slots instead of the ~2600 that physically fit (49 MiB per rank-8 slot, ~131 GiB free on an H200);
# training itself never reads it without --offload-train
MARGIN_BYTES=${MARGIN_BYTES:-137000000000}
STEPS=${STEPS:-3}
MAX_SAMPLE_TOKENS=${MAX_SAMPLE_TOKENS:-16}
TINKER_PORT=${TINKER_PORT:-9645}
RUN_ROOT=${RUN_ROOT:-/scratch/asc3199-e2e}

source "$REPO/e2e/lib.sh"

e2e_preflight
log "run dir $RUN_DIR; GPUs $GPUS ($TRAIN_GPUS train + $ROLLOUT_GPUS rollout); slots=$N_ADAPTERS rank=$LORA_RANK"
e2e_start_ray

SERVE_ARGS="--hf-checkpoint $MODEL --megatron-to-hf-mode bridge \
 --tinker-checkpoint-root $RUN_DIR/ckpt --tinker-server-port $TINKER_PORT \
 --multi-lora-n-adapters $N_ADAPTERS --lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA --lora-dropout 0 \
 --target-modules linear_qkv,linear_proj,linear_fc1,linear_fc2 --no-gradient-accumulation-fusion \
 --train-backend megatron --qkv-format thd \
 --actor-num-nodes 1 --actor-num-gpus-per-node $TRAIN_GPUS \
 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
 --seq-length $MAX_TOKENS_PER_GPU --rollout-max-context-len $MAX_TOKENS_PER_GPU \
 --use-dynamic-batch-size --max-tokens-per-gpu $MAX_TOKENS_PER_GPU --micro-batch-size 1 --global-batch-size 8 \
 --train-memory-margin-bytes $MARGIN_BYTES \
 --attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
 --attention-backend flash --optimizer adam --lr 1e-4 \
 --rollout-num-gpus $ROLLOUT_GPUS --rollout-num-gpus-per-engine $GPUS_PER_ENGINE \
 --sglang-lora-backend triton --sglang-mem-fraction-static 0.5 --sglang-context-length $MAX_TOKENS_PER_GPU \
 --sglang-max-running-requests 16"
e2e_submit_gateway "$(e2e_model_args "$MODEL_TYPE")" "$SERVE_ARGS"
e2e_wait_ready

"$PY" "$REPO/e2e/e2e_client.py" --base-url "http://127.0.0.1:$TINKER_PORT" --base-model "$MODEL" \
    --lora-rank "$LORA_RANK" --steps "$STEPS" --max-tokens "$MAX_SAMPLE_TOKENS" 2>&1 | tee "$RUN_DIR/client.log"
exit "${PIPESTATUS[0]}"
