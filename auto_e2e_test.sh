#!/usr/bin/env bash
# auto_e2e_test.sh: start the multi-LoRA Tinker gateway for Qwen3-30B-A3B with the slot count
# measured by `--multi-lora-n-adapters auto`, then run that many tenants at once. Every tenant
# sends the same requests: DAPO on dapo-math-17k inside an 8K context, three dependent steps of
# rollout -> forward/backward -> optim_step -> publish -> next rollout from the new version.
# Knobs are environment variables; see e2e/lib.sh for the shared plumbing and teardown rules.
#
# One node (default): 4 training GPUs TP2/EP4 + 4 sampling GPUs, a private Ray head on this node.
# A cluster: submit to an existing multi-node Ray instead, with the tree, run dir and dataset on
# storage every node mounts, e.g. two training nodes (TP2/EP8) and two sampling nodes:
#   REPO=/personal/<tree> RUN_ROOT=/personal/<runs> TRAIN_NODES=2 TRAIN_GPUS=8 EP=8 ROLLOUT_GPUS=16 \
#   EXTERNAL_RAY_GCS=<head-ip>:<gcs-port> EXTERNAL_RAY_DASH=<head-ip>:<dashboard-port> HEAD_IP=<head-ip> \
#   NODE_IPS=<ip1>,<ip2>,<ip3>,<ip4> NET_IFNAME=bond0 MIN_FREE_GPUS=32 bash auto_e2e_test.sh
# Afterwards run e2e/sweep_node.sh on the other nodes to stop any engine the job left behind.
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
MODEL=${MODEL:-/cluster-storage/models/Qwen3-30B-A3B}
MODEL_TYPE=${MODEL_TYPE:-qwen3-30B-A3B}         # scripts/models/<type>.py
DATASET=${DATASET:-/personal/datasets/dapo-math-17k/dapo-math-17k.jsonl}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}                   # one 8-GPU node: 4 training + 4 sampling
TRAIN_NODES=${TRAIN_NODES:-1}
TRAIN_GPUS=${TRAIN_GPUS:-4}                      # per training node
TP=${TP:-2}
EP=${EP:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
GPUS_PER_ENGINE=${GPUS_PER_ENGINE:-2}
N_ADAPTERS=${N_ADAPTERS:-auto}                  # the measured capacity; a count skips the probe
N_USERS=${N_USERS:-}                            # default: one tenant per slot
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
CONTEXT_LEN=${CONTEXT_LEN:-8192}                # prompt + response, training and sampling alike
MARGIN_BYTES=${MARGIN_BYTES:-1073741824}        # the flag's default; raise it to bound `auto`
STEPS=${STEPS:-3}
PROMPTS_PER_STEP=${PROMPTS_PER_STEP:-2}         # accepted (mixed-reward) prompt groups per step
SAMPLES_PER_PROMPT=${SAMPLES_PER_PROMPT:-8}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-2048}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-6144}
LR=${LR:-1e-5}
ENABLE_THINKING=${ENABLE_THINKING:-0}           # 1: Qwen3 thinking mode (long rollouts)
TINKER_PORT=${TINKER_PORT:-9646}
READY_TIMEOUT=${READY_TIMEOUT:-3600}
RUN_ROOT=${RUN_ROOT:-/scratch/asc3199-auto-e2e}

source "$REPO/e2e/lib.sh"

[ -f "$DATASET" ] || { log "dataset $DATASET missing"; exit 2; }
e2e_preflight
log "run dir $RUN_DIR; ${TRAIN_NODES}x$TRAIN_GPUS train GPUs TP$TP/EP$EP + $ROLLOUT_GPUS rollout GPUs; slots=$N_ADAPTERS rank=$LORA_RANK"
e2e_start_ray

SERVE_ARGS="--hf-checkpoint $MODEL --megatron-to-hf-mode bridge \
 --tinker-checkpoint-root $RUN_DIR/ckpt --tinker-server-port $TINKER_PORT \
 --multi-lora-n-adapters $N_ADAPTERS --lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA --lora-dropout 0 \
 --target-modules linear_qkv,linear_proj,linear_fc1,linear_fc2 --no-gradient-accumulation-fusion \
 --train-backend megatron --qkv-format thd \
 --actor-num-nodes $TRAIN_NODES --actor-num-gpus-per-node $TRAIN_GPUS \
 --tensor-model-parallel-size $TP --sequence-parallel --expert-model-parallel-size $EP --expert-tensor-parallel-size 1 \
 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
 --seq-length $CONTEXT_LEN --rollout-max-context-len $CONTEXT_LEN \
 --use-dynamic-batch-size --max-tokens-per-gpu $CONTEXT_LEN --micro-batch-size 1 --global-batch-size 8 \
 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
 --train-memory-margin-bytes $MARGIN_BYTES \
 --attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
 --attention-backend flash --optimizer adam --lr $LR \
 --rollout-num-gpus $ROLLOUT_GPUS --rollout-num-gpus-per-engine $GPUS_PER_ENGINE --sglang-ep-size $GPUS_PER_ENGINE \
 --sglang-lora-backend triton --sglang-mem-fraction-static 0.85 --sglang-context-length $CONTEXT_LEN \
 --sglang-max-running-requests 128 --sglang-chunked-prefill-size $CONTEXT_LEN --sglang-cuda-graph-max-bs-decode 16 \
 --sglang-moe-runner-backend triton"
e2e_submit_gateway "$(e2e_model_args "$MODEL_TYPE")" "$SERVE_ARGS"
e2e_wait_ready

SLOTS=$(e2e_resolved_slots)
[ -n "$SLOTS" ] || { log "could not read the slot count from serve.log"; exit 1; }
N_USERS=${N_USERS:-$SLOTS}
log "gateway has $SLOTS slots; running $N_USERS tenants x $STEPS steps of DAPO ($PROMPTS_PER_STEP prompts x $SAMPLES_PER_PROMPT samples, <= $CONTEXT_LEN tokens)"

thinking_flag=""; [ "$ENABLE_THINKING" = "1" ] && thinking_flag="--enable-thinking"
PYTHONPATH="$REPO" "$PY" "$REPO/e2e/auto_e2e_client.py" --base-url "http://127.0.0.1:$TINKER_PORT" --base-model "$MODEL" \
    --dataset "$DATASET" --n-users "$N_USERS" --steps "$STEPS" --lora-rank "$LORA_RANK" --lr "$LR" \
    --prompts-per-step "$PROMPTS_PER_STEP" --samples-per-prompt "$SAMPLES_PER_PROMPT" \
    --max-prompt-tokens "$MAX_PROMPT_TOKENS" --max-new-tokens "$MAX_NEW_TOKENS" --context-len "$CONTEXT_LEN" \
    $thinking_flag 2>&1 | tee "$RUN_DIR/client.log"
exit "${PIPESTATUS[0]}"
