#!/usr/bin/env bash
# auto_e2e_test.sh: the multi-LoRA Tinker gateway at its measured capacity, under N Tinker users.
#
#   1. start the gateway for Qwen3-30B-A3B on two nodes with --multi-lora-n-adapters auto: the
#      slot count is the smallest of the trainer's measured memory, the rollout engines' memory
#      with every slot sampling at once, and torch._grouped_mm's group limit on the expert adapters
#   2. read the N slots it resolved to
#   3. run N copies of e2e/e2e_client.py at once (e2e/run_clients.py), one Tinker user per slot;
#      each is the standard client chain on its own LoRA, DAPO on GSM8K:
#      forward_backward -> optim_step -> save_weights_for_sampler -> sample from the new version
#
# The gateway is submitted as a job to an existing Ray cluster spanning the nodes; this script
# runs on the head node. The tree, the run dir and the dataset must be on storage every node
# mounts. Two 8-GPU nodes: the trainer takes 8 GPUs (TP2/EP8), four TP2 engines take the other 8.
#   node A:  ray start --head --port 6399 --dashboard-port 8299 --num-gpus 8
#   node B:  ray start --address <A>:6399 --num-gpus 8
#   node A:  REPO=/personal/<tree> RUN_ROOT=/personal/<runs> RAY_GCS=<A>:6399 RAY_DASH=<A>:8299 \
#            HEAD_IP=<A> NODE_IPS=<A>,<B> NET_IFNAME=bond0 bash auto_e2e_test.sh
# Knobs are environment variables; e2e/lib.sh holds the plumbing and the teardown rules.
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
MODEL=${MODEL:-/cluster-storage/models/Qwen3-30B-A3B}
MODEL_TYPE=${MODEL_TYPE:-qwen3-30B-A3B}          # scripts/models/<type>.py
DATASET=${DATASET:-/personal/datasets/gsm8k/train.parquet}   # or a dapo-math-17k jsonl
TRAIN_NODES=${TRAIN_NODES:-1}
TRAIN_GPUS=${TRAIN_GPUS:-8}                      # per training node
TP=${TP:-2}
EP=${EP:-8}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-8}
GPUS_PER_ENGINE=${GPUS_PER_ENGINE:-2}
N_ADAPTERS=${N_ADAPTERS:-auto}                   # the measured capacity; a count skips the probe
N_USERS=${N_USERS:-}                             # default: one Tinker user per slot
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
CONTEXT_LEN=${CONTEXT_LEN:-8192}                 # prompt + response, training and sampling alike
MARGIN_BYTES=${MARGIN_BYTES:-1073741824}         # --train-memory-margin-bytes: head-room the probe leaves free
STEPS=${STEPS:-3}
PROMPTS_PER_STEP=${PROMPTS_PER_STEP:-2}          # accepted (mixed-reward) prompt groups per step
SAMPLES_PER_PROMPT=${SAMPLES_PER_PROMPT:-8}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-2048}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}               # a response may fill the context after its prompt
LR=${LR:-1e-5}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.92}            # every engine keeps one LoRA buffer per slot
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-512}
SGLANG_CUDA_GRAPH_MAX_BS=${SGLANG_CUDA_GRAPH_MAX_BS:-512}   # decode batch captured in cuda graphs
SGLANG_MOE_RUNNER=${SGLANG_MOE_RUNNER:-triton}              # the runner that applies expert LoRA
SGLANG_MAX_LOADED_LORAS=${SGLANG_MAX_LOADED_LORAS:-128}     # adapter versions each engine keeps in host RAM (LRU); >= slots
ENABLE_THINKING=${ENABLE_THINKING:-0}            # 1: Qwen3 thinking mode (long rollouts)
TINKER_PORT=${TINKER_PORT:-9646}
READY_TIMEOUT=${READY_TIMEOUT:-3600}
RUN_ROOT=${RUN_ROOT:-/scratch/asc3199-auto-e2e}

source "$REPO/e2e/lib.sh"

[ -f "$DATASET" ] || { log "dataset $DATASET missing"; exit 2; }
e2e_preflight
log "run dir $RUN_DIR; ${TRAIN_NODES}x$TRAIN_GPUS train GPUs TP$TP/EP$EP + $ROLLOUT_GPUS rollout GPUs; slots=$N_ADAPTERS rank=$LORA_RANK"

# every slot samples PROMPTS_PER_STEP x SAMPLES_PER_PROMPT sequences of up to CONTEXT_LEN tokens at once
SERVE_ARGS="--hf-checkpoint $MODEL --megatron-to-hf-mode bridge \
 --tinker-checkpoint-root $RUN_DIR/ckpt --tinker-server-port $TINKER_PORT \
 --multi-lora-n-adapters $N_ADAPTERS --lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA --lora-dropout 0 \
 --multi-lora-rollout-seqs-per-slot $((PROMPTS_PER_STEP * SAMPLES_PER_PROMPT)) --multi-lora-rollout-tokens-per-seq $CONTEXT_LEN \
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
 --sglang-lora-backend triton --sglang-mem-fraction-static $SGLANG_MEM_FRACTION --sglang-context-length $CONTEXT_LEN \
 --sglang-max-running-requests $SGLANG_MAX_RUNNING_REQUESTS --sglang-chunked-prefill-size $CONTEXT_LEN \
 --sglang-cuda-graph-max-bs-decode $SGLANG_CUDA_GRAPH_MAX_BS --sglang-moe-runner-backend $SGLANG_MOE_RUNNER \
 --sglang-max-loaded-loras $SGLANG_MAX_LOADED_LORAS"
e2e_submit_gateway "$(e2e_model_args "$MODEL_TYPE")" "$SERVE_ARGS"
e2e_wait_ready

SLOTS=$(e2e_resolved_slots)
[ -n "$SLOTS" ] || { log "could not read the slot count from serve.log"; exit 1; }
N_USERS=${N_USERS:-$SLOTS}
log "gateway has $SLOTS slots; running $N_USERS clients x $STEPS steps of DAPO ($PROMPTS_PER_STEP prompts x $SAMPLES_PER_PROMPT samples, <= $CONTEXT_LEN tokens)"

thinking_flag=""; [ "$ENABLE_THINKING" = "1" ] && thinking_flag="--enable-thinking"
PYTHONPATH="$REPO" "$PY" "$REPO/e2e/run_clients.py" --n-clients "$N_USERS" -- \
    --base-url "http://127.0.0.1:$TINKER_PORT" --base-model "$MODEL" --dataset "$DATASET" \
    --steps "$STEPS" --lora-rank "$LORA_RANK" --lr "$LR" \
    --prompts-per-step "$PROMPTS_PER_STEP" --samples-per-prompt "$SAMPLES_PER_PROMPT" \
    --max-prompt-tokens "$MAX_PROMPT_TOKENS" --max-new-tokens "$MAX_NEW_TOKENS" --context-len "$CONTEXT_LEN" \
    $thinking_flag 2>&1 | tee "$RUN_DIR/client.log"
exit "${PIPESTATUS[0]}"
