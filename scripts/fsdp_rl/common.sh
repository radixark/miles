#!/bin/bash
# Shared launcher for the FSDP-backend RL reference scripts.
#
# A per-model script sets a few env vars then `source`s this file. Policy (fixed for all models):
#   - train data : DAPO-math-17k        - eval : gsm8k        - seq : 8k (response len)
#   - GRPO, AdamW, colocate rollout      - NO checkpoint saving (no --save/--load)
#
# Per-model knobs (with defaults):
#   MODEL              (required) HF hub id, absolute path, or a name under $MODELS_DIR
#   NNODES=1           number of nodes
#   GPUS_PER_NODE=8    GPUs per node
#   CPU_OFFLOAD=0      1 -> --fsdp-cpu-offload (params/grads/optimizer to CPU; for big models)
#   MAX_TOKENS_PER_GPU=16384   dynamic-batch token budget (must exceed prompt+SEQ)
#   SGLANG_MEM=0.55    sglang static KV fraction
#   SEQ=8192           rollout/eval max response length
#   EXTRA_ARGS=()      optional extra train.py flags (bash array)
#
# Path overrides: MODELS_DIR, DATA_DIR, MILES_DIR, MEGATRON_PATH, HF_HOME.
# wandb is enabled automatically when WANDB_API_KEY is set.

: "${MODELS_DIR:=/cluster_public/miles_data/models}"
: "${DATA_DIR:=/cluster_public/miles_data/datasets}"
: "${MILES_DIR:=/root/miles}"
: "${MEGATRON_PATH:=/root/Megatron-LM/}"
: "${HF_HOME:=/storage/cache/huggingface}"
: "${NNODES:=1}"
: "${GPUS_PER_NODE:=8}"
: "${CPU_OFFLOAD:=1}"     # default on: optimizer/params to CPU so the colocated training step fits (validated)
: "${ROLLOUT_GPUS_PER_ENGINE:=1}"   # sglang TP per engine; 26B+ models need 2 (don't fit one GPU at SGLANG_MEM)
: "${MAX_TOKENS_PER_GPU:=10240}"   # fits one 8k-response sequence + prompt; bigger OOMs the loss/logits
: "${SGLANG_MEM:=0.4}"   # colocate: leave enough GPU for the training step (raise CPU_OFFLOAD / lower this if OOM)
: "${SEQ:=8192}"
: "${RUN_ID:=fsdp-rl}"
: "${MASTER_ADDR:=127.0.0.1}"
[ -n "${MODEL:-}" ] || { echo "ERROR: MODEL not set"; exit 1; }
[ "${#EXTRA_ARGS[@]}" -ge 0 ] 2>/dev/null || EXTRA_ARGS=()

# MODEL with a "/" is a hub id or absolute path; a bare name resolves under $MODELS_DIR.
case "$MODEL" in */*) HF="$MODEL" ;; *) HF="$MODELS_DIR/$MODEL" ;; esac
DAPO="$DATA_DIR/dapo-math-17k/dapo-math-17k.jsonl"
GSM8K_TEST="$DATA_DIR/gsm8k/test.parquet"

pkill -9 sglang 2>/dev/null; ray stop --force 2>/dev/null; pkill -9 ray 2>/dev/null; pkill -9 python 2>/dev/null; sleep 2
ulimit -n 524288 2>/dev/null || true   # raylet spawns many workers; raise the fd limit
export PYTHONUNBUFFERED=1 HF_HOME

CKPT_ARGS=( --hf-checkpoint "$HF" )                      # no --save/--load: nothing is checkpointed
ROLLOUT_ARGS=(
   --prompt-data "$DAPO" --input-key prompt --label-key label
   --apply-chat-template --rollout-shuffle --rm-type math
   --num-rollout 200 --rollout-batch-size 32 --n-samples-per-prompt 8
   --rollout-max-response-len "$SEQ" --rollout-temperature 1.0 --global-batch-size 256
)
EVAL_ARGS=(
   --eval-interval 10 --eval-prompt-data gsm8k "$GSM8K_TEST" --eval-input-key messages
   --n-samples-per-eval-prompt 1 --eval-max-response-len "$SEQ" --eval-top-k 1
)
GRPO_ARGS=( --advantage-estimator grpo --kl-loss-coef 0.0 --kl-coef 0.0 --entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.28 )
OPTIMIZER_ARGS=( --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 )
SGLANG_ARGS=( --rollout-num-gpus-per-engine "$ROLLOUT_GPUS_PER_ENGINE" --sglang-mem-fraction-static "$SGLANG_MEM" --sglang-decode-log-interval 1000 --sglang-chunked-prefill-size 4096 --sglang-attention-backend fa3 )
TRAIN_BACKEND_ARGS=(
   --train-backend fsdp --update-weight-buffer-size 536870912
   --gradient-checkpointing --attn-implementation eager
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
)
[ "$CPU_OFFLOAD" = "1" ] && TRAIN_BACKEND_ARGS+=( --fsdp-cpu-offload )
PERF_ARGS=( --use-dynamic-batch-size --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU" )
# --ci-test is required: the FSDP backend is experimental and gated behind it (arguments.py raises otherwise).
MISC_ARGS=( --actor-num-nodes "$NNODES" --actor-num-gpus-per-node "$GPUS_PER_NODE" --colocate --use-fault-tolerance --ci-test --ci-disable-logprobs-checker )
WANDB_ARGS=()
[ -n "${WANDB_API_KEY:-}" ] && WANDB_ARGS=( --use-wandb --wandb-project "${WANDB_PROJECT:-miles_fsdp_rl}" --wandb-group "$RUN_ID" --wandb-key "$WANDB_API_KEY" )

ray start --head --node-ip-address "$MASTER_ADDR" --num-gpus "$GPUS_PER_NODE" --disable-usage-stats
if [ "$NNODES" -gt 1 ]; then
   echo ">> NNODES=$NNODES: start ray on the other $((NNODES-1)) node(s) with"
   echo "     ray start --address=$MASTER_ADDR:6379 --num-gpus=$GPUS_PER_NODE"
   echo "   (or launch via your multi-node cluster tooling) so all $((NNODES*GPUS_PER_NODE)) GPUs join before training."
fi
RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"$MEGATRON_PATH\",\"no_proxy\":\"127.0.0.1\"}}"

cd "$MILES_DIR"
set -x
ray job submit --address="http://127.0.0.1:8265" --runtime-env-json="$RUNTIME_ENV_JSON" \
   -- python3 train.py \
   "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${EVAL_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" "${GRPO_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" "${TRAIN_BACKEND_ARGS[@]}" "${PERF_ARGS[@]}" "${MISC_ARGS[@]}" "${WANDB_ARGS[@]}" "${EXTRA_ARGS[@]}"
