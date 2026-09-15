#!/usr/bin/env bash
# Serve the multi-LoRA Tinker gateway at its measured capacity, run one tinker-cookbook tenant per slot, print the timing tables.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
MODEL=${MODEL:-/root/models/Qwen3-30B-A3B}
MODEL_TYPE=${MODEL_TYPE:-qwen3-30B-A3B}
TINKER_BASE_MODEL=${TINKER_BASE_MODEL:-Qwen/Qwen3-30B-A3B}  # the name the gateway serves; the cookbook resolves its tokenizer from it
ACTOR_GPUS=${ACTOR_GPUS:-8}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-8}
TP=${TP:-2}
EP=${EP:-8}
GPUS_PER_ENGINE=${GPUS_PER_ENGINE:-2}
N_ADAPTERS=${N_ADAPTERS:--1}  # -1: the measured capacity
N_CLIENTS=${N_CLIENTS:-}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
CONTEXT_LEN=${CONTEXT_LEN:-8192}
STEPS=${STEPS:-3}
TASK=${TASK:-rl}  # run_client_recipes.py --mode: rl (GRPO on GSM8K), sft, both
MAX_TOKENS=${MAX_TOKENS:-512}  # a tenant's response length; must leave room for the prompt within CONTEXT_LEN
BATCH_SIZE=${BATCH_SIZE:-}  # prompts per tenant step; empty keeps the recipe's default
GROUP_SIZE=${GROUP_SIZE:-4}  # responses per prompt
CLIENT_LORA_RANK=${CLIENT_LORA_RANK:-8}  # the rank each tenant trains at; LORA_RANK above is what one slot is sized for
CLIENT_PYTHON=${CLIENT_PYTHON:-python3}  # the interpreter with tinker-cookbook installed, e.g. a venv's; the gateway keeps python3
SEQS_PER_SLOT=${SEQS_PER_SLOT:-8}  # sequences one tenant samples at once, for the engine-side capacity bound
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.92}
SGLANG_EP=${SGLANG_EP:-$GPUS_PER_ENGINE}
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-512}
SGLANG_CUDA_GRAPH_MAX_BS=${SGLANG_CUDA_GRAPH_MAX_BS:-512}
RECOMPUTE=${RECOMPUTE:-1}
KEEP_CKPT=${KEEP_CKPT:-0}
TINKER_PORT=${TINKER_PORT:-10613}
READY_TIMEOUT=${READY_TIMEOUT:-3600}
RAY_DASHBOARD=${RAY_ADDRESS:-http://127.0.0.1:8265}
TINKER_HOST=${TINKER_HOST:-$(echo "$RAY_DASHBOARD" | sed -E 's#^https?://([^:/]+).*#\1#')}  # the Tinker API listens where the Ray job driver runs
export MASTER_ADDR=${MASTER_ADDR:-$TINKER_HOST}  # torch.distributed rendezvous on the Ray head node
[ -z "${NCCL_SOCKET_IFNAME:-}" ] || export NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}
RUN_DIR=${RUN_DIR:-/tmp/multi-lora-pressure/$(date +%Y%m%d-%H%M%S)}
EXTRA_SERVE_ARGS=${EXTRA_SERVE_ARGS:-}

log() { echo "[pressure $(date +%H:%M:%S)] $*"; }
mkdir -p "$RUN_DIR"
[ -d "$MODEL" ] || { log "model dir $MODEL missing"; exit 2; }
[ "$MAX_TOKENS" -lt "$CONTEXT_LEN" ] || { log "MAX_TOKENS=$MAX_TOKENS must be below CONTEXT_LEN=$CONTEXT_LEN"; exit 2; }
[ "$CLIENT_LORA_RANK" -le "$LORA_RANK" ] || { log "CLIENT_LORA_RANK=$CLIENT_LORA_RANK exceeds the slot rank LORA_RANK=$LORA_RANK"; exit 2; }
recipes="sl_loop"; [ "$TASK" = "sft" ] || recipes="rl_loop"; [ "$TASK" = "both" ] && recipes="sl_loop, rl_loop"
"$CLIENT_PYTHON" -c "from tinker_cookbook.recipes import $recipes" 2>/dev/null \
    || { log "CLIENT_PYTHON needs the tinker-cookbook pinned in examples/multi_lora/run_client_recipes.py (with its math-rl extras for rl)"; exit 2; }

SERVE_PID=""
SAMPLER_PID=""
stop_gateway() {
    curl -sf "$RAY_DASHBOARD/api/jobs/" 2>/dev/null | python3 -c '
import json, sys
for job in json.load(sys.stdin):
    if "serve_tinker.py" in (job.get("entrypoint") or "") and job.get("status") in ("PENDING", "RUNNING"):
        print(job["submission_id"])' 2>/dev/null | xargs -r -I{} ray job stop --address "$RAY_DASHBOARD" {} >/dev/null 2>&1 || true
    [ -n "$SERVE_PID" ] && kill "$SERVE_PID" 2>/dev/null || true
}
cleanup() {
    local rc=$?
    trap - EXIT
    [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null || true
    stop_gateway
    [ "$KEEP_CKPT" = "1" ] || rm -rf "$RUN_DIR/ckpt"
    log "logs in $RUN_DIR (serve.log, client-*.log, report.txt, gpu-*.csv)"
    [ $rc -eq 0 ] && log "PRESSURE TEST PASS" || log "PRESSURE TEST FAIL (exit $rc)"
    exit $rc
}
trap cleanup EXIT
trap 'exit 130' INT TERM  # an interrupted run is a failed run

# 1. the gateway at its measured capacity
SERVE_EXTRA="--tinker-base-model $TINKER_BASE_MODEL --multi-lora-rollout-seqs-per-slot $SEQS_PER_SLOT \
 --multi-lora-rollout-tokens-per-seq $CONTEXT_LEN --seq-length $CONTEXT_LEN --rollout-max-context-len $CONTEXT_LEN \
 --sglang-context-length $CONTEXT_LEN --sglang-ep-size $SGLANG_EP --sglang-max-running-requests $SGLANG_MAX_RUNNING_REQUESTS \
 --sglang-cuda-graph-max-bs-decode $SGLANG_CUDA_GRAPH_MAX_BS --sglang-moe-runner-backend triton"
[ "$RECOMPUTE" = "1" ] && SERVE_EXTRA="$SERVE_EXTRA --recompute-granularity full --recompute-method uniform --recompute-num-layers 1"
SERVE_EXTRA="$SERVE_EXTRA $EXTRA_SERVE_ARGS"
log "starting the gateway: $ACTOR_GPUS train GPUs TP$TP/EP$EP + $ROLLOUT_GPUS rollout GPUs, slots=$N_ADAPTERS, rank $LORA_RANK, context $CONTEXT_LEN"
python3 "$REPO/examples/multi_lora/serve_qwen3_30b_a3b_tinker.py" serve \
    --hf-checkpoint "$MODEL" --hf-repo "$TINKER_BASE_MODEL" --model-type "$MODEL_TYPE" \
    --actor-num-gpus "$ACTOR_GPUS" --rollout-num-gpus "$ROLLOUT_GPUS" --tp "$TP" --ep "$EP" \
    --rollout-num-gpus-per-engine "$GPUS_PER_ENGINE" --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION" \
    --n-adapters "$N_ADAPTERS" --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
    --tinker-port "$TINKER_PORT" --save-dir "$RUN_DIR/ckpt" --extra-args "$SERVE_EXTRA" > "$RUN_DIR/serve.log" 2>&1 &
SERVE_PID=$!

# 2. wait for the Tinker API, read the slot count
deadline=$((SECONDS + READY_TIMEOUT))
until curl -sf "http://$TINKER_HOST:$TINKER_PORT/api/v1/healthz" >/dev/null 2>&1; do
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then log "the gateway exited before serving:"; tail -60 "$RUN_DIR/serve.log"; exit 1; fi
    if [ $SECONDS -ge $deadline ]; then log "gateway not ready after ${READY_TIMEOUT}s"; tail -60 "$RUN_DIR/serve.log"; exit 1; fi
    sleep 10
done
log "gateway ready after ${SECONDS}s"
grep -m1 "multi-LoRA capacity" "$RUN_DIR/serve.log" | sed 's/^/[pressure] /' || true
grep -m1 "capacity is bound by" "$RUN_DIR/serve.log" | sed 's/^/[pressure] WARNING /' || true
if [ "$N_ADAPTERS" = "-1" ]; then
    SLOTS=$(grep -m1 -o "multi-LoRA capacity: [0-9]* slots" "$RUN_DIR/serve.log" | grep -o "[0-9]*")
    [ -n "$SLOTS" ] || { log "could not read the resolved slot count from serve.log"; exit 1; }
else
    SLOTS=$N_ADAPTERS
fi
N_CLIENTS=${N_CLIENTS:-$SLOTS}
log "gateway has $SLOTS slots; running $N_CLIENTS tenants x $STEPS steps of the cookbook $TASK recipe (rank $CLIENT_LORA_RANK, $MAX_TOKENS tokens)"
( exec timeout 14400 nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 15 \
    > "$RUN_DIR/gpu-$(hostname -I | tr ' ' '\n' | grep -m1 .).csv" 2>/dev/null ) &
SAMPLER_PID=$!

# 3. one tenant per slot
pids=(); started=$SECONDS
for i in $(seq 0 $((N_CLIENTS - 1))); do
    TINKER_API_KEY="tml-pressure-user-$(printf %02d "$i")" "$CLIENT_PYTHON" "$REPO/examples/multi_lora/run_client_recipes.py" \
        --base-url "http://$TINKER_HOST:$TINKER_PORT" --base-model "$TINKER_BASE_MODEL" --mode "$TASK" --steps "$STEPS" \
        --max-tokens "$MAX_TOKENS" --group-size "$GROUP_SIZE" --lora-rank "$CLIENT_LORA_RANK" ${BATCH_SIZE:+--batch-size "$BATCH_SIZE"} \
        > "$RUN_DIR/client-$i.log" 2>&1 &
    pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
log "$((N_CLIENTS - failed))/$N_CLIENTS tenants passed $STEPS steps in $((SECONDS - started))s"
rc=$(( failed > 0 ))

# 4. the tables
sleep 5
kill "$SAMPLER_PID" 2>/dev/null || true
SAMPLER_PID=""
python3 -m miles.utils.multi_lora_profiling --serve-log "$RUN_DIR/serve.log" $(ls "$RUN_DIR"/gpu-*.csv 2>/dev/null | sed 's/^/--gpu-csv /') \
    $(ls "$RUN_DIR"/client-*.log 2>/dev/null | sed 's/^/--client-log /') \
    | tee "$RUN_DIR/report.txt" || log "report failed"
exit "$rc"
