#!/usr/bin/env bash
# run_pressure_test.sh: the multi-LoRA Tinker gateway at its measured capacity, one tenant per slot.
#
#   1. examples/multi_lora/serve_qwen3_30b_a3b_tinker.py serve --n-adapters auto
#      the trainer probes its memory, resolves the slot count N (the smallest of the trainer's
#      memory, the rollout engines' memory with every slot sampling at once, and the grouped-GEMM
#      limit), rebuilds at N and launches the engines
#   2. wait for the Tinker API, read N from the gateway log
#   3. examples/multi_lora/run_multi_tenant_example.py --mode multi --clients N --task dapo
#      N tenants at once, each DAPO on GSM8K on its own LoRA; every step trains on the rollout of
#      the version just published; every tenant times the phases it waits through
#   4. python -m miles.utils.multi_lora_profiling: the client-side phase table, the gateway's
#      per-op timing (where the trainer's time goes) and every node's GPU memory / utilization
#      peaks (nvidia-smi sampled every 15 s while the tenants ran), saved as $RUN_DIR/report.txt
#
# Single node: run as is (the launcher starts Ray). Several nodes: join them into one Ray cluster
# first, then MILES_SCRIPT_EXTERNAL_RAY=1 RAY_ADDRESS=http://<head>:8265 bash run_pressure_test.sh
# on any node of it, with the model and the dataset on storage every node mounts; the gateway's
# Tinker API is reached on the head node (TINKER_HOST), where Ray runs the job driver.
# Knobs are environment variables; the defaults are the two-node Qwen3-30B-A3B layout
# (trainer TP2/EP8 on 8 GPUs, four TP2 engines on 8 GPUs, 8K context).
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"   # this tree's miles for the launcher, the tenants and the report
MODEL=${MODEL:-/root/models/Qwen3-30B-A3B}
MODEL_TYPE=${MODEL_TYPE:-qwen3-30B-A3B}                     # scripts/models/<type>.py
DATASET=${DATASET:-/root/datasets/gsm8k/train.parquet}      # or a dapo-math-17k jsonl
ACTOR_GPUS=${ACTOR_GPUS:-8}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-8}
TP=${TP:-2}
EP=${EP:-8}
GPUS_PER_ENGINE=${GPUS_PER_ENGINE:-2}
N_ADAPTERS=${N_ADAPTERS:-auto}                              # a count skips the probe
N_CLIENTS=${N_CLIENTS:-}                                    # default: one tenant per slot
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
CONTEXT_LEN=${CONTEXT_LEN:-8192}                            # prompt + response, training and sampling alike
STEPS=${STEPS:-3}
PROMPTS_PER_STEP=${PROMPTS_PER_STEP:-2}
SAMPLES_PER_PROMPT=${SAMPLES_PER_PROMPT:-8}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-2048}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.92}            # every engine keeps one LoRA buffer per slot
SGLANG_EP=${SGLANG_EP:-$GPUS_PER_ENGINE}                    # experts split across an engine's GPUs: halves the base weights per GPU
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-512}
SGLANG_CUDA_GRAPH_MAX_BS=${SGLANG_CUDA_GRAPH_MAX_BS:-512}   # decode batch captured in cuda graphs
RECOMPUTE=${RECOMPUTE:-1}                                   # full activation recompute: an 8K-token step peaks at ~5 GB instead of ~42 GB
NODE_IPS=${NODE_IPS:-}                                      # other nodes to sample GPUs on (comma-separated); this node is always sampled
KEEP_CKPT=${KEEP_CKPT:-0}                                   # 1: keep the exported adapter versions (~1.5 GB each) after the run
TINKER_PORT=${TINKER_PORT:-10613}
READY_TIMEOUT=${READY_TIMEOUT:-3600}
RAY_DASHBOARD=${RAY_ADDRESS:-http://127.0.0.1:8265}         # the Jobs API used to stop the gateway
# the gateway's Tinker API listens where the Ray job driver runs: the head node
TINKER_HOST=${TINKER_HOST:-$(echo "$RAY_DASHBOARD" | sed -E 's#^https?://([^:/]+).*#\1#')}
RUN_DIR=${RUN_DIR:-/tmp/multi-lora-pressure/$(date +%Y%m%d-%H%M%S)}
EXTRA_SERVE_ARGS=${EXTRA_SERVE_ARGS:-}

log() { echo "[pressure $(date +%H:%M:%S)] $*"; }
mkdir -p "$RUN_DIR"
# every process the gateway starts carries this line; the SGLang servers outlive their Ray actors
RUN_MARKER="MILES_PRESSURE_RUN=$(basename "$RUN_DIR")-$$"
sweep_marked() {  # $1: environment-line pattern; this node directly, every other NODE_IPS node through Ray
    local ip
    bash "$REPO/examples/multi_lora/sweep_marked.sh" "$1" | sed 's/^/[pressure] /'
    for ip in ${NODE_IPS//,/ }; do
        [ "$ip" = "$(hostname -I | tr ' ' '\n' | grep -m1 .)" ] && continue
        timeout 120 ray job submit --address "$RAY_DASHBOARD" --entrypoint-resources "{\"node:$ip\": 0.001}" \
            -- bash "$REPO/examples/multi_lora/sweep_marked.sh" "$1" 2>/dev/null | grep "\[sweep" | sed 's/^/[pressure] /' || true
    done
}
[ -d "$MODEL" ] || { log "model dir $MODEL missing"; exit 2; }
[ -f "$DATASET" ] || { log "dataset $DATASET missing"; exit 2; }

SERVE_PID=""
SAMPLER_PID=""
SAMPLER_JOBS=""
start_gpu_samplers() {  # nvidia-smi every 15 s on this node and, through Ray, on every node in NODE_IPS
    local ip job
    ( exec timeout 14400 nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader -l 15 > "$RUN_DIR/gpu-$(hostname -I | tr ' ' '\n' | grep -m1 . ).csv" 2>/dev/null ) &
    SAMPLER_PID=$!
    for ip in ${NODE_IPS//,/ }; do
        [ "$ip" = "$(hostname -I | tr ' ' '\n' | grep -m1 .)" ] && continue
        job="gpu-sampler-$(date +%s)-${ip//./-}"
        ray job submit --address "$RAY_DASHBOARD" --submission-id "$job" --no-wait \
            --entrypoint-resources "{\"node:$ip\": 0.001}" -- bash -c \
            "timeout 14400 nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 15 > $RUN_DIR/gpu-$ip.csv" \
            > /dev/null 2>&1 && SAMPLER_JOBS="$SAMPLER_JOBS $job" || true
    done
}
stop_gpu_samplers() {
    local job
    [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null || true
    for job in $SAMPLER_JOBS; do ray job stop --address "$RAY_DASHBOARD" "$job" > /dev/null 2>&1 || true; done
}
stop_gateway() {
    # the launcher's `ray job submit` runs in the foreground; stopping the job tears the actors down
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
    stop_gpu_samplers
    stop_gateway
    sweep_marked "^$RUN_MARKER\$"
    [ "$KEEP_CKPT" = "1" ] || rm -rf "$RUN_DIR/ckpt"   # every publish exported ~1.5 GB; a few runs fill a volume
    log "logs in $RUN_DIR (serve.log, client.log, summary.json, report.txt, gpu-*.csv)"
    [ $rc -eq 0 ] && log "PRESSURE TEST PASS" || log "PRESSURE TEST FAIL (exit $rc)"
    exit $rc
}
trap cleanup EXIT

# ---- 1. the gateway at its measured capacity ----
# every slot samples PROMPTS_PER_STEP x SAMPLES_PER_PROMPT sequences of up to CONTEXT_LEN tokens at once
SERVE_EXTRA="--multi-lora-rollout-seqs-per-slot $((PROMPTS_PER_STEP * SAMPLES_PER_PROMPT)) \
 --multi-lora-rollout-tokens-per-seq $CONTEXT_LEN --seq-length $CONTEXT_LEN --rollout-max-context-len $CONTEXT_LEN \
 --sglang-context-length $CONTEXT_LEN --sglang-ep-size $SGLANG_EP --sglang-max-running-requests $SGLANG_MAX_RUNNING_REQUESTS \
 --sglang-cuda-graph-max-bs-decode $SGLANG_CUDA_GRAPH_MAX_BS --sglang-moe-runner-backend triton"
[ "$RECOMPUTE" = "1" ] && SERVE_EXTRA="$SERVE_EXTRA --recompute-granularity full --recompute-method uniform --recompute-num-layers 1"
SERVE_EXTRA="$SERVE_EXTRA $EXTRA_SERVE_ARGS"
sweep_marked "^MILES_PRESSURE_RUN="   # leftovers of earlier runs on any node
log "starting the gateway: $ACTOR_GPUS train GPUs TP$TP/EP$EP + $ROLLOUT_GPUS rollout GPUs, slots=$N_ADAPTERS, rank $LORA_RANK, context $CONTEXT_LEN"
python3 "$REPO/examples/multi_lora/serve_qwen3_30b_a3b_tinker.py" serve \
    --extra-env-vars "$RUN_MARKER" \
    --hf-checkpoint "$MODEL" --model-type "$MODEL_TYPE" \
    --actor-num-gpus "$ACTOR_GPUS" --rollout-num-gpus "$ROLLOUT_GPUS" --tp "$TP" --ep "$EP" \
    --rollout-num-gpus-per-engine "$GPUS_PER_ENGINE" --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION" \
    --n-adapters "$N_ADAPTERS" --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
    --tinker-port "$TINKER_PORT" --save-dir "$RUN_DIR/ckpt" --extra-args "$SERVE_EXTRA" > "$RUN_DIR/serve.log" 2>&1 &
SERVE_PID=$!

# ---- 2. wait for the Tinker API, read the slot count ----
deadline=$((SECONDS + READY_TIMEOUT))
until curl -sf "http://$TINKER_HOST:$TINKER_PORT/api/v1/healthz" >/dev/null 2>&1; do
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then log "the gateway exited before serving:"; tail -60 "$RUN_DIR/serve.log"; exit 1; fi
    if [ $SECONDS -ge $deadline ]; then log "gateway not ready after ${READY_TIMEOUT}s"; tail -60 "$RUN_DIR/serve.log"; exit 1; fi
    sleep 10
done
log "gateway ready after ${SECONDS}s"
grep -m1 "multi-LoRA capacity" "$RUN_DIR/serve.log" | sed 's/^/[pressure] /' || true
grep -m1 "capacity is bound by" "$RUN_DIR/serve.log" | sed 's/^/[pressure] WARNING /' || true
if [ "$N_ADAPTERS" = "auto" ]; then
    SLOTS=$(grep -m1 -o "multi-LoRA capacity: [0-9]* slots" "$RUN_DIR/serve.log" | grep -o "[0-9]*")
    [ -n "$SLOTS" ] || { log "could not read the resolved slot count from serve.log"; exit 1; }
else
    SLOTS=$N_ADAPTERS
fi
N_CLIENTS=${N_CLIENTS:-$SLOTS}
log "gateway has $SLOTS slots; running $N_CLIENTS tenants x $STEPS DAPO steps ($PROMPTS_PER_STEP prompts x $SAMPLES_PER_PROMPT samples, <= $CONTEXT_LEN tokens)"
start_gpu_samplers

# ---- 3. one tenant per slot ----
if python3 "$REPO/examples/multi_lora/run_multi_tenant_example.py" \
    --base-url "http://$TINKER_HOST:$TINKER_PORT" --base-model "$MODEL" --mode multi --clients "$N_CLIENTS" \
    --task dapo --dataset "$DATASET" --steps "$STEPS" --lora-rank "$LORA_RANK" \
    --prompts-per-step "$PROMPTS_PER_STEP" --samples-per-prompt "$SAMPLES_PER_PROMPT" \
    --max-prompt-tokens "$MAX_PROMPT_TOKENS" --context-len "$CONTEXT_LEN" --max-new-tokens "$CONTEXT_LEN" \
    --summary-json "$RUN_DIR/summary.json" 2>&1 | tee "$RUN_DIR/client.log"; then
    rc=0
else
    rc=$?
fi

# ---- 4. the tables: client-side phases, gateway-side op timing, GPU peaks per node ----
sleep 5  # let the gateway flush its last profile line
stop_gpu_samplers
python3 -m miles.utils.multi_lora_profiling --summary-json "$RUN_DIR/summary.json" \
    --serve-log "$RUN_DIR/serve.log" $(ls "$RUN_DIR"/gpu-*.csv 2>/dev/null | sed 's/^/--gpu-csv /') \
    | tee "$RUN_DIR/report.txt" || log "report failed"
exit "$rc"
