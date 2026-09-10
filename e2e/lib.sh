# Shared plumbing for the gateway end-to-end scripts (e2e_test.sh, auto_e2e_test.sh).
# Source it after setting the knobs; it defines the functions and the teardown trap.
#
# Everything a run starts (its own Ray head, the gateway job, the SGLang engines) carries two
# markers in its environment: MILES_E2E_FAMILY lets a new run sweep a previous run's leftovers,
# MILES_E2E_RUN keeps a dying run's own teardown away from a newer run. Nothing else on the node
# (another user's Ray cluster or engines) is ever touched.

: "${PY:=/opt/sglang/bin/python3}"
: "${MEGATRON_PATH:=/root/Megatron-LM}"
: "${SGLANG_PYTHONPATH:=/personal/miles-pressure-test-20260910/src/sglang/python}"
: "${RAY_PORT:=6399}"
: "${RAY_DASH_PORT:=8299}"
: "${RAY_CLIENT_PORT:=10099}"
: "${RAY_AGENT_PORT:=52399}"
: "${RAY_WORKER_PORT_MIN:=35000}"
: "${RAY_WORKER_PORT_MAX:=35999}"
: "${READY_TIMEOUT:=2400}"
: "${KEEP_GATEWAY:=0}"   # 1: leave the gateway up after a passing client (debugging)
: "${RUN_DIR:=$RUN_ROOT/$(date +%Y%m%d-%H%M%S)}"

export MILES_E2E_FAMILY=${MILES_E2E_FAMILY:-miles-e2e-$(basename "$RUN_ROOT")}
export MILES_E2E_RUN="$MILES_E2E_FAMILY-$(basename "$RUN_DIR")-$$"
export RAY_ADDRESS=http://127.0.0.1:$RAY_DASH_PORT
JOB_ID=""
OWN_SID=$(ps -o sid= -p $$ | tr -d ' ')

log() { echo "[e2e $(date +%H:%M:%S)] $*"; }

marker_pids() {  # $1: environment line to look for, e.g. MILES_E2E_RUN=<value>
    local p pid
    for p in /proc/[0-9]*; do
        pid=${p#/proc/}
        if { tr '\0' '\n' < "$p/environ" | grep -qx "$1"; } 2>/dev/null; then
            # skip this script's own session (itself, its pipelines, the log tailer); the ray head,
            # the gateway job and the engines all live in sessions of their own
            [ "$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ')" = "$OWN_SID" ] && continue
            echo "$pid"
        fi
    done
}

kill_marked() {  # $1: environment line selecting the processes to stop
    local pids
    pids=$(marker_pids "$1" | tr '\n' ' ')
    [ -z "${pids// /}" ] && return 0
    log "stopping processes marked $1: $pids"
    kill -TERM $pids 2>/dev/null || true
    sleep 8
    pids=$(marker_pids "$1" | tr '\n' ' ')
    [ -n "${pids// /}" ] && kill -KILL $pids 2>/dev/null || true
    sleep 2
}

job_status() {  # the Ray Jobs REST API; the CLI's wording changes between releases
    curl -sf "$RAY_ADDRESS/api/jobs/$JOB_ID" 2>/dev/null \
        | "$PY" -c 'import json, sys; print(json.load(sys.stdin).get("status", ""))' 2>/dev/null || true
}

e2e_cleanup() {
    local rc=$?
    trap - EXIT
    if [ "$KEEP_GATEWAY" = "1" ] && [ $rc -eq 0 ]; then
        log "KEEP_GATEWAY=1: leaving the gateway up (job $JOB_ID, Tinker :$TINKER_PORT); the next run sweeps it"
        exit 0
    fi
    [ -n "$JOB_ID" ] && ray job stop --address "$RAY_ADDRESS" "$JOB_ID" >/dev/null 2>&1 || true
    kill_marked "MILES_E2E_RUN=$MILES_E2E_RUN"
    log "GPU memory after teardown:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$GPUS" | sed 's/^/    /'
    log "logs: $RUN_DIR (serve.log, client.log, serve-command.txt)"
    if [ $rc -eq 0 ]; then log "E2E PASS"; else log "E2E FAIL (exit $rc)"; fi
    exit $rc
}
trap e2e_cleanup EXIT

e2e_preflight() {
    [ -d "$MODEL" ] || { log "model dir $MODEL missing"; exit 2; }
    [ -f "$REPO/serve_tinker.py" ] || { log "$REPO is not a miles tree"; exit 2; }
    [ -d "$SGLANG_PYTHONPATH/sglang" ] || { log "SGLANG_PYTHONPATH $SGLANG_PYTHONPATH has no sglang package"; exit 2; }
    local g used port
    for g in ${GPUS//,/ }; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
        [ "$used" -lt 2048 ] || { log "GPU $g has ${used} MiB in use by someone else; refusing to share it"; exit 2; }
    done
    NGPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
    [ "$NGPUS" -ge $((TRAIN_GPUS + ROLLOUT_GPUS)) ] || { log "need $((TRAIN_GPUS + ROLLOUT_GPUS)) GPUs, GPUS=$GPUS has $NGPUS"; exit 2; }
    kill_marked "MILES_E2E_FAMILY=$MILES_E2E_FAMILY"  # leftovers of previous runs of this family
    for port in "$TINKER_PORT" "$RAY_PORT" "$RAY_DASH_PORT"; do
        if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$port\$"; then log "port $port is busy"; exit 2; fi
    done
    mkdir -p "$RUN_DIR/ckpt" "$RUN_DIR/ray"
}

e2e_start_ray() {
    export CUDA_VISIBLE_DEVICES=$GPUS PYTHONUNBUFFERED=1
    ray start --head --node-ip-address 127.0.0.1 --port "$RAY_PORT" --dashboard-port "$RAY_DASH_PORT" \
        --ray-client-server-port "$RAY_CLIENT_PORT" --dashboard-agent-listen-port "$RAY_AGENT_PORT" \
        --min-worker-port "$RAY_WORKER_PORT_MIN" --max-worker-port "$RAY_WORKER_PORT_MAX" \
        --num-gpus "$NGPUS" --temp-dir "$RUN_DIR/ray" --disable-usage-stats > "$RUN_DIR/ray-start.log" 2>&1
    log "ray head up at $RAY_ADDRESS (GPUs $GPUS)"
}

e2e_model_args() {  # $1: model type under scripts/models, e.g. qwen3-30B-A3B; prints the shell-quoted line
    PYTHONPATH="$REPO" "$PY" -c "from miles.utils.external_utils.command_utils import shell_safe_model_args; print(shell_safe_model_args('$1'))" 2>/dev/null | tail -1
}

e2e_submit_gateway() {  # $1: model-args line (shell-quoted), $2: serve args
    local runtime_env model_args_arr
    eval "model_args_arr=($1)"
    runtime_env=$(cat <<JSON
{"env_vars": {"PYTHONUNBUFFERED": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "1", "NCCL_NVLS_ENABLE": "0",
 "no_proxy": "127.0.0.1,localhost", "MASTER_ADDR": "127.0.0.1", "RAY_DEDUP_LOGS": "0",
 "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1", "MILES_E2E_FAMILY": "$MILES_E2E_FAMILY", "MILES_E2E_RUN": "$MILES_E2E_RUN",
 "PYTHONPATH": "$REPO:$MEGATRON_PATH:$SGLANG_PYTHONPATH"}}
JSON
)
    JOB_ID=e2e-$(date +%s)
    echo "python3 $REPO/serve_tinker.py $1 $2" > "$RUN_DIR/serve-command.txt"
    ray job submit --address "$RAY_ADDRESS" --submission-id "$JOB_ID" --runtime-env-json "$runtime_env" --no-wait \
        -- python3 "$REPO/serve_tinker.py" "${model_args_arr[@]}" $2 > "$RUN_DIR/ray-submit.log" 2>&1
    ray job logs --address "$RAY_ADDRESS" -f "$JOB_ID" > "$RUN_DIR/serve.log" 2>&1 &
    log "gateway job $JOB_ID submitted; waiting for http://127.0.0.1:$TINKER_PORT/api/v1/healthz (timeout ${READY_TIMEOUT}s)"
}

e2e_wait_ready() {
    local deadline=$((SECONDS + READY_TIMEOUT)) status
    until curl -sf "http://127.0.0.1:$TINKER_PORT/api/v1/healthz" > /dev/null 2>&1; do
        status=$(job_status)
        case "$status" in
            FAILED|STOPPED|SUCCEEDED) log "gateway job ended before serving: $status"; tail -80 "$RUN_DIR/serve.log"; exit 1 ;;
        esac
        if [ $SECONDS -ge $deadline ]; then log "gateway not ready after ${READY_TIMEOUT}s"; tail -80 "$RUN_DIR/serve.log"; exit 1; fi
        sleep 10
    done
    log "gateway ready after $SECONDS s"
    grep -m1 "agree with predicted\|diverge from predicted" "$RUN_DIR/serve.log" | sed 's/^/[e2e] /' || true
    grep -m1 "multi-LoRA capacity" "$RUN_DIR/serve.log" | sed 's/^/[e2e] /' || log "(no capacity line: explicit slot count)"
}

e2e_resolved_slots() {  # the slot count the gateway ended up with
    if [ "$N_ADAPTERS" = "auto" ]; then
        grep -m1 -o "multi-LoRA capacity: [0-9]* slots" "$RUN_DIR/serve.log" | grep -o "[0-9]*"
    else
        echo "$N_ADAPTERS"
    fi
}
