# Shared plumbing for the gateway end-to-end scripts (e2e_test.sh, auto_e2e_test.sh).
# Source it after setting the knobs; it defines the functions and the teardown trap.
#
# Everything a run starts (its own Ray head, the gateway job, the SGLang engines) carries
# MILES_E2E_RUN in its environment: preflight sweeps any process still carrying one from an
# earlier run of either script, and teardown stops only the processes carrying this run's value,
# so a dying run never reaches a newer one. Nothing else on the node (another user's Ray
# cluster or engines) is ever touched. MILES_E2E_FAMILY only names the run in logs.

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
: "${RAY_TEMP:=/scratch/e2e-ray-$$}"   # short on purpose: ray's AF_UNIX socket paths must stay under 107 bytes
: "${EXTERNAL_RAY_GCS:=}"       # host:port of an existing Ray GCS to submit to (multi-node); empty: start our own head
: "${EXTERNAL_RAY_DASH:=}"      # host:port of that cluster's dashboard (jobs API)
: "${HEAD_IP:=127.0.0.1}"       # MASTER_ADDR for torch.distributed; the head node's IP on a cluster
: "${NODE_IPS:=}"               # comma-separated node IPs for no_proxy on a cluster
: "${NET_IFNAME:=}"             # NCCL/GLOO socket interface on a cluster, e.g. bond0
: "${MIN_FREE_GPUS:=0}"         # cluster mode: refuse to submit unless the cluster reports this many free GPUs
: "${EXTERNAL_RAY_TEMP:=}"      # the cluster's ray temp dir on this (head) node; found from the raylet when empty
: "${RUN_DIR:=$RUN_ROOT/$(date +%Y%m%d-%H%M%S)}"

export MILES_E2E_FAMILY=${MILES_E2E_FAMILY:-miles-e2e-$(basename "$RUN_ROOT")}
export MILES_E2E_RUN="$MILES_E2E_FAMILY-$(basename "$RUN_DIR")-$$"
if [ -n "$EXTERNAL_RAY_GCS" ]; then
    export RAY_ADDRESS=http://$EXTERNAL_RAY_DASH
else
    export RAY_ADDRESS=http://127.0.0.1:$RAY_DASH_PORT
fi
JOB_ID=""

log() { echo "[e2e $(date +%H:%M:%S)] $*"; }

own_chain() {  # this script and its ancestors: the only marked processes a sweep must leave alone.
    local pid=$$ chain=" "   # ray's daemons, the gateway job and the engines re-parent away and stay fair game
    while [ -n "$pid" ] && [ "$pid" != "0" ] && [ "$pid" != "1" ]; do
        chain="$chain$pid "
        pid=$(awk '{print $4}' "/proc/$pid/stat" 2>/dev/null)
    done
    echo "$chain"
}
OWN_CHAIN=$(own_chain)

marker_pids() {  # $1: grep pattern for one environment line, e.g. ^MILES_E2E_RUN=<value>$
    local p pid
    for p in /proc/[0-9]*; do
        pid=${p#/proc/}
        case "$OWN_CHAIN" in *" $pid "*) continue ;; esac
        if { tr '\0' '\n' < "$p/environ" | grep -q "$1"; } 2>/dev/null; then
            echo "$pid"
        fi
    done
}

kill_marked() {  # $1: grep pattern selecting the processes to stop by an environment line
    local pids
    pids=$(marker_pids "$1" | tr '\n' ' ')
    [ -z "${pids// /}" ] && return 0
    log "stopping processes whose environment matches $1: $pids"
    kill -TERM $pids 2>/dev/null || true
    sleep 8
    pids=$(marker_pids "$1" | tr '\n' ' ')
    [ -n "${pids// /}" ] && kill -KILL $pids 2>/dev/null || true
    sleep 2
}

e2e_ray_temp() {  # cluster mode: where this node's raylet keeps the session (driver logs under session_latest/logs)
    if [ -n "$EXTERNAL_RAY_TEMP" ]; then echo "$EXTERNAL_RAY_TEMP"; return; fi
    pgrep -fa 'raylet' | grep -o -- '--temp_dir=[^ ]*' | head -1 | cut -d= -f2
}

e2e_sweep_cluster_nodes() {  # $1: environment-line pattern; runs e2e/sweep_node.sh on every other node through ray
    local ip
    [ -n "$EXTERNAL_RAY_GCS" ] && [ -n "$NODE_IPS" ] || return 0
    for ip in ${NODE_IPS//,/ }; do
        [ "$ip" = "$HEAD_IP" ] && continue   # this node is swept directly
        timeout 240 ray job submit --address "$RAY_ADDRESS" --entrypoint-resources "{\"node:$ip\": 0.001}" \
            -- bash "$REPO/e2e/sweep_node.sh" "$1" 2>&1 | grep "\[sweep" | sed 's/^/[e2e] /' || true
    done
}

port_busy() {  # $1: port; true when something listens on it (no ss/netstat on these images)
    "$PY" -c 'import socket, sys
s = socket.socket(); s.settimeout(0.3)
sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)' "$1" 2>/dev/null
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
    kill_marked "^MILES_E2E_RUN=$MILES_E2E_RUN\$"
    rm -rf "$RAY_TEMP"
    e2e_sweep_cluster_nodes "^MILES_E2E_RUN=$MILES_E2E_RUN\$"
    log "GPU memory after teardown (this node):"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$GPUS" | sed 's/^/    /'
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
    kill_marked "^MILES_E2E_RUN="  # leftovers of any earlier run of either script, whatever its family
    if [ -n "$EXTERNAL_RAY_GCS" ]; then
        # the cluster's own free-GPU count is the only thing we can check from this node
        local gpu_line used_gpus total_gpus
        gpu_line=$(ray status --address "$EXTERNAL_RAY_GCS" 2>/dev/null | grep -E "^ *[0-9.]+/[0-9.]+ GPU" | head -1)
        [ -n "$gpu_line" ] || { log "cannot read GPU usage from the Ray cluster at $EXTERNAL_RAY_GCS"; exit 2; }
        used_gpus=${gpu_line%%/*}; used_gpus=${used_gpus// /}; total_gpus=$(echo "$gpu_line" | sed 's#.*/\([0-9.]*\) GPU.*#\1#')
        e2e_sweep_cluster_nodes "^MILES_E2E_RUN="
        gpu_line=$(ray status --address "$EXTERNAL_RAY_GCS" 2>/dev/null | grep -E "^ *[0-9.]+/[0-9.]+ GPU" | head -1)
        used_gpus=${gpu_line%%/*}; used_gpus=${used_gpus// /}; total_gpus=$(echo "$gpu_line" | sed 's#.*/\([0-9.]*\) GPU.*#\1#')
        log "cluster $EXTERNAL_RAY_GCS: $used_gpus of $total_gpus GPUs in use"
        awk -v u="$used_gpus" -v t="$total_gpus" -v m="$MIN_FREE_GPUS" 'BEGIN { exit !(t - u >= m) }' \
            || { log "cluster has fewer than $MIN_FREE_GPUS free GPUs; someone else is using it"; exit 2; }
        for port in "$TINKER_PORT"; do
            if port_busy "$port"; then log "port $port is busy"; exit 2; fi
        done
    else
        for g in ${GPUS//,/ }; do
            used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
            [ "$used" -lt 2048 ] || { log "GPU $g has ${used} MiB in use by someone else; refusing to share it"; exit 2; }
        done
        NGPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
        [ "$NGPUS" -ge $((TRAIN_GPUS + ROLLOUT_GPUS)) ] || { log "need $((TRAIN_GPUS + ROLLOUT_GPUS)) GPUs, GPUS=$GPUS has $NGPUS"; exit 2; }
        for port in "$TINKER_PORT" "$RAY_PORT" "$RAY_DASH_PORT"; do
            if port_busy "$port"; then log "port $port is busy"; exit 2; fi
        done
    fi
    mkdir -p "$RUN_DIR/ckpt" "$RAY_TEMP"
}

e2e_start_ray() {
    if [ -n "$EXTERNAL_RAY_GCS" ]; then
        log "using the existing Ray cluster at $EXTERNAL_RAY_GCS (jobs API $RAY_ADDRESS); no local head"
        return 0
    fi
    export CUDA_VISIBLE_DEVICES=$GPUS PYTHONUNBUFFERED=1
    ray start --head --node-ip-address 127.0.0.1 --port "$RAY_PORT" --dashboard-port "$RAY_DASH_PORT" \
        --ray-client-server-port "$RAY_CLIENT_PORT" --dashboard-agent-listen-port "$RAY_AGENT_PORT" \
        --min-worker-port "$RAY_WORKER_PORT_MIN" --max-worker-port "$RAY_WORKER_PORT_MAX" \
        --num-gpus "$NGPUS" --temp-dir "$RAY_TEMP" --disable-usage-stats > "$RUN_DIR/ray-start.log" 2>&1
    log "ray head up at $RAY_ADDRESS (GPUs $GPUS)"
}

e2e_model_args() {  # $1: model type under scripts/models, e.g. qwen3-30B-A3B; prints the shell-quoted line
    PYTHONPATH="$REPO" "$PY" -c "from miles.utils.external_utils.command_utils import shell_safe_model_args; print(shell_safe_model_args('$1'))" 2>/dev/null | tail -1
}

e2e_submit_gateway() {  # $1: model-args line (shell-quoted), $2: serve args
    local runtime_env model_args_arr
    eval "model_args_arr=($1)"
    local cluster_env=""
    if [ -n "$EXTERNAL_RAY_GCS" ]; then
        # the driver and every worker must agree on the GCS, and NCCL/GLOO must pick the fabric interface
        cluster_env=", \"RAY_ADDRESS\": \"$EXTERNAL_RAY_GCS\""
        [ -n "$NET_IFNAME" ] && cluster_env="$cluster_env, \"NCCL_SOCKET_IFNAME\": \"$NET_IFNAME\", \"GLOO_SOCKET_IFNAME\": \"$NET_IFNAME\""
    fi
    runtime_env=$(cat <<JSON
{"env_vars": {"PYTHONUNBUFFERED": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "1", "NCCL_NVLS_ENABLE": "0",
 "no_proxy": "127.0.0.1,localhost${NODE_IPS:+,$NODE_IPS}", "MASTER_ADDR": "$HEAD_IP", "RAY_DEDUP_LOGS": "0",
 "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1", "MILES_E2E_FAMILY": "$MILES_E2E_FAMILY", "MILES_E2E_RUN": "$MILES_E2E_RUN",
 "PYTHONPATH": "$REPO:$MEGATRON_PATH:$SGLANG_PYTHONPATH"$cluster_env}}
JSON
)
    JOB_ID=e2e-$(date +%s)
    echo "python3 $REPO/serve_tinker.py $1 $2" > "$RUN_DIR/serve-command.txt"
    ray job submit --address "$RAY_ADDRESS" --submission-id "$JOB_ID" --runtime-env-json "$runtime_env" --no-wait \
        -- python3 "$REPO/serve_tinker.py" "${model_args_arr[@]}" $2 > "$RUN_DIR/ray-submit.log" 2>&1
    if [ -n "$EXTERNAL_RAY_GCS" ]; then
        # `ray job logs -f` against a foreign dashboard streams nothing; the driver log is a file on the head node
        local driver_log="$(e2e_ray_temp)/session_latest/logs/job-driver-$JOB_ID.log"
        ( for _ in $(seq 1 90); do [ -f "$driver_log" ] && break; sleep 2; done; tail -n +1 -F "$driver_log" ) > "$RUN_DIR/serve.log" 2>&1 &
    else
        ray job logs --address "$RAY_ADDRESS" -f "$JOB_ID" > "$RUN_DIR/serve.log" 2>&1 &
    fi
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
    local line
    if [ "$N_ADAPTERS" = "auto" ]; then
        line=$(grep -m1 -o "multi-LoRA capacity: [0-9]* slots" "$RUN_DIR/serve.log")
        [ -n "$line" ] || line=$(ray job logs --address "$RAY_ADDRESS" "$JOB_ID" 2>/dev/null | grep -m1 -o "multi-LoRA capacity: [0-9]* slots")
        echo "$line" | grep -o "[0-9]*"
    else
        echo "$N_ADAPTERS"
    fi
}
