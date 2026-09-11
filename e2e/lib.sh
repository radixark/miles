# Plumbing for auto_e2e_test.sh: submit the gateway to the Ray cluster, wait for it, read the
# slot count, and tear down. Source it after setting the knobs; it installs the teardown trap.
#
# Everything a run starts (the gateway job, the SGLang engines) carries MILES_E2E_RUN in its
# environment: preflight sweeps any process still carrying one from an earlier run, and
# teardown stops only the processes carrying this run's value, so a dying run never reaches a
# newer one. Nothing else on the nodes (another user's Ray cluster or engines) is ever touched.

: "${PY:=/opt/sglang/bin/python3}"
: "${MEGATRON_PATH:=/root/Megatron-LM}"
: "${SGLANG_PYTHONPATH:=}"         # an sglang source tree to put ahead of the installed one (empty: installed)
: "${RAY_GCS:=127.0.0.1:6379}"     # host:port of the cluster's GCS
: "${RAY_DASH:=127.0.0.1:8265}"    # host:port of its dashboard (jobs API)
: "${RAY_TEMP:=}"                  # the cluster's ray temp dir on this node; found from the raylet when empty
: "${HEAD_IP:=127.0.0.1}"          # MASTER_ADDR for torch.distributed: this (head) node's IP on the fabric
: "${NODE_IPS:=}"                  # comma-separated node IPs: no_proxy, and the nodes to sweep at teardown
: "${NET_IFNAME:=}"                # NCCL/GLOO socket interface, e.g. bond0
: "${MIN_FREE_GPUS:=16}"           # refuse to submit unless the cluster reports this many free GPUs
: "${READY_TIMEOUT:=3600}"
: "${KEEP_GATEWAY:=0}"             # 1: leave the gateway up after a passing run (debugging)
: "${KEEP_CKPT:=0}"                # 1: keep the run's exported adapter versions (about 1.5 GB each on disk) after it ends
: "${KEEP_VERSIONS:=2}"            # exported versions kept per client while it runs (0: all); a client only samples its latest
: "${MIN_FREE_GB:=150}"            # refuse to start unless RUN_ROOT's volume has this much room (slots x KEEP_VERSIONS x 1.5 GB)
: "${RUN_DIR:=$RUN_ROOT/$(date +%Y%m%d-%H%M%S)}"

export MILES_E2E_RUN="miles-e2e-$(basename "$RUN_DIR")-$$"
export RAY_ADDRESS=http://$RAY_DASH
JOB_ID=""
PRUNE_PID=""
GPU_SAMPLER_PID=""
GPU_JOBS=""
READY_S=""

log() { echo "[e2e $(date +%H:%M:%S)] $*"; }

own_chain() {  # this script and its ancestors: the only marked processes a sweep must leave alone
    local pid=$$ chain=" "
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

e2e_ray_temp() {  # where this node's raylet keeps the session (driver logs under session_latest/logs)
    if [ -n "$RAY_TEMP" ]; then echo "$RAY_TEMP"; return; fi
    pgrep -fa 'raylet' | grep -o -- '--temp_dir=[^ ]*' | head -1 | cut -d= -f2
}

e2e_sweep_other_nodes() {  # $1: environment-line pattern; runs e2e/sweep_node.sh on every other node through ray
    local ip
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

cluster_gpu_usage() {  # prints "<used> <total>" from ray status
    local line
    line=$(ray status --address "$RAY_GCS" 2>/dev/null | grep -E "^ *[0-9.]+/[0-9.]+ GPU" | head -1)
    [ -n "$line" ] || return 1
    echo "$(echo "$line" | sed 's#^ *\([0-9.]*\)/.*#\1#') $(echo "$line" | sed 's#.*/\([0-9.]*\) GPU.*#\1#')"
}

e2e_cleanup() {
    local rc=$?
    trap - EXIT
    if [ "$KEEP_GATEWAY" = "1" ] && [ $rc -eq 0 ]; then
        log "KEEP_GATEWAY=1: leaving the gateway up (job $JOB_ID, Tinker :$TINKER_PORT); the next run sweeps it"
        exit 0
    fi
    [ -n "$PRUNE_PID" ] && kill "$PRUNE_PID" 2>/dev/null || true
    e2e_stop_gpu_samplers
    [ -n "$JOB_ID" ] && ray job stop --address "$RAY_ADDRESS" "$JOB_ID" >/dev/null 2>&1 || true
    kill_marked "^MILES_E2E_RUN=$MILES_E2E_RUN\$"
    e2e_sweep_other_nodes "^MILES_E2E_RUN=$MILES_E2E_RUN\$"
    # every publish exported an adapter version to the shared volume; a few runs fill it
    [ "$KEEP_CKPT" = "1" ] || rm -rf "$RUN_DIR/ckpt"
    log "GPU memory after teardown (this node):"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/    /'
    log "logs: $RUN_DIR (serve.log, client.log, serve-command.txt)"
    if [ $rc -eq 0 ]; then log "E2E PASS"; else log "E2E FAIL (exit $rc)"; fi
    exit $rc
}
trap e2e_cleanup EXIT

e2e_preflight() {
    local usage
    [ -d "$MODEL" ] || { log "model dir $MODEL missing"; exit 2; }
    [ -f "$REPO/serve_tinker.py" ] || { log "$REPO is not a miles tree"; exit 2; }
    [ -z "$SGLANG_PYTHONPATH" ] || [ -d "$SGLANG_PYTHONPATH/sglang" ] || { log "SGLANG_PYTHONPATH $SGLANG_PYTHONPATH has no sglang package"; exit 2; }
    kill_marked "^MILES_E2E_RUN="  # leftovers of any earlier run
    e2e_sweep_other_nodes "^MILES_E2E_RUN="
    usage=$(cluster_gpu_usage) || { log "cannot read GPU usage from the Ray cluster at $RAY_GCS"; exit 2; }
    log "cluster $RAY_GCS: ${usage% *} of ${usage#* } GPUs in use"
    awk -v u="${usage% *}" -v t="${usage#* }" -v m="$MIN_FREE_GPUS" 'BEGIN { exit !(t - u >= m) }' \
        || { log "cluster has fewer than $MIN_FREE_GPUS free GPUs; someone else is using it"; exit 2; }
    if port_busy "$TINKER_PORT"; then log "port $TINKER_PORT is busy"; exit 2; fi
    mkdir -p "$RUN_DIR/ckpt"
    local free_gb
    free_gb=$(df -BG --output=avail "$RUN_DIR" | tail -1 | tr -dc '0-9')
    [ "${free_gb:-0}" -ge "$MIN_FREE_GB" ] || { log "only ${free_gb} GB free under $RUN_ROOT; the exported adapter versions need $MIN_FREE_GB GB (MIN_FREE_GB)"; exit 2; }
}

e2e_model_args() {  # $1: model type under scripts/models, e.g. qwen3-30B-A3B; prints the shell-quoted line
    PYTHONPATH="$REPO" "$PY" -c "from miles.utils.external_utils.command_utils import shell_safe_model_args; print(shell_safe_model_args('$1'))" 2>/dev/null | tail -1
}

e2e_submit_gateway() {  # $1: model-args line (shell-quoted), $2: serve args
    local runtime_env model_args_arr fabric_env="" driver_log
    eval "model_args_arr=($1)"
    # the driver and every worker must agree on the GCS, and NCCL/GLOO must pick the fabric interface
    [ -n "$NET_IFNAME" ] && fabric_env=", \"NCCL_SOCKET_IFNAME\": \"$NET_IFNAME\", \"GLOO_SOCKET_IFNAME\": \"$NET_IFNAME\""
    runtime_env=$(cat <<JSON
{"env_vars": {"PYTHONUNBUFFERED": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "1", "NCCL_NVLS_ENABLE": "0",
 "no_proxy": "127.0.0.1,localhost${NODE_IPS:+,$NODE_IPS}", "MASTER_ADDR": "$HEAD_IP", "RAY_DEDUP_LOGS": "0",
 "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1", "MILES_E2E_RUN": "$MILES_E2E_RUN", "RAY_ADDRESS": "$RAY_GCS",
 "PYTHONPATH": "$REPO:$MEGATRON_PATH${SGLANG_PYTHONPATH:+:$SGLANG_PYTHONPATH}"$fabric_env}}
JSON
)
    JOB_ID=e2e-$(date +%s)
    echo "python3 $REPO/serve_tinker.py $1 $2" > "$RUN_DIR/serve-command.txt"
    ray job submit --address "$RAY_ADDRESS" --submission-id "$JOB_ID" --runtime-env-json "$runtime_env" --no-wait \
        -- python3 "$REPO/serve_tinker.py" "${model_args_arr[@]}" $2 > "$RUN_DIR/ray-submit.log" 2>&1
    # `ray job logs -f` streams nothing on some dashboards; the driver log is a file on this node
    driver_log="$(e2e_ray_temp)/session_latest/logs/job-driver-$JOB_ID.log"
    ( for _ in $(seq 1 90); do [ -f "$driver_log" ] && break; sleep 2; done; tail -n +1 -F "$driver_log" ) > "$RUN_DIR/serve.log" 2>&1 &
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
    READY_S=$SECONDS
    log "gateway ready after $READY_S s"
    grep -m1 "agree with predicted\|diverge from predicted" "$RUN_DIR/serve.log" | sed 's/^/[e2e] /' || true
    grep -m1 "multi-LoRA capacity" "$RUN_DIR/serve.log" | sed 's/^/[e2e] /' || log "(no capacity line: explicit slot count)"
    grep -m1 "capacity is bound by" "$RUN_DIR/serve.log" | sed 's/^/[e2e] WARNING /' || true
}

e2e_prune_versions() {  # keep each client's newest KEEP_VERSIONS exported versions; the rest are never sampled again
    local dir
    for dir in "$RUN_DIR"/ckpt/*/sampler_weights; do
        [ -d "$dir" ] || continue
        # a client that has not published yet has no step-* dir: that is not an error
        ls -1dt "$dir"/step-* 2>/dev/null | tail -n +$((KEEP_VERSIONS + 1)) | xargs -r rm -rf || true
    done
}

e2e_start_pruning() {  # every publish exports ~1.5 GB to the shared volume; without this a few runs fill it
    [ "$KEEP_VERSIONS" -gt 0 ] || return 0
    # the loop must outlive one failed ls: no errexit/pipefail in the subshell
    ( set +e +o pipefail; while true; do sleep 60; e2e_prune_versions; done ) &
    PRUNE_PID=$!
}

e2e_start_gpu_samplers() {  # nvidia-smi every 15 s on every node while the clients run; the report reads the CSVs
    local ip job
    bash "$REPO/e2e/gpu_sampler.sh" "$RUN_DIR/gpu-$HEAD_IP.csv" 7200 &
    GPU_SAMPLER_PID=$!
    for ip in ${NODE_IPS//,/ }; do
        [ "$ip" = "$HEAD_IP" ] && continue
        job="gpu-$(date +%s)-${ip//./-}"
        ray job submit --address "$RAY_ADDRESS" --submission-id "$job" --no-wait \
            --entrypoint-resources "{\"node:$ip\": 0.001}" \
            --runtime-env-json "{\"env_vars\": {\"MILES_E2E_RUN\": \"$MILES_E2E_RUN\"}}" \
            -- bash "$REPO/e2e/gpu_sampler.sh" "$RUN_DIR/gpu-$ip.csv" 7200 > /dev/null 2>&1 && GPU_JOBS="$GPU_JOBS $job" || true
    done
}

e2e_stop_gpu_samplers() {
    local job
    [ -n "$GPU_SAMPLER_PID" ] && kill "$GPU_SAMPLER_PID" 2>/dev/null || true
    for job in $GPU_JOBS; do ray job stop --address "$RAY_ADDRESS" "$job" > /dev/null 2>&1 || true; done
    GPU_SAMPLER_PID=""; GPU_JOBS=""
}

e2e_report() {  # the box table for this run: setup, results, GPU peaks; also saved as report.txt
    PYTHONPATH="$REPO" "$PY" "$REPO/e2e/report.py" --run-dir "$RUN_DIR" \
        --knob TRAIN_NODES="$TRAIN_NODES" --knob TRAIN_GPUS="$TRAIN_GPUS" --knob TP="$TP" --knob EP="$EP" \
        --knob ROLLOUT_GPUS="$ROLLOUT_GPUS" --knob GPUS_PER_ENGINE="$GPUS_PER_ENGINE" \
        --knob GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" \
        --knob MODEL="$MODEL" --knob DATASET="$DATASET" --knob LORA_RANK="$LORA_RANK" --knob LR="$LR" --knob STEPS="$STEPS" \
        --knob PROMPTS_PER_STEP="$PROMPTS_PER_STEP" --knob SAMPLES_PER_PROMPT="$SAMPLES_PER_PROMPT" \
        --knob MAX_PROMPT_TOKENS="$MAX_PROMPT_TOKENS" --knob CONTEXT_LEN="$CONTEXT_LEN" --knob MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
        --knob SGLANG_MEM_FRACTION="$SGLANG_MEM_FRACTION" --knob SGLANG_MAX_RUNNING_REQUESTS="$SGLANG_MAX_RUNNING_REQUESTS" \
        --knob SGLANG_CUDA_GRAPH_MAX_BS="$SGLANG_CUDA_GRAPH_MAX_BS" --knob SGLANG_MOE_RUNNER="$SGLANG_MOE_RUNNER" \
        --knob SGLANG_MAX_LOADED_LORAS="$SGLANG_MAX_LOADED_LORAS" --knob N_ADAPTERS="$N_ADAPTERS" --knob READY_S="$READY_S" \
        || log "report failed"
}

e2e_resolved_slots() {  # the slot count the gateway ended up with
    if [ "$N_ADAPTERS" = "auto" ]; then
        grep -m1 -o "multi-LoRA capacity: [0-9]* slots" "$RUN_DIR/serve.log" | grep -o "[0-9]*"
    else
        echo "$N_ADAPTERS"
    fi
}
