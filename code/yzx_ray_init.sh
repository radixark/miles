#!/usr/bin/env bash
set -euo pipefail

# Host-side script to bootstrap Ray inside Kubernetes pods.
# Workflow with MPIJob YAML / k8s_init.sh: see K8S_MULTI_NODE_RL.md in this directory.
#
# Behavior:
# 1) Selects <GPU_WORKER_PREFIX>-0 as Ray head.
# 2) Detects pod IP via Kubernetes PodIP (with hostname -I fallback).
# 3) Detects GPU count inside each pod (NVIDIA or ROCm, else 0).
# 4) Starts head first, then all other GPU workers join, then CPU workers join.

NAMESPACE="${NAMESPACE:-nlp-train}"
GPU_WORKER_PREFIX="${GPU_WORKER_PREFIX:-ycy-miles-test-massnode-job-worker-}"
# CPU_WORKER_PREFIX="${CPU_WORKER_PREFIX:-chenweikai-slime-cpu-job-worker-}"
HEAD_SUFFIX="${HEAD_SUFFIX:-0}"

RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
# HEAD_NODE_IP="${HEAD_NODE_IP:-127.0.0.1}"
HEAD_NODE_IP="${HEAD_NODE_IP:-auto}"

# Number of non-head GPU worker pods to mark as rollout nodes.
# Rollout pods are selected from the tail of non-head GPU pods in sorted order.
ROLLOUT_NODE_COUNT="${ROLLOUT_NODE_COUNT:-8}"

HEAD_RESOURCES_JSON="${HEAD_RESOURCES_JSON:-{\"actor_node\": 1}}"
GPU_WORKER_RESOURCES_JSON="${GPU_WORKER_RESOURCES_JSON:-{\"rollout_node\": 1}}"
CPU_WORKER_RESOURCES_JSON="${CPU_WORKER_RESOURCES_JSON:-{\"agent_node\": 1}}"
GPU_ACTOR_RESOURCES_JSON="${GPU_ACTOR_RESOURCES_JSON:-{\"actor_node\": 1}}"
CPU_WORKER_NUM_GPUS="${CPU_WORKER_NUM_GPUS:-0}"
JOIN_WORKER_CONCURRENCY="${JOIN_WORKER_CONCURRENCY:-0}"

KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"
DRY_RUN="${DRY_RUN:-0}"

# Paths must exist **inside the pod** (e.g. under mounted /fs). Missing files/dirs are skipped with a warning.
SGLANG_PATCH_FILE="${SGLANG_PATCH_FILE:-/fs/nlp/chenweikai/workspace/assets/patches/sglang/u2.patch}"
SWE_AGENT_PATCH_FILE="${SWE_AGENT_PATCH_FILE:-/fs/nlp/chenweikai/workspace/assets/patches/SWE-agent/env.patch}"
MINISWE_AGENT_SRC="${MINISWE_AGENT_SRC:-/fs/nlp/chenweikai/workspace/public/mini-swe-agent}"

JOIN_JOB_PIDS=()
JOIN_JOB_LABELS=()

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

exec_in_pod() {
  local pod="$1"
  shift
  "${KUBECTL_BIN}" exec -n "${NAMESPACE}" "${pod}" -- bash -lc "$*"
}

is_dry_run_enabled() {
  case "${DRY_RUN}" in
    1|true|TRUE|yes|YES|y|Y|on|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

exec_in_pod_mutating() {
  local pod="$1"
  shift
  local cmd="$*"

  if is_dry_run_enabled; then
    log "[DRY-RUN] ${KUBECTL_BIN} exec -n ${NAMESPACE} ${pod} -- bash -lc '<command>'"
    printf '%s\n' "${cmd}"
    return 0
  fi

  exec_in_pod "${pod}" "${cmd}"
}

list_running_pods_by_prefix() {
  local prefix="$1"
  "${KUBECTL_BIN}" get pods -n "${NAMESPACE}" \
    --field-selector=status.phase=Running \
    -o custom-columns=NAME:.metadata.name --no-headers \
    | grep -E "^${prefix}[0-9]+$" \
    | sort -V || true
}

pod_ip() {
  local pod="$1"
  local ip
  ip="$("${KUBECTL_BIN}" get pod -n "${NAMESPACE}" "${pod}" -o jsonpath='{.status.podIP}' 2>/dev/null || true)"
  if [[ -n "${ip}" ]]; then
    echo "${ip}"
    return 0
  fi
  # Fallback for unexpected API failures.
  exec_in_pod "${pod}" "hostname -I | awk '{print \$1}'"
}

pod_num_gpus() {
  local pod="$1"
  exec_in_pod "${pod}" '
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name --format=csv,noheader | wc -l
elif command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi -i | grep -E "^GPU[0-9]+" | wc -l
else
  echo 0
fi
'
}

cleanup_ray() {
  local pod="$1"
  local q_sg q_swe q_mini
  q_sg=$(printf '%q' "${SGLANG_PATCH_FILE}")
  q_swe=$(printf '%q' "${SWE_AGENT_PATCH_FILE}")
  q_mini=$(printf '%q' "${MINISWE_AGENT_SRC}")

  log "Cleaning processes in ${pod}"
  exec_in_pod_mutating "${pod}" '
pkill -9 sglang || true
ray stop --force || true
pkill -9 python || true
'

  log "Applying sglang patch in ${pod} (if present)"
  exec_in_pod_mutating "${pod}" "patch_file=${q_sg}; if [[ ! -f \"\${patch_file}\" ]]; then echo \"WARN: sglang patch missing, skip: \${patch_file}\"; elif [[ ! -d /opt/extra_modules/sglang ]]; then echo \"WARN: /opt/extra_modules/sglang missing, skip patch\"; else cd /opt/extra_modules/sglang && if git apply --reverse --check \"\${patch_file}\" >/dev/null 2>&1; then echo \"Patch already applied: \${patch_file}\"; elif git apply --check \"\${patch_file}\" >/dev/null 2>&1; then git apply \"\${patch_file}\"; else echo \"WARN: sglang patch not cleanly applicable, skip: \${patch_file}\"; fi; fi"

  log "Applying SWE-agent patch in ${pod} (if present)"
  exec_in_pod_mutating "${pod}" "patch_file=${q_swe}; if [[ ! -f \"\${patch_file}\" ]]; then echo \"WARN: SWE-agent patch missing, skip: \${patch_file}\"; elif [[ ! -d /opt/extra_modules/SWE-agent ]]; then echo \"WARN: /opt/extra_modules/SWE-agent missing, skip patch\"; else cd /opt/extra_modules/SWE-agent && if git apply --reverse --check \"\${patch_file}\" >/dev/null 2>&1; then echo \"Patch already applied: \${patch_file}\"; elif git apply --check \"\${patch_file}\" >/dev/null 2>&1; then git apply \"\${patch_file}\"; else echo \"WARN: SWE-agent patch not cleanly applicable, skip: \${patch_file}\"; fi; fi"

  log "Installing minisweagent in ${pod} (if src present)"
  exec_in_pod_mutating "${pod}" "src=${q_mini}; if [[ ! -d \"\${src}\" ]]; then echo \"WARN: minisweagent src missing, skip: \${src}\"; elif python -c \"import minisweagent\" >/dev/null 2>&1; then echo \"minisweagent already installed\"; else pip install -e \"\${src}\"; fi"
}

start_head() {
  local pod="$1"
  local head_ip="$2"
  local head_node_ip="$3"
  local gpus="$4"

  log "Starting Ray head on ${pod} (ip=${head_ip}, gpus=${gpus})"
  exec_in_pod_mutating "${pod}" "
export MASTER_ADDR='${head_ip}'
ray start --head \
  --node-ip-address='${head_node_ip}' \
  --num-gpus='${gpus}' \
  --disable-usage-stats \
  --port='${RAY_PORT}' \
  --dashboard-port='${RAY_DASHBOARD_PORT}' \
  --include-dashboard=True \
  --resources '${HEAD_RESOURCES_JSON}'
"
}

validate_non_negative_integer() {
  local v="$1"
  [[ "${v}" =~ ^[0-9]+$ ]]
}

wait_for_one_join_job() {
  local finished_pid=""
  local status=0
  local i

  if ! wait -n -p finished_pid; then
    status=$?
  fi

  for i in "${!JOIN_JOB_PIDS[@]}"; do
    if [[ "${JOIN_JOB_PIDS[i]}" != "${finished_pid}" ]]; then
      continue
    fi

    local label="${JOIN_JOB_LABELS[i]}"
    unset 'JOIN_JOB_PIDS[i]'
    unset 'JOIN_JOB_LABELS[i]'
    JOIN_JOB_PIDS=("${JOIN_JOB_PIDS[@]}")
    JOIN_JOB_LABELS=("${JOIN_JOB_LABELS[@]}")

    if (( status == 0 )); then
      log "Worker bootstrap finished: ${label}"
      return 0
    fi

    die "Worker bootstrap failed: ${label} (exit=${status})"
  done

  (( status == 0 )) || die "Worker bootstrap failed (pid=${finished_pid}, exit=${status})"
}

wait_for_join_slot() {
  if (( JOIN_WORKER_CONCURRENCY <= 0 )); then
    return 0
  fi

  while (( ${#JOIN_JOB_PIDS[@]} >= JOIN_WORKER_CONCURRENCY )); do
    wait_for_one_join_job
  done
}

wait_for_all_join_jobs() {
  while (( ${#JOIN_JOB_PIDS[@]} > 0 )); do
    wait_for_one_join_job
  done
}

bootstrap_worker() {
  local pod="$1"
  local resources_json="$2"
  local head_ip="$3"
  local gpu_override="${4:-auto}"

  cleanup_ray "${pod}"

  local ip gpus
  ip="$(pod_ip "${pod}")"
  if [[ "${gpu_override}" == "auto" ]]; then
    gpus="$(pod_num_gpus "${pod}")"
  else
    gpus="${gpu_override}"
  fi

  join_worker "${pod}" "${ip}" "${gpus}" "${resources_json}" "${head_ip}"
}

start_join_job() {
  local pod="$1"
  local resources_json="$2"
  local head_ip="$3"
  local gpu_override="${4:-auto}"

  wait_for_join_slot
  log "Scheduling ${pod} worker bootstrap in background"

  bootstrap_worker "${pod}" "${resources_json}" "${head_ip}" "${gpu_override}" &
  JOIN_JOB_PIDS+=("$!")
  JOIN_JOB_LABELS+=("${pod}")
}

join_worker() {
  local pod="$1"
  local pod_ip_addr="$2"
  local gpus="$3"
  local resources_json="$4"
  local head_ip="$5"

  log "Joining ${pod} (ip=${pod_ip_addr}, gpus=${gpus}) -> ${head_ip}:${RAY_PORT}"

  exec_in_pod_mutating "${pod}" "
ray start \
  --address='${head_ip}:${RAY_PORT}' \
  --num-gpus='${gpus}' \
  --node-ip-address='${pod_ip_addr}' \
  --disable-usage-stats \
  --resources '${resources_json}'
"
}

main() {
  require_cmd "${KUBECTL_BIN}"

  local head_pod="${GPU_WORKER_PREFIX}${HEAD_SUFFIX}"

  log "Namespace: ${NAMESPACE}"
  log "GPU worker prefix: ${GPU_WORKER_PREFIX}"
  log "CPU worker prefix: ${CPU_WORKER_PREFIX}"
  log "Head pod: ${head_pod}"
  log "Head node-ip-address mode: ${HEAD_NODE_IP}"
  log "Rollout node count: ${ROLLOUT_NODE_COUNT}"
  log "Join worker concurrency: ${JOIN_WORKER_CONCURRENCY} (0 means unlimited)"
  log "Dry run: ${DRY_RUN}"
  log "CPU worker num-gpus override: ${CPU_WORKER_NUM_GPUS}"

  local gpu_pods cpu_pods
  gpu_pods="$(list_running_pods_by_prefix "${GPU_WORKER_PREFIX}")"
  cpu_pods="$(list_running_pods_by_prefix "${CPU_WORKER_PREFIX}")"

  [[ -n "${gpu_pods}" ]] || die "No running GPU worker pods found for prefix: ${GPU_WORKER_PREFIX}"
  echo "${gpu_pods}" | grep -qx "${head_pod}" || die "Head pod not running: ${head_pod}"
  validate_non_negative_integer "${ROLLOUT_NODE_COUNT}" || die "ROLLOUT_NODE_COUNT must be a non-negative integer, got: ${ROLLOUT_NODE_COUNT}"
  validate_non_negative_integer "${JOIN_WORKER_CONCURRENCY}" || die "JOIN_WORKER_CONCURRENCY must be a non-negative integer, got: ${JOIN_WORKER_CONCURRENCY}"

  local -a gpu_pod_arr non_head_gpu_pods
  gpu_pod_arr=()
  non_head_gpu_pods=()
  mapfile -t gpu_pod_arr <<< "${gpu_pods}"

  local pod
  for pod in "${gpu_pod_arr[@]}"; do
    [[ -n "${pod}" ]] || continue
    [[ "${pod}" == "${head_pod}" ]] && continue
    non_head_gpu_pods+=("${pod}")
  done

  local total_gpu_pod_count="${#gpu_pod_arr[@]}"
  local non_head_gpu_count="${#non_head_gpu_pods[@]}"
  local rollout_count="${ROLLOUT_NODE_COUNT}"

  # Head pod is reserved for actor role, so rollout assignment only applies to non-head GPU pods.
  if (( rollout_count > non_head_gpu_count )); then
    log "Requested rollout GPU nodes (${rollout_count}) exceed non-head GPU pods (${non_head_gpu_count}); clamping."
    rollout_count="${non_head_gpu_count}"
  fi

  local rollout_start_index=$((non_head_gpu_count - rollout_count))
  if (( rollout_start_index < 0 )); then
    rollout_start_index=0
  fi

  log "GPU pods total=${total_gpu_pod_count}, non-head=${non_head_gpu_count}, rollout=${rollout_count}, actor(non-head)=$((non_head_gpu_count - rollout_count))"

  local head_ip head_gpus
  cleanup_ray "${head_pod}"
  head_ip="$(pod_ip "${head_pod}")"
  head_gpus="$(pod_num_gpus "${head_pod}")"

  local effective_head_node_ip="${HEAD_NODE_IP}"
  if [[ "${effective_head_node_ip}" == "auto" ]]; then
    # Auto mode: use Kubernetes PodIP of the head pod.
    effective_head_node_ip="${head_ip}"
  fi
  log "Resolved head node-ip-address: ${effective_head_node_ip}"

  start_head "${head_pod}" "${head_ip}" "${effective_head_node_ip}" "${head_gpus}"

  local i=0
  for pod in "${non_head_gpu_pods[@]}"; do
    [[ -n "${pod}" ]] || continue

    local resources_json="${GPU_ACTOR_RESOURCES_JSON}"
    if (( i >= rollout_start_index )); then
      resources_json="${GPU_WORKER_RESOURCES_JSON}"
      log "Assign ${pod} as rollout node"
    else
      log "Assign ${pod} as actor node"
    fi

    start_join_job "${pod}" "${resources_json}" "${head_ip}" "auto"
    i=$((i + 1))
  done

  if [[ -n "${cpu_pods}" ]]; then
    while read -r pod; do
      [[ -n "${pod}" ]] || continue

      log "Assign ${pod} as agent node"
      start_join_job "${pod}" "${CPU_WORKER_RESOURCES_JSON}" "${head_ip}" "${CPU_WORKER_NUM_GPUS}"
    done <<< "${cpu_pods}"
  fi

  wait_for_all_join_jobs

  log "Ray cluster bootstrap complete."
  log "Head address: ${head_ip}:${RAY_PORT}"
  log "Tip: ${KUBECTL_BIN} exec -n ${NAMESPACE} ${head_pod} -- ray status"
}

main "$@"
