#!/usr/bin/env bash
set -euo pipefail

# Minimal Ray bootstrap for K8s MPIJob workers:
# - Head: <GPU_WORKER_PREFIX><HEAD_SUFFIX> (default: ...worker-0)
# - Other GPU workers + optional CPU workers join the head.
# - Exports MASTER_ADDR on the head pod to the head Pod IP (same convention as ray_init.sh).
#
# Recovery after worker crash/OOM (ray job submit / Dashboard dead): from host with kubectl,
# run this script again; do not rely on a partially dead cluster.
#
# No patches, no custom Ray --resources, no rollout/actor placement logic.

NAMESPACE="${NAMESPACE:-nlp-train}"
JOB_NAME="${JOB_NAME:-ycy-miles-test-m25-job}"
GPU_WORKER_PREFIX="${GPU_WORKER_PREFIX:-${JOB_NAME}-worker-}"
HEAD_SUFFIX="${HEAD_SUFFIX:-0}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
HEAD_NODE_IP="${HEAD_NODE_IP:-auto}"

# Set non-empty only if you have separate CPU-worker pods (MPIJob name prefix + worker ordinal).
CPU_WORKER_PREFIX="${CPU_WORKER_PREFIX:-}"

CPU_WORKER_NUM_GPUS="${CPU_WORKER_NUM_GPUS:-0}"

KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"
DRY_RUN="${DRY_RUN:-0}"

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
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
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
  [[ -n "${prefix}" ]] || return 0
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

ray_stop_soft() {
  local pod="$1"
  log "Stopping Ray (if any) on ${pod}"
  exec_in_pod_mutating "${pod}" 'ray stop --force || true'
}

start_head() {
  local pod="$1"
  local head_ip="$2"
  local head_node_ip="$3"
  local gpus="$4"

  log "Starting Ray head on ${pod} (MASTER_ADDR/Pod IP=${head_ip}, num-gpus=${gpus})"
  exec_in_pod_mutating "${pod}" "
export MASTER_ADDR='${head_ip}'
ray start --head \
  --node-ip-address='${head_node_ip}' \
  --num-gpus='${gpus}' \
  --disable-usage-stats \
  --port='${RAY_PORT}' \
  --dashboard-port='${RAY_DASHBOARD_PORT}' \
  --include-dashboard=True \
  --dashboard-host=0.0.0.0
"
}

join_worker() {
  local pod="$1"
  local pod_ip_addr="$2"
  local gpus="$3"
  local head_ip="$4"

  log "Joining ${pod} (ip=${pod_ip_addr}, gpus=${gpus}) -> ${head_ip}:${RAY_PORT}"
  exec_in_pod_mutating "${pod}" "
export MASTER_ADDR='${head_ip}'
ray start \
  --address='${head_ip}:${RAY_PORT}' \
  --num-gpus='${gpus}' \
  --node-ip-address='${pod_ip_addr}' \
  --disable-usage-stats
"
}

main() {
  require_cmd "${KUBECTL_BIN}"

  local head_pod="${GPU_WORKER_PREFIX}${HEAD_SUFFIX}"

  log "Namespace: ${NAMESPACE}"
  log "GPU worker prefix: ${GPU_WORKER_PREFIX}"
  log "Head pod (MASTER_ADDR target): ${head_pod}"
  log "CPU worker prefix: ${CPU_WORKER_PREFIX:-<none>}"
  log "Dry run: ${DRY_RUN}"

  local gpu_pods cpu_pods
  gpu_pods="$(list_running_pods_by_prefix "${GPU_WORKER_PREFIX}")"

  [[ -n "${gpu_pods}" ]] || die "No running GPU worker pods for prefix: ${GPU_WORKER_PREFIX}"
  echo "${gpu_pods}" | grep -qx "${head_pod}" || die "Head pod not running: ${head_pod}"

  cpu_pods=""
  if [[ -n "${CPU_WORKER_PREFIX}" ]]; then
    cpu_pods="$(list_running_pods_by_prefix "${CPU_WORKER_PREFIX}")"
  fi

  local head_ip head_gpus effective_head_node_ip

  ray_stop_soft "${head_pod}"
  head_ip="$(pod_ip "${head_pod}")"
  head_gpus="$(pod_num_gpus "${head_pod}")"

  effective_head_node_ip="${HEAD_NODE_IP}"
  if [[ "${effective_head_node_ip}" == "auto" ]]; then
    effective_head_node_ip="${head_ip}"
  fi
  log "Head IP for Ray/cluster (MASTER_ADDR): ${head_ip}; node-ip-address: ${effective_head_node_ip}"

  start_head "${head_pod}" "${head_ip}" "${effective_head_node_ip}" "${head_gpus}"

  local pod
  while read -r pod; do
    [[ -n "${pod}" ]] || continue
    [[ "${pod}" == "${head_pod}" ]] && continue

    ray_stop_soft "${pod}"
    join_worker "${pod}" "$(pod_ip "${pod}")" "$(pod_num_gpus "${pod}")" "${head_ip}"
  done <<< "${gpu_pods}"

  if [[ -n "${cpu_pods}" ]]; then
    while read -r pod; do
      [[ -n "${pod}" ]] || continue

      ray_stop_soft "${pod}"
      join_worker "${pod}" "$(pod_ip "${pod}")" "${CPU_WORKER_NUM_GPUS}" "${head_ip}"
    done <<< "${cpu_pods}"
  fi

  log "Ray cluster ready."
  log "Head address: ${head_ip}:${RAY_PORT}"
  log "Tip: ${KUBECTL_BIN} exec -n ${NAMESPACE} ${head_pod} -- ray status"
}

main "$@"
