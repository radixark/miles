#!/usr/bin/env bash
set -euo pipefail

# Host-side script to stop Ray services inside Kubernetes worker pods.
#
# Behavior:
# 1) Finds running GPU worker pods by prefix.
# 2) Finds running CPU worker pods by prefix.
# 3) Runs "ray stop --force" in each matched pod.
# 4) Optionally kills sglang/python processes if requested.

NAMESPACE="${NAMESPACE:-nlp}"
GPU_WORKER_PREFIX="${GPU_WORKER_PREFIX:-chenweikai-slime-massnode-job-worker-}"
CPU_WORKER_PREFIX="${CPU_WORKER_PREFIX:-chenweikai-slime-cpu-job-worker-}"

KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"
DRY_RUN="${DRY_RUN:-0}"
STOP_SGLANG="${STOP_SGLANG:-1}"
STOP_PYTHON="${STOP_PYTHON:-1}"

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

is_enabled() {
  case "$1" in
    1|true|TRUE|yes|YES|y|Y|on|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

exec_in_pod() {
  local pod="$1"
  shift
  "${KUBECTL_BIN}" exec -n "${NAMESPACE}" "${pod}" -- bash -lc "$*"
}

exec_in_pod_mutating() {
  local pod="$1"
  shift
  local cmd="$*"

  if is_enabled "${DRY_RUN}"; then
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

stop_ray_in_pod() {
  local pod="$1"
  local -a commands
  commands=("ray stop --force || true")

  if is_enabled "${STOP_SGLANG}"; then
    commands=("pkill -9 sglang || true" "${commands[@]}")
  fi

  if is_enabled "${STOP_PYTHON}"; then
    commands+=("pkill -9 python || true")
  fi

  log "Stopping services in ${pod}"
  exec_in_pod_mutating "${pod}" "$(printf '%s\n' "${commands[@]}")"
}

main() {
  require_cmd "${KUBECTL_BIN}"

  log "Namespace: ${NAMESPACE}"
  log "GPU worker prefix: ${GPU_WORKER_PREFIX}"
  log "CPU worker prefix: ${CPU_WORKER_PREFIX}"
  log "Dry run: ${DRY_RUN}"
  log "Stop sglang: ${STOP_SGLANG}"
  log "Stop python: ${STOP_PYTHON}"

  local gpu_pods cpu_pods
  gpu_pods="$(list_running_pods_by_prefix "${GPU_WORKER_PREFIX}")"
  cpu_pods="$(list_running_pods_by_prefix "${CPU_WORKER_PREFIX}")"

  if [[ -z "${gpu_pods}" && -z "${cpu_pods}" ]]; then
    die "No running worker pods found for prefixes: ${GPU_WORKER_PREFIX}, ${CPU_WORKER_PREFIX}"
  fi

  local total_stopped=0
  local pod

  if [[ -n "${gpu_pods}" ]]; then
    while read -r pod; do
      [[ -n "${pod}" ]] || continue
      stop_ray_in_pod "${pod}"
      total_stopped=$((total_stopped + 1))
    done <<< "${gpu_pods}"
  fi

  if [[ -n "${cpu_pods}" ]]; then
    while read -r pod; do
      [[ -n "${pod}" ]] || continue
      stop_ray_in_pod "${pod}"
      total_stopped=$((total_stopped + 1))
    done <<< "${cpu_pods}"
  fi

  log "Stop complete. Processed pods: ${total_stopped}"
}

main "$@"
