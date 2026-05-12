#!/usr/bin/env bash
set -euo pipefail

# Host-side script to run startup commands on all related Kubernetes worker pods.
# Used after MPIJob pods are Running; see K8S_MULTI_NODE_RL.md in this directory.
#
# Behavior:
# 1) Finds running pods matching GPU_WORKER_PREFIX and CPU_WORKER_PREFIX.
# 2) Executes STARTUP_COMMANDS in each pod via "bash -lc".
# 3) Prints a success/failure summary.

NAMESPACE="${NAMESPACE:-nlp-train}"
GPU_WORKER_PREFIX="${GPU_WORKER_PREFIX:-ycy-miles-test-massnode-job-worker-}"
CPU_WORKER_PREFIX="${CPU_WORKER_PREFIX:-ycy-miles-test-massnode-job-worker-}"
KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"
DRY_RUN="${DRY_RUN:-0}"
FAIL_FAST="${FAIL_FAST:-0}"
PARALLEL="${PARALLEL:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

# Edit this block directly to customize startup actions for every target pod.
STARTUP_COMMANDS=$(cat <<'EOF'
pip install -e '/fs/nlp/chenweikai/workspace/public/sglang/python[all]'
pip install '/fs/nlp/chenweikai/workspace/assets/wheels/flashinfer_jit_cache-0.6.7+cu129-cp39-abi3-manylinux_2_28_x86_64.whl'
EOF
)

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

validate_positive_int() {
  local v="$1"
  [[ "${v}" =~ ^[1-9][0-9]*$ ]]
}

is_true() {
  case "$1" in
    1|true|TRUE|yes|YES|y|Y|on|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

list_running_pods_by_prefix() {
  local prefix="$1"
  "${KUBECTL_BIN}" get pods -n "${NAMESPACE}" \
    --field-selector=status.phase=Running \
    -o custom-columns=NAME:.metadata.name --no-headers \
    | grep -E "^${prefix}[0-9]+$" \
    | sort -V || true
}

exec_startup_commands() {
  local pod="$1"

  if is_true "${DRY_RUN}"; then
    log "[DRY-RUN] ${KUBECTL_BIN} exec -n ${NAMESPACE} ${pod} -- bash -lc '<startup commands>'"
    printf '%s\n' "${STARTUP_COMMANDS}"
    return 0
  fi

  "${KUBECTL_BIN}" exec -n "${NAMESPACE}" "${pod}" -- bash -lc "${STARTUP_COMMANDS}"
}

run_single_pod() {
  local pod="$1"
  local ok_file="$2"
  local fail_file="$3"

  log "Running startup commands on ${pod}"
  if exec_startup_commands "${pod}"; then
    printf '%s\n' "${pod}" >> "${ok_file}"
    log "Startup commands succeeded on ${pod}"
  else
    printf '%s\n' "${pod}" >> "${fail_file}"
    log "Startup commands failed on ${pod}"
    return 1
  fi
}

main() {
  require_cmd "${KUBECTL_BIN}"

  log "Namespace: ${NAMESPACE}"
  log "GPU worker prefix: ${GPU_WORKER_PREFIX}"
  log "CPU worker prefix: ${CPU_WORKER_PREFIX}"
  log "Dry run: ${DRY_RUN}"
  log "Fail fast: ${FAIL_FAST}"
  log "Parallel: ${PARALLEL}"
  log "Max parallel: ${MAX_PARALLEL}"

  validate_positive_int "${MAX_PARALLEL}" || die "MAX_PARALLEL must be a positive integer, got: ${MAX_PARALLEL}"

  local gpu_pods cpu_pods
  gpu_pods="$(list_running_pods_by_prefix "${GPU_WORKER_PREFIX}")"
  cpu_pods="$(list_running_pods_by_prefix "${CPU_WORKER_PREFIX}")"

  local -a all_pods
  all_pods=()

  if [[ -n "${gpu_pods}" ]]; then
    while read -r pod; do
      [[ -n "${pod}" ]] || continue
      all_pods+=("${pod}")
    done <<< "${gpu_pods}"
  fi

  if [[ -n "${cpu_pods}" ]]; then
    while read -r pod; do
      [[ -n "${pod}" ]] || continue
      all_pods+=("${pod}")
    done <<< "${cpu_pods}"
  fi

  (( ${#all_pods[@]} > 0 )) || die "No running pods found for prefixes: ${GPU_WORKER_PREFIX}, ${CPU_WORKER_PREFIX}"

  log "Found ${#all_pods[@]} target pods"

  local -a ok_pods failed_pods
  ok_pods=()
  failed_pods=()

  local tmp_dir ok_file fail_file
  tmp_dir="$(mktemp -d)"
  ok_file="${tmp_dir}/ok.txt"
  fail_file="${tmp_dir}/fail.txt"
  trap 'rm -rf "${tmp_dir}"' EXIT

  local pod
  if is_true "${PARALLEL}"; then
    if is_true "${FAIL_FAST}"; then
      log "FAIL_FAST is ignored when PARALLEL is enabled"
    fi

    local -a pids
    pids=()

    for pod in "${all_pods[@]}"; do
      run_single_pod "${pod}" "${ok_file}" "${fail_file}" &
      pids+=("$!")

      if (( ${#pids[@]} >= MAX_PARALLEL )); then
        local pid
        for pid in "${pids[@]}"; do
          wait "${pid}" || true
        done
        pids=()
      fi
    done

    if (( ${#pids[@]} > 0 )); then
      local pid
      for pid in "${pids[@]}"; do
        wait "${pid}" || true
      done
    fi
  else
    for pod in "${all_pods[@]}"; do
      if run_single_pod "${pod}" "${ok_file}" "${fail_file}"; then
        :
      else
        if is_true "${FAIL_FAST}"; then
          die "Aborting because FAIL_FAST is enabled"
        fi
      fi
    done
  fi

  if [[ -s "${ok_file}" ]]; then
    mapfile -t ok_pods < "${ok_file}"
  fi

  if [[ -s "${fail_file}" ]]; then
    mapfile -t failed_pods < "${fail_file}"
  fi

  log "Startup execution finished. Success=${#ok_pods[@]}, Failed=${#failed_pods[@]}"

  if (( ${#failed_pods[@]} > 0 )); then
    log "Failed pods: ${failed_pods[*]}"
    exit 1
  fi
}

main "$@"
