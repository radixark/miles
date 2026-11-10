#!/usr/bin/env bash
# Body of one launch.sh job; runs on the allocated MI35x node.
#
# Starts the ROCm CI container with only the GPUs the scheduler assigned this
# job, so two 4-GPU shards can share a node, and runs container_steps.sh in it.
#
# Docker containers are not part of the scheduler job, so a job killed before
# its EXIT trap runs leaves its container, and that container's GPU memory,
# behind. Every job therefore names its container ci_miles_<jobid>, first
# removes such containers whose job has ended, then checks that its own GPUs
# are actually free.
#
# The checkout is copied to node-local scratch first: the container runs as
# root and writes into its checkout mount, which a root-squashed NFS home would
# refuse.

exec 2>&1   # Spur's batch --output captures stdout only
set -euo pipefail

: "${CI_SRC:?}" "${CI_IMAGE:?}" "${CI_NODE_SCRATCH:?}" "${CI_EXIT_FILE:?}" "${CI_SUMMARY_FILE:?}"
CI_GPUS="${CI_GPUS:-4}"
JOB_ID="${SLURM_JOB_ID:-${SPUR_JOB_ID:-$$}}"
WS="$CI_NODE_SCRATCH/ci-$JOB_ID"
DATA="$CI_NODE_SCRATCH/miles-data"
CONTAINER="ci_miles_$JOB_ID"
STEPS=".github/workflows/scripts/spur/container_steps.sh"

# Passed by name into the container (values never appear on a command line).
CONTAINER_ENV=(
    CI_COMMAND CI_SKIP_DEPENDENCY_INSTALL SGLANG_PR MEGATRON_PR BRIDGE_REF
    MILES_HARDWARE_PLATFORM GITHUB_COMMIT_NAME WANDB_API_KEY HF_TOKEN
    GITHUB_ACTIONS GITHUB_SHA GITHUB_REF GITHUB_EVENT_NAME GITHUB_RUN_ID
    GITHUB_RUN_ATTEMPT GITHUB_REPOSITORY GITHUB_SERVER_URL
)

echo "[miles-ci-job] start on $(hostname) job=$JOB_ID gpus=$CI_GPUS image=$CI_IMAGE"

remove_job() {  # remove_job CONTAINER WORKSPACE
    # Root-owned files from the container can only be removed from inside it.
    docker exec "$1" bash -c 'rm -rf /miles-checkout/* /miles-checkout/.[!.]*' >/dev/null 2>&1 || true
    docker rm -f "$1" >/dev/null 2>&1 || true
    rm -rf "$2" 2>/dev/null || true
}

finish() {
    local rc=$?
    cp "$WS/.ci-step-summary.md" "$CI_SUMMARY_FILE" 2>/dev/null || true
    remove_job "$CONTAINER" "$WS"
    echo "$rc" > "$CI_EXIT_FILE"
    echo "[miles-ci-job] exit $rc"
}
trap finish EXIT

# A job is gone once the scheduler no longer lists it. Ask twice: the
# controller occasionally answers with an empty queue for a live job.
job_alive() {
    local _
    for _ in 1 2; do
        [[ -n "$(squeue -h -j "$1" -o %T 2>/dev/null)" ]] && return 0
        sleep 5
    done
    return 1
}

remove_stale_containers() {
    local name id
    for name in $(docker ps -a --format '{{.Names}}' | grep -E '^ci_miles_[0-9]+$' || true); do
        id="${name#ci_miles_}"
        [[ "$id" == "$JOB_ID" ]] && continue
        job_alive "$id" && continue
        echo "[miles-ci-job] removing $name: job $id has ended"
        remove_job "$name" "$CI_NODE_SCRATCH/ci-$id"
    done
}

# GPU indices as the scheduler reports them (ROCm order) -> /dev/dri/renderD*.
# ROCm numbers GPUs in KFD topology order, skipping CPU nodes (gpu_id 0).
render_devices() {
    local topo=/sys/class/kfd/kfd/topology/nodes want=",$1," idx=0 node minor
    local out=()
    for node in $(ls "$topo" | sort -n); do
        [[ "$(cat "$topo/$node/gpu_id")" == 0 ]] && continue
        if [[ "$want" == *",$idx,"* ]]; then
            minor=$(awk '$1 == "drm_render_minor" {print $2}' "$topo/$node/properties")
            out+=("/dev/dri/renderD$minor")
        fi
        idx=$((idx + 1))
    done
    echo "${out[*]}"
}

# Fail unless each of our GPUs is under 5% VRAM. Memory freed by a removed
# container can take a few seconds to drop.
check_gpus_free() {  # check_gpus_free DEVICE...
    local dev minor used total busy _
    for _ in 1 2 3 4 5 6; do
        busy=""
        for dev in "$@"; do
            minor="${dev##*renderD}"
            used=$(cat "/sys/class/drm/renderD$minor/device/mem_info_vram_used")
            total=$(cat "/sys/class/drm/renderD$minor/device/mem_info_vram_total")
            (( used * 100 > total * 5 )) && busy+=" $dev:$((used >> 30))GiB"
        done
        [[ -z "$busy" ]] && return 0
        sleep 10
    done
    echo "ERROR: GPUs assigned to this job still hold memory:$busy" >&2
    echo "Containers on this node:" >&2
    docker ps -a --format '  {{.Names}} {{.Status}} {{.Image}}' >&2 || true
    return 1
}

retry() {  # retry ATTEMPTS COMMAND...
    local n="$1" i
    shift
    for i in $(seq 1 "$n"); do
        "$@" && return 0
        (( i < n )) && sleep $((i * 15))
    done
    return 1
}

mkdir -p "$WS" "$DATA/models" "$DATA/datasets" "$DATA/hf_cache"
cp -a "$CI_SRC/." "$WS/"

remove_stale_containers
DEVICE_ARGS=(--device=/dev/kfd)
if (( CI_GPUS == 8 )); then
    DEVICE_ARGS+=(--device=/dev/dri)   # --exclusive: the node is ours
else
    gpus="${SPUR_JOB_GPUS:-${ROCR_VISIBLE_DEVICES:-}}"
    read -ra devs <<< "$(render_devices "$gpus")"
    if (( ${#devs[@]} != CI_GPUS )); then
        echo "ERROR: asked for $CI_GPUS GPUs, scheduler assigned '$gpus' -> '${devs[*]}'" >&2
        exit 1
    fi
    check_gpus_free "${devs[@]}"
    for dev in "${devs[@]}"; do DEVICE_ARGS+=(--device="$dev"); done
    echo "[miles-ci-job] GPUs $gpus -> ${devs[*]}"
fi

# The cluster's anonymous Docker Hub pulls hit the rate limit; log in when the
# runner provides credentials.
if [[ -n "${DOCKERHUB_USERNAME:-}" && -n "${DOCKERHUB_TOKEN:-}" ]]; then
    printenv DOCKERHUB_TOKEN | docker login -u "$DOCKERHUB_USERNAME" --password-stdin >/dev/null
fi
retry 4 docker pull "$CI_IMAGE"

ENV_ARGS=(-e GITHUB_WORKSPACE=/miles-checkout -e PYTHONPATH=/miles-checkout
          -e GITHUB_STEP_SUMMARY=/miles-checkout/.ci-step-summary.md)
for name in "${CONTAINER_ENV[@]}"; do
    [[ -n "${!name+x}" ]] && ENV_ARGS+=(-e "$name")
done

# The same options as the ARC-era container in _run-ci-rocm.yml, minus
# --privileged, which would expose every GPU on the node to this job.
docker run -dt --name "$CONTAINER" --user root \
    "${DEVICE_ARGS[@]}" \
    --security-opt seccomp=unconfined \
    --group-add video \
    --group-add render \
    --cap-add=SYS_PTRACE \
    --ipc=host \
    --shm-size=32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ulimit nofile=65535:65535 \
    -v "$WS:/miles-checkout" \
    -v "$DATA:/data/miles_ci" \
    -v "$DATA/models:/root/models" \
    -v "$DATA/datasets:/root/datasets" \
    -v "$DATA/hf_cache:/root/.cache/huggingface" \
    -w /miles-checkout \
    "${ENV_ARGS[@]}" \
    "$CI_IMAGE" >/dev/null

docker exec "$CONTAINER" bash "/miles-checkout/$STEPS"
