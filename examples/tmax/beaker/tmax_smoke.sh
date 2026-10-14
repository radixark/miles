#!/bin/bash
# Beaker entrypoint for the TMax runs: installs Podman inside the task, then runs
# run_tmax_qwen3_5.py prepare + execute. See examples/tmax/README.md ("Beaker").
#
# Runs in docker.io/radixark/miles:latest (CUDA 13) with BEAKER_ALLOW_SUBCONTAINERS=1.
#
# Env:
#   MODE             sync | async | opd | opd-async (4-GPU smoke tests), or
#                    prod-1node (8 GPUs: the open-instruct production recipe on one node)  (default: sync)
#   TMAX_CACHE_DIR   shared directory for models, datasets and outputs                       (required)
#   EXP_NAME         prod-1node checkpoint directory; a restarted job with the same name resumes
#   SANDBOX_SHARDS   podman daemons, each with its own storage (default: 1; prod-1node: 4)
#   MIRROR_URL       docker.io pull-through mirror(s), as in open-instruct (optional)
#   EXTRA_ARGS       extra execute flags, appended last
#   DOCKER_PAT, DOCKERHUB_USERNAME   Docker Hub login for the sandbox image pulls
set -euxo pipefail

REPO=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)
CACHE=${TMAX_CACHE_DIR:?set TMAX_CACHE_DIR to a shared directory}
MODE=${MODE:-sync}
SMOKE="--num-prompts 64 --rollout-batch-size 4 --n-samples-per-prompt 4 --num-rollout 3 --max-seq-len 16384 \
    --tensor-model-parallel-size 1 --no-optimizer-cpu-offload --sglang-mem-fraction-static 0.6 \
    --save-interval 1000 --no-filter-zero-std"
OPD_MODEL="--model-name Qwen3.5-4B-tmax-template --opd-teacher tmax-4b"

case "$MODE" in
    sync)      NUM_GPUS=4; MODEL_ARGS="";          EXEC_ARGS="$SMOKE" ;;
    async)     NUM_GPUS=4; MODEL_ARGS="";          EXEC_ARGS="$SMOKE --async-rollout-gpus 2" ;;
    opd)       NUM_GPUS=4; MODEL_ARGS="$OPD_MODEL"; EXEC_ARGS="$SMOKE" ;;
    opd-async) NUM_GPUS=4; MODEL_ARGS="$OPD_MODEL"; EXEC_ARGS="$SMOKE --async-rollout-gpus 2" ;;
    prod-1node)
        # open-instruct qwen35_9b_dppo_repro_4node_64k_holmes_cu13.sh on one node: 8 prompts x 32 samples,
        # 64k per episode / 16k per turn, fully async (staleness <= 4), active sampling, 1:1 train:rollout GPUs.
        NUM_GPUS=8
        SANDBOX_SHARDS=${SANDBOX_SHARDS:-4}
        export TMAX_START_CONCURRENCY=${TMAX_START_CONCURRENCY:-64} TMAX_EXEC_CONCURRENCY=${TMAX_EXEC_CONCURRENCY:-256}
        MODEL_ARGS="--model-name Qwen3.5-9B-tmax-template"
        EXEC_ARGS="--exp-name ${EXP_NAME:?set EXP_NAME for prod-1node} --num-rollout 500 --save-interval 5 \
            --rollout-batch-size 8 --n-samples-per-prompt 32 --max-seq-len 67584 --rollout-max-response-len 16384 \
            --async-rollout-gpus 4 --max-weight-staleness 4 \
            --tensor-model-parallel-size 2 --context-parallel-size 2 --max-tokens-per-gpu 34816 \
            --no-optimizer-cpu-offload --sglang-mem-fraction-static 0.8 \
            --extra-args '--log-probs-backend fused'" ;;
    *) echo "unknown MODE=$MODE" >&2; exit 1 ;;
esac
SANDBOX_SHARDS=${SANDBOX_SHARDS:-1}

# ---- 1. Podman, brought up exactly as open-instruct's Terminal-RL jobs do (beaker/podman/, vendored) ----
# The Miles image has no podman: install it from apt (4.9; open-instruct's image builds 5.6.2 from source),
# then run the vendored config + docker_login.sh: SANDBOX_SHARDS daemons with their own storage, the
# docker.io mirror when MIRROR_URL is set, the image janitor, and Docker Hub credentials.
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    podman crun conmon netavark passt uidmap fuse-overlayfs golang-github-containers-common
PODMAN_DIR="$REPO/examples/tmax/beaker/podman"
mkdir -p /etc/containers/registries.conf.d
cp "$PODMAN_DIR/containers.conf" /etc/containers/containers.conf
cp "$PODMAN_DIR/policy.json" /etc/containers/policy.json
cp "$PODMAN_DIR/10-unqualified-search-registries.conf" /etc/containers/registries.conf.d/
install -m 755 "$PODMAN_DIR/setup_dockerio_mirror" /usr/local/bin/setup_dockerio_mirror
# podman 4.9 rejects a storage.conf without runroot/graphroot; setup_dockerio_mirror only patches the driver.
printf '[storage]\ndriver = "overlay"\nrunroot = "/run/containers/storage"\ngraphroot = "/var/lib/containers/storage"\n' \
    > /etc/containers/storage.conf
grep -q "^root:10000:11165536" /etc/subuid 2>/dev/null || echo "root:10000:11165536" >> /etc/subuid
grep -q "^root:10000:11165536" /etc/subgid 2>/dev/null || echo "root:10000:11165536" >> /etc/subgid
export SWERL_PODMAN_SERVICE_COUNT=$SANDBOX_SHARDS PODMAN_NUM_LOCKS=${PODMAN_NUM_LOCKS:-65536}
export SWERL_PODMAN_IMAGE_JANITOR_ENABLED=${SWERL_PODMAN_IMAGE_JANITOR_ENABLED:-1}
export SWERL_PODMAN_IMAGE_JANITOR_INTERVAL_S=${SWERL_PODMAN_IMAGE_JANITOR_INTERVAL_S:-60}
export SWERL_PODMAN_IMAGE_JANITOR_UNTIL=${SWERL_PODMAN_IMAGE_JANITOR_UNTIL:-10m}
set +u  # docker_login.sh reads unset variables
source "$PODMAN_DIR/docker_login.sh"
set -u
export TMAX_DOCKER_HOST="$SWERL_PODMAN_DOCKER_HOSTS"
if [ -n "${DOCKER_PAT:-}" ]; then
    echo "$DOCKER_PAT" | podman login docker.io -u "$DOCKERHUB_USERNAME" --password-stdin
    export TMAX_REGISTRY_USERNAME=$DOCKERHUB_USERNAME TMAX_REGISTRY_PASSWORD=$DOCKER_PAT
fi
podman --url "${TMAX_DOCKER_HOST%%,*}" run --rm docker.io/library/alpine:latest echo podman-sandbox-ok

# ---- 2. Python deps the example needs ----
pip install docker

# ---- 3. Prepare (download + convert, cached) and train ----
export PYTHONPATH=$REPO
export HF_HOME=$CACHE/hf_home
cd "$REPO"
nvidia-smi
COMMON="--num-gpus-per-node $NUM_GPUS --model-dir $CACHE/models --data-dir $CACHE/datasets --output-dir $CACHE/output \
    $MODEL_ARGS"
# EXEC_ARGS may carry a quoted --extra-args value, hence eval.
eval python examples/tmax/run_tmax_qwen3_5.py prepare $COMMON $(echo "$EXEC_ARGS" | grep -o -- "--num-prompts [0-9]*" || true)
eval python examples/tmax/run_tmax_qwen3_5.py execute $COMMON "$EXEC_ARGS" "${EXTRA_ARGS:-}"
