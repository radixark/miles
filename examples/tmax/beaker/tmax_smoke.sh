#!/bin/bash
# Beaker entrypoint for the TMax smoke tests: installs Podman inside the task, then runs
# run_tmax_qwen3_5.py prepare + execute. See examples/tmax/README.md ("Beaker").
#
# Runs in docker.io/radixark/miles:latest (CUDA 13) with BEAKER_ALLOW_SUBCONTAINERS=1.
#
# Env:
#   MODE             sync | async | opd | opd-async                     (default: sync)
#   TMAX_CACHE_DIR   shared directory for models, datasets and outputs  (required)
#   SMOKE_ARGS       extra execute flags                                (default: a 3-4 step smoke)
#   DOCKER_PAT, DOCKERHUB_USERNAME   Docker Hub login for the sandbox image pulls
set -euxo pipefail

REPO=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)
CACHE=${TMAX_CACHE_DIR:?set TMAX_CACHE_DIR to a shared directory}
MODE=${MODE:-sync}
SMOKE_ARGS=${SMOKE_ARGS:-"--rollout-batch-size 4 --n-samples-per-prompt 4 --num-rollout 3 --max-seq-len 16384"}

case "$MODE" in
    sync)      MODEL_ARGS="";                                                        EXEC_ARGS="" ;;
    async)     MODEL_ARGS="";                                                        EXEC_ARGS="--async-rollout-gpus 2" ;;
    opd)       MODEL_ARGS="--model-name Qwen3.5-4B-tmax-template --opd-teacher tmax-4b"; EXEC_ARGS="" ;;
    opd-async) MODEL_ARGS="--model-name Qwen3.5-4B-tmax-template --opd-teacher tmax-4b"; EXEC_ARGS="--async-rollout-gpus 2" ;;
    *) echo "unknown MODE=$MODE" >&2; exit 1 ;;
esac

# ---- 1. Podman, configured like open-instruct's Terminal-RL image (host network, no cgroups) ----
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    podman crun conmon netavark passt uidmap fuse-overlayfs golang-github-containers-common
mkdir -p /etc/containers/registries.conf.d /run/podman
cat > /etc/containers/containers.conf <<'EOF'
[containers]
netns="host"
userns="auto:size=1024"
ipcns="host"
utsns="host"
cgroupns="host"
cgroups="disabled"
keyring=false
log_driver = "k8s-file"
volumes = ["/proc:/proc"]
default_sysctls = []
[engine]
cgroup_manager = "cgroupfs"
events_logger="file"
num_locks = 8192
runtime="crun"
EOF
cat > /etc/containers/policy.json <<'EOF'
{"default": [{"type": "insecureAcceptAnything"}],
 "transports": {"docker-daemon": {"": [{"type": "insecureAcceptAnything"}]}}}
EOF
echo 'unqualified-search-registries = ["docker.io"]' > /etc/containers/registries.conf.d/10-unqualified-search-registries.conf
# runroot and graphroot are required once storage.conf exists.
printf '[storage]\ndriver = "overlay"\nrunroot = "/run/containers/storage"\ngraphroot = "/var/lib/containers/storage"\n' \
    > /etc/containers/storage.conf
grep -q "^root:10000:11165536" /etc/subuid 2>/dev/null || echo "root:10000:11165536" >> /etc/subuid
grep -q "^root:10000:11165536" /etc/subgid 2>/dev/null || echo "root:10000:11165536" >> /etc/subgid
ulimit -n 1048576 || true
podman system service --time=0 unix:///run/podman/podman.sock > /tmp/podman-service.log 2>&1 &
for _ in $(seq 1 60); do [ -S /run/podman/podman.sock ] && break; sleep 1; done
if [ -n "${DOCKER_PAT:-}" ]; then
    echo "$DOCKER_PAT" | podman login docker.io -u "$DOCKERHUB_USERNAME" --password-stdin
    export TMAX_REGISTRY_USERNAME=$DOCKERHUB_USERNAME TMAX_REGISTRY_PASSWORD=$DOCKER_PAT
fi
podman run --rm docker.io/library/alpine:latest echo podman-sandbox-ok

# ---- 2. Python deps the example needs ----
pip install docker

# ---- 3. Prepare (download + convert, cached) and train ----
export PYTHONPATH=$REPO
export HF_HOME=$CACHE/hf_home
cd "$REPO"
nvidia-smi
COMMON="--num-gpus-per-node 4 --model-dir $CACHE/models --data-dir $CACHE/datasets --output-dir $CACHE/output \
    --num-prompts 64 $MODEL_ARGS"
python examples/tmax/run_tmax_qwen3_5.py prepare $COMMON
python examples/tmax/run_tmax_qwen3_5.py execute $COMMON \
    --tensor-model-parallel-size 1 --no-optimizer-cpu-offload --sglang-mem-fraction-static 0.6 \
    --docker-host unix:///run/podman/podman.sock --save-interval 1000 --no-filter-zero-std \
    $EXEC_ARGS $SMOKE_ARGS
