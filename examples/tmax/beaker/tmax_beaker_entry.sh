#!/bin/bash
# Beaker entrypoint for a TMax experiment config (submitted by beaker/submit.py, which pins the commit):
#   bash examples/tmax/beaker/tmax_beaker_entry.sh examples/tmax/configs/<experiment>.yaml
# Runs in docker.io/radixark/miles with BEAKER_ALLOW_SUBCONTAINERS=1: brings podman up the way open-instruct's
# Terminal-RL jobs do, prunes old training checkpoints, then runs the launcher's prepare + execute on the config.
#
# On NUM_NODES > 1 replicas, as open-instruct's multi-node jobs: every replica brings podman up; replica 0 runs
# the rest and the Ray head; the others join its Ray cluster as GPU workers (configs/beaker_configs/
# ray_node_setup.sh) and exit when it is gone. The agent processes run on the head, which reaches each worker's
# podman shards through TCP forwards that accept the head's address only; tmax_sandbox caps the live sandboxes
# per node (sandboxes_per_node), as open-instruct's --pool_size actors spread over its nodes.
#
# Env (set by submit.py from the config's beaker: section):
#   NUM_NODES                 Beaker replicas, one Ray node each                           (default: 1)
#   SANDBOX_SHARDS            podman daemons, each with its own storage                    (default: 1)
#   MIRROR_URL                docker.io pull-through mirror(s), as in open-instruct        (optional)
#   KEEP_LAST_N_CHECKPOINTS   keep only the newest N training checkpoints, 0 keeps all     (default: 0)
#   EXTRA_ARGS                launcher flags that override the config
#   DOCKER_PAT, DOCKERHUB_USERNAME   Docker Hub login for the sandbox image pulls
set -euxo pipefail

REPO=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)
CONFIG="${1:?usage: tmax_beaker_entry.sh <config.yaml>}"
SANDBOX_SHARDS=${SANDBOX_SHARDS:-1}
NUM_NODES=${NUM_NODES:-1}
RANK=${BEAKER_REPLICA_RANK:-0}
RAY_PORT=6379
HEAD_DONE=/tmp/miles-tmax-head-done  # the head marks every node once training succeeded
PODMAN_TCP_PORT=23750  # worker replicas forward <port + shard> to their podman shard sockets
cd "$REPO"
git log -1 --format='running miles @ %H (%s)' || true
nvidia-smi
pip install docker  # the sandbox client; the Miles image does not ship it
if [ "$NUM_NODES" -gt 1 ]; then
    HEAD_IP=$(getent hosts "$BEAKER_LEADER_REPLICA_HOSTNAME" | awk '{print $1; exit}')
    export MASTER_ADDR=$HEAD_IP
fi

# ---- 1. Podman, brought up exactly as open-instruct's Terminal-RL jobs do (beaker/podman/, vendored) ----
# The Miles image has no podman: install it from apt (4.9; open-instruct's image builds 5.6.2 from source),
# then run the vendored config + docker_login.sh: SANDBOX_SHARDS daemons with their own storage, the
# docker.io mirror when MIRROR_URL is set, the image janitor, and Docker Hub credentials.
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    podman crun conmon netavark passt uidmap fuse-overlayfs golang-github-containers-common socat
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
fi
podman --url "${TMAX_DOCKER_HOST%%,*}" run --rm docker.io/library/alpine:latest echo podman-sandbox-ok

# ---- 2. Ray worker replicas: serve podman to the head, join its Ray cluster and stay until it goes away ----
if [ "$NUM_NODES" -gt 1 ] && [ "$RANK" != 0 ]; then
    # The Docker API is root on this node, so only the head's address may connect.
    shard=0
    for host in ${SWERL_PODMAN_DOCKER_HOSTS//,/ }; do
        socat TCP-LISTEN:$((PODMAN_TCP_PORT + shard)),fork,reuseaddr,range="$HEAD_IP/32" "UNIX-CONNECT:${host#unix://}" &
        shard=$((shard + 1))
    done
    # The head starts after prepare, which converts the checkpoint on a first run, hence the long wait.
    deadline=$((SECONDS + 7200))
    until ray status --address="$HEAD_IP:$RAY_PORT" >/dev/null 2>&1; do
        [ $SECONDS -lt $deadline ] || { echo "Ray head $HEAD_IP:$RAY_PORT never came up"; exit 1; }
        sleep 15
    done
    ray start --address="$HEAD_IP:$RAY_PORT" --num-gpus "$(nvidia-smi -L | wc -l)" --disable-usage-stats
    # As open-instruct's ray_node_setup.sh, a worker that loses the head fails the job, so propagateFailure stops
    # a run that would otherwise hang on the missing node; unlike it, a worker leaves with 0 after the head marked
    # the run successful, so a finished experiment is not failed and retried.
    trap 'ray stop --force >/dev/null 2>&1 || true; exit 1' TERM INT HUP EXIT
    set +x
    misses=0
    while sleep 5; do
        if timeout 10 ray status --address="$HEAD_IP:$RAY_PORT" >/dev/null 2>&1; then
            misses=0
        elif [ $((misses += 1)) -ge 6 ]; then
            if [ -f "$HEAD_DONE" ]; then
                echo "Ray head $HEAD_IP:$RAY_PORT finished training; leaving"
                trap - TERM INT HUP EXIT
                ray stop --force >/dev/null 2>&1 || true
                exit 0
            fi
            echo "Ray head $HEAD_IP:$RAY_PORT unreachable 6 times in a row; leaving"
            exit 1
        fi
    done
fi

# ---- 3. Keep the newest KEEP_LAST_N_CHECKPOINTS training checkpoints (open-instruct --keep_last_n_checkpoints) ----
read -r CKPT_DIR MODEL_DIR < <(python - "$CONFIG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
ckpt = f"{c['output_dir']}/{c['exp_name']}/checkpoints" if c.get("exp_name") and c.get("output_dir") else "-"
print(ckpt, c.get("model_dir", "/root/models"))
PY
)
[ "$CKPT_DIR" = "-" ] && CKPT_DIR=""
if [ "${KEEP_LAST_N_CHECKPOINTS:-0}" -gt 0 ] && [ -n "$CKPT_DIR" ]; then
    (
        while sleep 600; do
            python - "$CKPT_DIR" "$KEEP_LAST_N_CHECKPOINTS" <<'PY' || true
import re, shutil, sys
from pathlib import Path
ckpt_dir, keep = Path(sys.argv[1]), int(sys.argv[2])
tracker = ckpt_dir / "latest_checkpointed_iteration.txt"
if tracker.exists():
    latest = tracker.read_text().strip()
    iters = sorted((d for d in ckpt_dir.glob("iter_*") if re.fullmatch(r"iter_\d+", d.name)), key=lambda d: int(d.name[5:]))
    protected = f"iter_{int(latest):07d}" if latest.isdigit() else None  # the checkpoint a resume loads
    for old in iters[:-keep]:
        if old.name != protected:
            print(f"pruning old checkpoint {old}", flush=True)
            shutil.rmtree(old, ignore_errors=True)
PY
        done
    ) &
fi

# ---- 4. Prepare (download + convert, cached), start the Ray cluster when multi-node, and train ----
export PYTHONPATH=$REPO
export HF_HOME=${HF_HOME:-$(dirname "$MODEL_DIR")/hf_home}  # next to the models, so downloads are cached
# EXTRA_ARGS is a shell-quoted flag list, hence eval.
eval python examples/tmax/run_tmax_qwen3_5.py prepare "$CONFIG" "${EXTRA_ARGS:-}"
if [ "$NUM_NODES" -gt 1 ]; then
    GPUS=$(nvidia-smi -L | wc -l)
    PYTHONUNBUFFERED=1 ray start --head --node-ip-address "$HEAD_IP" --port $RAY_PORT --num-gpus "$GPUS" \
        --disable-usage-stats
    WORKER_IPS=$(python - "$((NUM_NODES * GPUS))" <<'PY'
import sys, time, ray
ray.init(address="auto", logging_level="ERROR")
want, deadline = int(sys.argv[1]), time.time() + 1800
while (have := int(ray.cluster_resources().get("GPU", 0))) < want:
    assert time.time() < deadline, f"only {have}/{want} GPUs joined the Ray cluster in 30 min"
    time.sleep(10)
# By node id, not address: replicas smaller than a machine can share one, each with its own podman.
head = ray.get_runtime_context().get_node_id()
workers = sorted(n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"] and n["NodeID"] != head)
print(f"Ray cluster has {have} GPUs; workers {workers}", file=sys.stderr, flush=True)
print(" ".join(workers))
PY
)
    # The head's own shards, then every worker's forwarded shards.
    for ip in $WORKER_IPS; do
        for shard in $(seq 0 $((SANDBOX_SHARDS - 1))); do
            TMAX_DOCKER_HOST+=",tcp://$ip:$((PODMAN_TCP_PORT + shard))"
        done
    done
    export MILES_SCRIPT_EXTERNAL_RAY=1 MILES_SCRIPT_NUM_NODES=$NUM_NODES
fi
eval python examples/tmax/run_tmax_qwen3_5.py execute "$CONFIG" "${EXTRA_ARGS:-}"
if [ "$NUM_NODES" -gt 1 ]; then
    python - "$HEAD_DONE" <<'PY'
import sys, ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
ray.init(address="auto", logging_level="ERROR")
mark = ray.remote(num_cpus=0)(lambda path: open(path, "w").close())
nodes = [n["NodeID"] for n in ray.nodes() if n["Alive"]]
ray.get([mark.options(scheduling_strategy=NodeAffinitySchedulingStrategy(n, soft=False)).remote(sys.argv[1]) for n in nodes])
print(f"marked {len(nodes)} nodes done", flush=True)
PY
fi
