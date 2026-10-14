#!/bin/bash
# Beaker entrypoint for a TMax experiment config (submitted by beaker/submit.py, which pins the commit):
#   bash examples/tmax/beaker/tmax_beaker_entry.sh examples/tmax/configs/<experiment>.yaml
# Runs in docker.io/radixark/miles with BEAKER_ALLOW_SUBCONTAINERS=1: brings podman up the way open-instruct's
# Terminal-RL jobs do, prunes old training checkpoints, then runs the launcher's prepare + execute on the config.
#
# Env (set by submit.py from the config's beaker: section):
#   SANDBOX_SHARDS            podman daemons, each with its own storage                    (default: 1)
#   MIRROR_URL                docker.io pull-through mirror(s), as in open-instruct        (optional)
#   KEEP_LAST_N_CHECKPOINTS   keep only the newest N training checkpoints, 0 keeps all     (default: 0)
#   EXTRA_ARGS                launcher flags that override the config
#   DOCKER_PAT, DOCKERHUB_USERNAME   Docker Hub login for the sandbox image pulls
set -euxo pipefail

REPO=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)
CONFIG="${1:?usage: tmax_beaker_entry.sh <config.yaml>}"
SANDBOX_SHARDS=${SANDBOX_SHARDS:-1}
cd "$REPO"
git log -1 --format='running miles @ %H (%s)' || true

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
fi
podman --url "${TMAX_DOCKER_HOST%%,*}" run --rm docker.io/library/alpine:latest echo podman-sandbox-ok

# ---- 2. Python deps the example needs ----
pip install docker

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

# ---- 4. Prepare (download + convert, cached) and train ----
export PYTHONPATH=$REPO
export HF_HOME=${HF_HOME:-$(dirname "$MODEL_DIR")/hf_home}  # next to the models, so downloads are cached
nvidia-smi
# EXTRA_ARGS is a shell-quoted flag list, hence eval.
eval python examples/tmax/run_tmax_qwen3_5.py prepare "$CONFIG" "${EXTRA_ARGS:-}"
eval python examples/tmax/run_tmax_qwen3_5.py execute "$CONFIG" "${EXTRA_ARGS:-}"
