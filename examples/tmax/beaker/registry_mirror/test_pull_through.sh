#!/bin/bash
# Pull-through and fallback test for the TMax registry mirrors, on the sandboxes' own path: apt podman 4.9 with
# open-instruct's vendored config, registries.conf from setup_dockerio_mirror, and a docker-py pull carrying the
# Docker Hub login as tmax_sandbox sends it. Run by `registry_mirror.py test`:
#   bash test_pull_through.sh <mirror,mirror,...> <docker.io image>
# Per case, prints which endpoint served the pull and how long it took:
#   each mirror alone; a dead mirror before a live one; a dead mirror alone (docker.io fallback); no mirror.
# Then Docker Hub's rate limit for the login and for anonymous pulls.
set -euo pipefail

MIRRORS="${1:?mirror list}"
IMAGE="${2:?image}"
DEAD="neptune-cs-aus-262.reviz.ai2.in:38048"  # the retired mirror gateway: refuses connections
HERE=$(cd "$(dirname "$0")" && pwd)
PODMAN_DIR="$HERE/../podman"

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    podman crun conmon netavark passt uidmap fuse-overlayfs golang-github-containers-common curl >/dev/null
cp "$PODMAN_DIR/containers.conf" /etc/containers/containers.conf
cp "$PODMAN_DIR/policy.json" /etc/containers/policy.json
printf '[storage]\ndriver = "overlay"\nrunroot = "/run/containers/storage"\ngraphroot = "/var/lib/containers/storage"\n' \
    > /etc/containers/storage.conf
install -m 755 "$PODMAN_DIR/setup_dockerio_mirror" /usr/local/bin/setup_dockerio_mirror
pip install -q docker
podman --version

run_case() {
    local label="$1" mirror_url="$2"
    rm -f /etc/containers/registries.conf  # an empty MIRROR_URL leaves the previous case's file in place
    MIRROR_URL="$mirror_url" /usr/local/bin/setup_dockerio_mirror >/dev/null
    podman system service --time=0 unix:///tmp/p.sock --log-level=debug > /tmp/svc.log 2>&1 &
    local svc=$!
    sleep 2
    python - "$IMAGE" "$label" <<'PY'
import os, sys, time, docker
image, label = sys.argv[1], sys.argv[2]
client = docker.DockerClient(base_url="unix:///tmp/p.sock", timeout=900)
try:
    client.images.remove(image, force=True)
except docker.errors.NotFound:  # podman 4.9 answers a missing image with a plain 404
    pass
auth = {"username": os.environ["DOCKERHUB_USERNAME"], "password": os.environ["DOCKER_PAT"]}
start = time.monotonic()
try:
    client.images.pull(image, auth_config=auth)
    print(f"[{label}] pull OK in {time.monotonic() - start:.1f}s", flush=True)
except Exception as e:
    print(f"[{label}] pull FAILED after {time.monotonic() - start:.1f}s: {str(e)[:200]}", flush=True)
PY
    kill "$svc"; wait "$svc" 2>/dev/null || true
    grep -o -E 'Trying to access \\"[^\\]+|Accessing \\"[^\\]+\\" failed: [^"]{0,120}|Returning credentials[^"]*' /tmp/svc.log \
        | sed 's/\\"//g; s/^/    /' | head -8 || true
}

IFS=, read -r -a mirrors <<< "$MIRRORS"
for m in "${mirrors[@]}"; do run_case "mirror $m" "$m"; done
run_case "dead mirror, then ${mirrors[0]}" "$DEAD,${mirrors[0]}"
run_case "dead mirror only (docker.io fallback)" "$DEAD"
run_case "no mirror (docker.io)" ""

echo "Docker Hub rate limit (ratelimit-limit / -remaining; absent = not limited):"
probe_limit() {
    local label="$1" token
    token=$(curl -s "${@:2}" "https://auth.docker.io/token?service=registry.docker.io&scope=repository:ratelimitpreview/test:pull" \
        | python -c "import json,sys; print(json.load(sys.stdin)['token'])") || { echo "    $label: no token"; return; }
    curl -s --head -H "Authorization: Bearer $token" https://registry-1.docker.io/v2/ratelimitpreview/test/manifests/latest \
        | grep -i -E "^ratelimit" | tr -d '\r' | sed "s/^/    $label: /" || echo "    $label: no ratelimit headers"
}
probe_limit "$DOCKERHUB_USERNAME" -u "$DOCKERHUB_USERNAME:$DOCKER_PAT"
probe_limit anonymous
