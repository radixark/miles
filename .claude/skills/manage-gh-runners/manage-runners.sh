#!/usr/bin/env bash
#
# manage-runners.sh — add/remove/list GitHub Actions runner containers on
# a CI host that uses the `gh-runner` image flow (raw docker run, env-var
# driven), NOT the `actions-runner` + docker-compose flow.
#
# See SKILL.md (next to this script) for the operator-facing docs.
#
# What this script encodes:
#   * launch convention — host paths, mount layout, runtime flags, env vars
#     observed on the existing externally-managed hosts; copy these so a new
#     container is bind-compatible with the host's docker-out-of-docker setup.
#   * label convention — name, hw-type, gpu-count meta-labels matching the
#     scheme `_run-ci.yml` and `pr-test.yml` route on.
#   * clean-deregister convention — run `config.sh remove` inside the
#     container BEFORE `docker rm -f`, so GitHub's runner list doesn't
#     accumulate offline zombies.

set -euo pipefail

REPO_URL=${REPO_URL:-}
IMAGE=${IMAGE:-gh-runner:latest}
SETUP_ROOT=/data/runner-setup/data
CACHE_ROOT=/data/runner-cache
HOOK_ROOT=/data/runner-setup/hooks
SKILL_DIR=$(cd "$(dirname "$0")" && pwd)
DEFAULT_PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
  cat >&2 <<EOF
usage:
  $0 list
  $0 add    --name NAME --cvd CVD --hwtype HWTYPE --ngpu N --token TOKEN [--tags TAGS]
  $0 remove --name NAME [--token TOKEN]

flags:
  --name NAME      runner + container name, e.g. <host>-4gpu-0
  --cvd CVD        CUDA_VISIBLE_DEVICES, e.g. 0,1,2,3
  --hwtype TYPE    hardware label, e.g. h200 / h100
  --ngpu N         gpu-count for the \${N}gpu meta-label (must match #devices in --cvd)
  --tags TAGS      comma-separated host tags, also added as runner labels.
                   aliyun: RoCE host; job containers get the host's RDMA view
                   (rdma-rail networks + the hooks/aliyun docker wrapper)
  --token TOKEN    registration token from
                   https://github.com/<org>/<repo>/settings/actions/runners/new
                   (the same token is accepted for config.sh remove)

env vars:
  REPO_URL  required for 'add', e.g. export REPO_URL=https://github.com/<org>/<repo>
  IMAGE     (default: $IMAGE)
EOF
  exit 1
}

parse_flags() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --name)   NAME=$2; shift 2 ;;
      --cvd)    CVD=$2; shift 2 ;;
      --hwtype) HWTYPE=$2; shift 2 ;;
      --ngpu)   NGPU=$2; shift 2 ;;
      --tags)   TAGS=$2; shift 2 ;;
      --token)  TOKEN=$2; shift 2 ;;
      *) die "unknown flag: $1" ;;
    esac
  done
}

cmd_list() {
  local ids
  ids=$(docker ps -q --filter "ancestor=$IMAGE")
  [ -n "$ids" ] || { echo "no $IMAGE containers running"; return 0; }
  printf "%-24s %-12s %-12s %s\n" NAME STATUS CVD LABELS
  for c in $ids; do
    local name status cvd labels
    name=$(docker inspect -f '{{.Name}}' "$c" | tr -d /)
    status=$(docker inspect -f '{{.State.Status}}' "$c")
    cvd=$(docker exec "$c" printenv CUDA_VISIBLE_DEVICES 2>/dev/null || echo -)
    labels=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$c" \
             | awk -F= '/^RUNNER_LABELS=/{ $1=""; sub(/^=/,""); print }')
    printf "%-24s %-12s %-12s %s\n" "$name" "$status" "$cvd" "$labels"
  done
}

# One gateway-less IPv6 ipvlan L2 network per RoCE bond (ipvlan keeps the bond's
# MAC, the only one the fabric forwards), in the bond's /64 above the host's ::1.
# The host's non-connected routes on that bond ride along as a label for the
# aliyun docker hook.
ensure_rdma_networks() {
  local dev net pfx routes
  for dev in $(ls /sys/class/net | grep -E '^bond[0-9]+$'); do
    net=rdma-rail${dev#bond}
    docker network inspect "$net" >/dev/null 2>&1 && continue
    pfx=$(ip -6 -o addr show dev "$dev" scope global | awk '$4 ~ /::1\/64$/ {sub(/::1\/64$/, "", $4); print $4; exit}')
    [ -n "$pfx" ] || die "$dev has no <prefix>::1/64 address"
    routes=$(ip -6 route show dev "$dev" | awk '$2 == "via" {m = ""; for (i = 3; i <= NF; i++) if ($i == "metric") m = " metric " $(i + 1); print $1 " via " $3 m}' | paste -sd, -)
    docker network create -d ipvlan -o ipvlan_mode=l2 -o parent="$dev" --ipv4=false --ipv6 \
      --subnet "$pfx::/64" --ip-range "$pfx:ffff::/80" --label rdma.routes="$routes" "$net" >/dev/null
    echo "created $net on $dev ($pfx::/64, routes: ${routes:-none})"
  done
  docker network ls --filter label=rdma.routes -q | grep -q . || die "no RoCE bonds found for the aliyun tag"
}

cmd_add() {
  NAME=${NAME:-}; CVD=${CVD:-}; HWTYPE=${HWTYPE:-}; NGPU=${NGPU:-}; TAGS=${TAGS:-}; TOKEN=${TOKEN:-}
  [ -n "$NAME" ] && [ -n "$CVD" ] && [ -n "$HWTYPE" ] && [ -n "$NGPU" ] && [ -n "$TOKEN" ] \
    || die "add requires --name --cvd --hwtype --ngpu --token (see --help)"
  [ -n "$REPO_URL" ] || die "REPO_URL not set; export REPO_URL=https://github.com/<org>/<repo>"

  local cvd_count
  cvd_count=$(awk -F, '{print NF}' <<<"$CVD")
  [ "$cvd_count" = "$NGPU" ] || die "--cvd has $cvd_count devices but --ngpu=$NGPU; refusing mismatched label"

  if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    die "container $NAME already exists; remove it first or pick a different --name"
  fi
  [ ! -e "$SETUP_ROOT/$NAME/.runner" ] || die "$SETUP_ROOT/$NAME still holds a registration; run remove first"

  local labels="self-hosted,Linux,X64,gpu-runner,$NAME,$HWTYPE,${NGPU}gpu${TAGS:+,$TAGS}"
  local hook_args=()
  if [[ ",$TAGS," == *",aliyun,"* ]]; then
    ensure_rdma_networks
    install -D -m 0755 "$SKILL_DIR/hooks/aliyun/docker" "$HOOK_ROOT/aliyun/docker"
    hook_args=(-v "$HOOK_ROOT/aliyun:/opt/runner-hooks/aliyun:ro" -e "PATH=/opt/runner-hooks/aliyun:$DEFAULT_PATH")
  fi
  mkdir -p "$SETUP_ROOT/$NAME" "$CACHE_ROOT/$NAME/.cache"

  local runner_env=(
    -e RUNNER_URL="$REPO_URL"
    -e RUNNER_NAME="$NAME"
    -e RUNNER_LABELS="$labels"
    -e RUNNER_DIR="$SETUP_ROOT/$NAME"
  )
  # Register through one-shot containers that read the token from a 0600 file,
  # so the runner container never carries it. The trap set right after `docker
  # create` deletes it on any exit until it starts; a failed start tries to deregister.
  local token_dir
  token_dir=$(mktemp -d)
  trap "rm -rf '$token_dir'" EXIT
  printf '%s' "$TOKEN" > "$token_dir/token"
  chmod 0600 "$token_dir/token"
  docker create --name "$NAME" \
    --restart unless-stopped \
    --runtime=nvidia --gpus all \
    "${runner_env[@]}" \
    -e CUDA_VISIBLE_DEVICES="$CVD" \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$SETUP_ROOT/$NAME:$SETUP_ROOT/$NAME" \
    -v "$CACHE_ROOT/$NAME/.cache:/root/.cache" \
    "${hook_args[@]}" \
    "$IMAGE" >/dev/null
  trap "rm -rf '$token_dir'; docker rm -f '$NAME' >/dev/null 2>&1" EXIT
  local oneshot=(docker run --rm "${runner_env[@]}" -e RUNNER_TOKEN_FILE=/run/runner-token
    -v "$token_dir/token:/run/runner-token:ro" -v "$SETUP_ROOT/$NAME:$SETUP_ROOT/$NAME")
  "${oneshot[@]}" -e RUNNER_REGISTER_ONLY=1 "$IMAGE" \
    || die "registration failed; if the output says RUNNER_TOKEN is required, $IMAGE predates RUNNER_TOKEN_FILE: rebuild it"
  if ! docker start "$NAME" >/dev/null; then
    "${oneshot[@]}" --entrypoint bash "$IMAGE" -c 'cd "$RUNNER_DIR" && ./config.sh remove --token "$(cat "$RUNNER_TOKEN_FILE")"' \
      || echo "WARN: could not deregister $NAME; delete it in the repository's runner settings" >&2
    die "$NAME failed to start; see the docker error above"
  fi
  trap "rm -rf '$token_dir'" EXIT

  if [ ${#hook_args[@]} -gt 0 ]; then
    [ "$(docker exec "$NAME" sh -c 'command -v docker')" = /opt/runner-hooks/aliyun/docker ] \
      || die "$NAME does not resolve docker to the aliyun hook"
  fi

  echo "started $NAME (cvd=$CVD, labels=$labels)"
  echo "tail logs with: docker logs -f $NAME"
}

cmd_remove() {
  NAME=${NAME:-}; TOKEN=${TOKEN:-}
  [ -n "$NAME" ] || die "remove requires --name (see --help)"

  local containers
  containers=$(docker ps -a --format '{{.Names}}') || die "failed to query docker containers"

  if grep -Fxq "$NAME" <<<"$containers"; then
    if [ -n "$TOKEN" ]; then
      if [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" != "true" ]; then
        echo "$NAME is stopped; starting it to deregister from GitHub..."
        docker start "$NAME" >/dev/null
      fi
      echo "deregistering $NAME from GitHub..."
      docker exec "$NAME" bash -c 'cd "$1" && ./config.sh remove --token "$2"' _ "$SETUP_ROOT/$NAME" "$TOKEN" \
        || die "config.sh remove failed; refusing to delete local runner state so deregistration can be retried"
    else
      echo "WARN: no --token given; $NAME will linger as 'offline' in GitHub UI until manually removed"
    fi
  else
    echo "$NAME container not found (skipping in-container deregister)"
  fi

  docker rm -f "$NAME" 2>/dev/null || true
  rm -rf "$SETUP_ROOT/$NAME" "$CACHE_ROOT/$NAME"
  echo "removed container + host dirs for $NAME"
}

case "${1:-}" in
  list)   shift; parse_flags "$@"; cmd_list ;;
  add)    shift; parse_flags "$@"; cmd_add ;;
  remove) shift; parse_flags "$@"; cmd_remove ;;
  -h|--help|"") usage ;;
  *) die "unknown subcommand: $1 (try $0 --help)" ;;
esac
