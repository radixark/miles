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

die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
  cat >&2 <<EOF
usage:
  $0 list
  $0 add    --name NAME --cvd CVD --hwtype HWTYPE --ngpu N --token TOKEN
  $0 remove --name NAME [--token TOKEN]

flags:
  --name NAME      runner + container name, e.g. <host>-4gpu-0
  --cvd CVD        CUDA_VISIBLE_DEVICES, e.g. 0,1,2,3
  --hwtype TYPE    hardware label, e.g. h200 / h100
  --ngpu N         gpu-count for the \${N}gpu meta-label (must match #devices in --cvd)
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

cmd_add() {
  NAME=${NAME:-}; CVD=${CVD:-}; HWTYPE=${HWTYPE:-}; NGPU=${NGPU:-}; TOKEN=${TOKEN:-}
  [ -n "$NAME" ] && [ -n "$CVD" ] && [ -n "$HWTYPE" ] && [ -n "$NGPU" ] && [ -n "$TOKEN" ] \
    || die "add requires --name --cvd --hwtype --ngpu --token (see --help)"
  [ -n "$REPO_URL" ] || die "REPO_URL not set; export REPO_URL=https://github.com/<org>/<repo>"

  local cvd_count
  cvd_count=$(awk -F, '{print NF}' <<<"$CVD")
  [ "$cvd_count" = "$NGPU" ] || die "--cvd has $cvd_count devices but --ngpu=$NGPU; refusing mismatched label"

  if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    die "container $NAME already exists; remove it first or pick a different --name"
  fi

  local labels="self-hosted,Linux,X64,gpu-runner,$NAME,$HWTYPE,${NGPU}gpu"
  mkdir -p "$SETUP_ROOT/$NAME" "$CACHE_ROOT/$NAME/.cache"

  docker run -d --name "$NAME" \
    --restart unless-stopped \
    --runtime=nvidia --gpus all \
    -e RUNNER_URL="$REPO_URL" \
    -e RUNNER_TOKEN="$TOKEN" \
    -e RUNNER_NAME="$NAME" \
    -e RUNNER_LABELS="$labels" \
    -e RUNNER_DIR="$SETUP_ROOT/$NAME" \
    -e CUDA_VISIBLE_DEVICES="$CVD" \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$SETUP_ROOT/$NAME:$SETUP_ROOT/$NAME" \
    -v "$CACHE_ROOT/$NAME/.cache:/root/.cache" \
    "$IMAGE" >/dev/null

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
