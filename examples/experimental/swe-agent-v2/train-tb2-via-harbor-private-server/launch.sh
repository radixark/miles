#!/usr/bin/env bash
# Launch a 2-node GLM-4.7-Flash agentic-async training run that uses the
# miles_agent_server from the harbor-private branch
# shi/rebase-on-upstream-v0.7.0 as the rollout backend.
#
# See README.md for prerequisites (running agent server, populated
# /root/swe_train.jsonl, a current miles checkout).
#
# Usage:
#   bash launch.sh <run-tag>            # e.g. bash launch.sh pr-smoke
#
# <run-tag> is threaded through --save-dir, --save-traces-dir, and
# --wandb-run-name so multiple attempts don't collide.
#
# Optional env overrides (sensible defaults shown):
#   AGENT_SERVER_URL=http://agent-server:8080
#       Base URL of the running miles_agent_server.
#   ROUTER_EXTERNAL_HOST=
#       Hostname/FQDN that the trainer's session-server advertises back
#       to the agent server, so the agent server can call into the
#       trainer for tokenization. Required when the agent server cannot
#       reach the trainer at the trainer's default service name (e.g. an
#       off-cluster agent server with an explicit ingress/proxy hostname).
#       Leave unset to omit --router-external-host.
#   WANDB_TEAM=
#       Wandb team to log to. Leave unset to omit --wandb-team.
#   OUTPUT_ROOT=/workspace
#       Filesystem root under which save-dir and save-traces-dir are
#       created. Must point at writable storage shared between the
#       training nodes; the cluster's persistent volume is usually right.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <run-tag>" >&2
    exit 64
fi
RUN_TAG="$1"

LAUNCHER="examples/experimental/swe-agent-v2/run-glm47-flash-agentic-async.py"
if [ ! -f "$LAUNCHER" ]; then
    echo "error: $LAUNCHER not found. Run this script from the miles repo root." >&2
    exit 66
fi

if [ ! -e /root/swe_train.jsonl ]; then
    echo "error: /root/swe_train.jsonl missing. See README.md step 'Launch'." >&2
    exit 65
fi

AGENT_SERVER_URL="${AGENT_SERVER_URL:-http://agent-server:8080}"
ROUTER_EXTERNAL_HOST="${ROUTER_EXTERNAL_HOST:-}"
WANDB_TEAM="${WANDB_TEAM:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace}"

ARGS=(
    --num-nodes 2 --train-num-nodes 1 --skip-prepare
    --max-seq-len 65536
    --save-dir "${OUTPUT_ROOT}/GLM-4.7-Flash_2node_tb2_${RUN_TAG}/"
    --save-traces-dir "${OUTPUT_ROOT}/flash-2node-traces-${RUN_TAG}/traces"
    --rollout-batch-size 4 --n-samples-per-prompt 8 --global-batch-size 32
    --save-interval 5
    --agent-server-url "$AGENT_SERVER_URL"
    --wandb-project glm47-flash-agentic-async
    --wandb-run-name "$RUN_TAG"
)
[ -n "$ROUTER_EXTERNAL_HOST" ] && ARGS+=(--router-external-host "$ROUTER_EXTERNAL_HOST")
[ -n "$WANDB_TEAM" ] && ARGS+=(--wandb-team "$WANDB_TEAM")

python "$LAUNCHER" "${ARGS[@]}" \
    2>&1 | tee "/tmp/flash-2node-launch-${RUN_TAG}.log"
