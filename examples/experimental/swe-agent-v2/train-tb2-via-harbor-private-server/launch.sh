#!/usr/bin/env bash
# Launch a 2-node GLM-4.7-Flash agentic-async training run that uses the
# miles_agent_server from the harbor-private branch
# shi/rebase-on-upstream-v0.7.0 as the rollout backend.
#
# See README.md for prerequisites (running agent server, populated
# /root/swe_train.jsonl, the threadpool-fix training branch).
#
# Usage:
#   bash launch.sh <run-tag>            # e.g. bash launch.sh pr-smoke
#
# <run-tag> is threaded through --save-dir, --save-traces-dir, and
# --wandb-run-name so multiple attempts don't collide.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <run-tag>" >&2
    exit 64
fi
RUN_TAG="$1"

# Resolve where the launcher script lives. We expect to be invoked from
# the miles repo root (cwd-sensitive: run-glm47-flash-agentic-async.py
# uses relative paths to other tooling and the prompt-data symlink).
LAUNCHER="examples/experimental/swe-agent-v2/run-glm47-flash-agentic-async.py"
if [ ! -f "$LAUNCHER" ]; then
    echo "error: $LAUNCHER not found. Run this script from the miles repo root." >&2
    exit 66
fi

# The rcli job name is used to construct the Tailscale-ingress FQDN that
# the rollout router advertises back to the external agent server.
# Default matches what we use; override via $RCLI_JOB_NAME.
RCLI_JOB_NAME="${RCLI_JOB_NAME:-shidong-flash-pr-train}"

# Dataset symlink check (the launcher hard-codes /root/swe_train.jsonl).
if [ ! -e /root/swe_train.jsonl ]; then
    echo "error: /root/swe_train.jsonl missing. See README.md step 'Launch'." >&2
    exit 65
fi

# Make sure the rcli job's namespace can route to the external agent server
# via the in-cluster ts-egress-aws-agent-server ExternalName.
AGENT_SERVER_URL="${AGENT_SERVER_URL:-http://ts-egress-aws-agent-server:8080}"

python "$LAUNCHER" \
    --num-nodes 2 --train-num-nodes 1 --skip-prepare \
    --max-seq-len 65536 \
    --save-dir "/workspace/GLM-4.7-Flash_2node_tb2_${RUN_TAG}/" \
    --save-traces-dir "/workspace/flash-2node-traces-${RUN_TAG}/traces" \
    --rollout-batch-size 4 --n-samples-per-prompt 8 --global-batch-size 32 \
    --save-interval 5 \
    --agent-server-url "$AGENT_SERVER_URL" \
    --router-external-host "${RCLI_JOB_NAME}-ts-ingress.tail134ba0.ts.net" \
    --wandb-project glm47-flash-agentic-async \
    --wandb-team ch271828n-team \
    --wandb-run-name "$RUN_TAG" \
    2>&1 | tee "/tmp/flash-2node-launch-${RUN_TAG}.log"
