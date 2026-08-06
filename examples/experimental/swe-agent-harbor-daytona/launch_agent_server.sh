#!/bin/bash
# Start the Harbor agent server in Daytona mode.
#
# Run this from the root of a harbor-framework/harbor checkout on the
# harbor-miles-v0.20.0 branch, which carries the Miles integration, before
# launching examples/swe-agent-harbor-docker/run.py. Trials are graded inside Daytona cloud
# sandboxes, so this host needs outbound HTTPS but no Docker daemon.
set -euo pipefail

: "${DAYTONA_API_KEY:?set DAYTONA_API_KEY to a Daytona API key}"
: "${HARBOR_TASKS_DIR:?set HARBOR_TASKS_DIR to the directory holding Harbor task dirs}"

TRIALS_DIR="${TRIALS_DIR:-/tmp/harbor_trials}"
PORT="${PORT:-11000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-32}"

export HARBOR_ENV_TYPE=daytona
export HARBOR_DAYTONA_DISK_GB="${HARBOR_DAYTONA_DISK_GB:-10}"
# Snapshot each task image on first use so later trials skip the build.
export HARBOR_DAYTONA_AUTO_SNAPSHOT=1

# terminus-2 runs as a host process and calls the model itself, so the model
# endpoint must be reachable from here rather than from inside the sandbox.
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:30000/v1}"
export OPENAI_BASE_URL="$OPENAI_API_BASE"

# Keep these consistent with --rollout-max-response-len and --max-seq-len on the
# trainer side; see the README section on sizing them.
export AGENT_MAX_INPUT_TOKENS="${AGENT_MAX_INPUT_TOKENS:-32768}"
export AGENT_MAX_OUTPUT_TOKENS="${AGENT_MAX_OUTPUT_TOKENS:-8192}"
export HARBOR_RESPONSE_LENGTH_POLICY=abort

mkdir -p "$TRIALS_DIR"

exec python miles_agent_server.py \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-concurrent "$MAX_CONCURRENT" \
    --agent-timeout 5400 \
    --trials-dir "$TRIALS_DIR"
