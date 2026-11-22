#!/bin/bash
# Entrypoint for the gh-runner image. Contract matches manage-runners.sh:
#   RUNNER_DIR   host-mounted, identity-bind, may be empty on first start
#   RUNNER_URL / RUNNER_TOKEN / RUNNER_NAME / RUNNER_LABELS  registration inputs
#   RUNNER_TOKEN_FILE      token read from a file when RUNNER_TOKEN is unset
#   RUNNER_REGISTER_ONLY   exit once registered (manage-runners.sh registers in a
#                          one-shot container, so the runner never holds the token)
# Populates RUNNER_DIR from the baked-in distribution if needed, then
# configures (once) and runs. .runner persists in the host volume.
set -e
export RUNNER_ALLOW_RUNASROOT=1

RUNNER_DIR="${RUNNER_DIR:-/opt/actions-runner/runner}"
mkdir -p "$RUNNER_DIR"

# First start on a fresh per-name volume: seed the runner binaries.
if [ ! -x "$RUNNER_DIR/config.sh" ]; then
    echo "Seeding runner dir $RUNNER_DIR from /opt/actions-runner-dist ..."
    cp -a /opt/actions-runner-dist/. "$RUNNER_DIR/"
fi
cd "$RUNNER_DIR"

# Already configured (survived a restart): just run.
if [ -f .runner ]; then
    [ -z "${RUNNER_REGISTER_ONLY:-}" ] || exit 0
    echo "Runner already configured, starting..."
    exec ./run.sh
fi

if [ -z "${RUNNER_TOKEN:-}" ] && [ -n "${RUNNER_TOKEN_FILE:-}" ]; then
    RUNNER_TOKEN=$(cat "$RUNNER_TOKEN_FILE")
fi

if [ -z "${RUNNER_TOKEN:-}" ]; then
    echo "ERROR: RUNNER_TOKEN is required for first-time configuration" >&2
    exit 1
fi

RUNNER_URL="${RUNNER_URL:-https://github.com/radixark/miles}"
RUNNER_NAME="${RUNNER_NAME:-$(hostname)}"
RUNNER_LABELS="${RUNNER_LABELS:-self-hosted,Linux,X64}"

echo "Configuring runner: ${RUNNER_NAME}"
echo "  URL:    ${RUNNER_URL}"
echo "  Labels: ${RUNNER_LABELS}"
echo "  Dir:    ${RUNNER_DIR}"
./config.sh \
    --url "${RUNNER_URL}" \
    --token "${RUNNER_TOKEN}" \
    --name "${RUNNER_NAME}" \
    --labels "${RUNNER_LABELS}" \
    --work "_work" \
    --unattended \
    --replace

[ -z "${RUNNER_REGISTER_ONLY:-}" ] || exit 0
echo "Runner configured, starting..."
exec ./run.sh
