#!/usr/bin/env bash
# One-command entry to the miles multi-LoRA console. Run FROM YOUR MAC:
#
#   bash ui/run.sh
#
# It will, in order:
#   1. make sure the launcher (ui/launcher.py) is running on the devbox
#   2. open an SSH tunnel  localhost:8067 -> devbox:8067  (kept alive in background)
#   3. launch the miles server if it is down (same as pressing ▶ in the UI)
#   4. open http://127.0.0.1:8067/ui in your browser
#
# Options:
#   bash ui/run.sh --fresh   start from a clean slate: stop the server first,
#                            wipe the usage ledger (usage.jsonl) and the
#                            adapter checkpoints under SAVE_DIR, then relaunch.
#                            ⚠ the billing ledger is gone for good — only for
#                            dev/demo environments.
#
# Overridable:  DEVBOX=miles-test PORT=8067 SLOTS=4 bash ui/run.sh
set -uo pipefail

FRESH=0
for arg in "$@"; do
  case "$arg" in
    --fresh) FRESH=1 ;;
    *) echo "unknown option: $arg (supported: --fresh)" >&2; exit 2 ;;
  esac
done

DEVBOX="${DEVBOX:-miles-test}"
PORT="${PORT:-8067}"
SLOTS="${SLOTS:-4}"
REMOTE_REPO="${REMOTE_REPO:-/personal/e2e_mla/miles}"
SAVE_DIR="${SAVE_DIR:-/personal/demo_v1/save}"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:---dump-details /personal/demo_v1/dump --use-miles-dashboard}"
SSH_ALIAS="${DEVBOX}-sync"
URL="http://127.0.0.1:${PORT}/ui"

say()  { printf '\033[1;36m[run]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[run]\033[0m %s\n' "$*" >&2; exit 1; }

local_status() { curl -s -m 2 "http://127.0.0.1:${PORT}/launcher/status" 2>/dev/null; }

# --- 0. ssh alias (installed by rx; needed for the tunnel and remote cmds) ---
rx devbox ssh-config "$DEVBOX" >/dev/null 2>&1 \
  || say "warning: 'rx devbox ssh-config $DEVBOX' failed — is the devbox running? (rx devbox list)"

# --- 1. SSH tunnel (skip if something already listens on the port) ---
if ! lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  say "opening SSH tunnel localhost:${PORT} -> ${DEVBOX}:${PORT} (stays in background)"
  nohup ssh -o BatchMode=yes -o ServerAliveInterval=30 -N \
    -L "${PORT}:127.0.0.1:${PORT}" "$SSH_ALIAS" >/dev/null 2>&1 &
  disown
  sleep 2
else
  say "port ${PORT} already forwarded/in use locally — reusing it"
fi

# --- 2. launcher on the devbox (start it if the tunnel answers nothing) ---
if [ -z "$(local_status)" ]; then
  say "launcher not answering — starting it on ${DEVBOX}…"
  ssh -o BatchMode=yes "$SSH_ALIAS" \
    "cd ${REMOTE_REPO} && setsid nohup python ui/launcher.py --port ${PORT} \
       --save-dir ${SAVE_DIR} --extra-serve-args '${EXTRA_SERVE_ARGS}' \
       --log-path ${SAVE_DIR%/save}/serve_ui.log \
       > ${SAVE_DIR%/save}/launcher.log 2>&1 < /dev/null & disown" 2>/dev/null
  for _ in $(seq 1 30); do
    [ -n "$(local_status)" ] && break
    sleep 2
  done
  [ -n "$(local_status)" ] || fail "launcher still not reachable at ${URL%/ui} — check: rx devbox list / ssh $SSH_ALIAS"
fi
say "launcher is up"

# --- 2.5 --fresh: stop the server, wipe ledger + checkpoints, then relaunch ---
if [ "$FRESH" = "1" ]; then
  if echo "$(local_status)" | grep -q '"serverUp":true'; then
    say "--fresh: stopping the miles server first (takes up to a couple of minutes)…"
    curl -s -m 10 -X POST "http://127.0.0.1:${PORT}/launcher/stop" >/dev/null
    for _ in $(seq 1 60); do
      echo "$(local_status)" | grep -q '"serverUp":false' && break
      sleep 3
    done
    echo "$(local_status)" | grep -q '"serverUp":false' || fail "--fresh: server did not stop; aborting before touching the ledger"
  fi
  say "--fresh: wiping usage ledger + adapter checkpoints under ${SAVE_DIR}"
  ssh -o BatchMode=yes "$SSH_ALIAS" \
    "rm -f ${SAVE_DIR}/multi_lora_controller/usage.jsonl && rm -rf ${SAVE_DIR}/adapters" \
    || fail "--fresh: wipe failed"
fi

# --- 3. launch the miles server if it is down (same as the ▶ button) ---
if echo "$(local_status)" | grep -q '"serverUp":true'; then
  say "miles server already up"
else
  say "launching miles server (${SLOTS} slots) — startup takes a few minutes; watch the page"
  curl -s -m 5 -X POST "http://127.0.0.1:${PORT}/launcher/start" \
    -H 'Content-Type: application/json' -d "{\"slots\": ${SLOTS}}" >/dev/null
fi

# --- 4. open the console ---
say "opening ${URL}"
if command -v open >/dev/null 2>&1; then open "$URL"; else xdg-open "$URL" >/dev/null 2>&1 || say "open ${URL} manually"; fi

say "done. tunnel keeps running in the background; close it later with:"
say "  pkill -f 'ssh.*-L ${PORT}:127.0.0.1:${PORT}'"
