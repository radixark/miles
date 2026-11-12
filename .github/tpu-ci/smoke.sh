#!/usr/bin/env bash
# TPU CI pod entrypoint (render_jobset.py): run one scripts/tpu launcher's recipe on the pod's v7x 2x2x1 host.
# Env: MILES_SHA, LAUNCHER (run_<LAUNCHER>_torchtitan.py), RECIPE. Exits with train's code, or 1 if sglang dies first.
set -uo pipefail

# core files of crashed TPU processes run to tens of GB
ulimit -c 0
export PYTHONUNBUFFERED=1
# keep caches off the container's root overlay: filling it gets the pod evicted
export HF_HOME=/data/hf XDG_CACHE_HOME=/tmp/cache TORCH_TPU_TIER3_ROOT=/dev/shm/torch_tpu_tier3

cd /root/miles || exit 1
if ! git fetch -q --depth 1 origin "$MILES_SHA" || ! git checkout -q FETCH_HEAD; then
  echo "FAIL: cannot check out miles ${MILES_SHA}"
  exit 1
fi
# the image installed miles' dependencies; only the source changes here
pip install -q --no-deps -e . || { echo "FAIL: pip install miles"; exit 1; }
echo "== miles $(git rev-parse HEAD) launcher=${LAUNCHER} recipe=${RECIPE} =="

# the launchers import `scripts.tpu`, which only the repo root on the path provides
export PYTHONPATH="/root/miles${PYTHONPATH:+:$PYTHONPATH}"
script="scripts/tpu/run_${LAUNCHER}_torchtitan.py"
args=(--recipe "$RECIPE" --model-dir /data/models --data-dir /data/datasets)

# the engine serves the checkpoint, so it must be on disk before sglang starts
python3 "$script" prepare "${args[@]}" || { echo "FAIL: ${script} prepare"; exit 1; }

python3 "$script" sglang "${args[@]}" > >(sed -u 's/^/[sglang] /') 2>&1 &
engine=$!
# the engine compiles only after the trainer joins, so train starts without waiting for /health
sleep 60
python3 "$script" train "${args[@]}" &
trainer=$!

while kill -0 "$trainer" 2>/dev/null; do
  if ! kill -0 "$engine" 2>/dev/null; then
    wait "$engine"
    echo "FAIL: sglang exited (rc=$?) before training finished"
    exit 1
  fi
  sleep 10
done
wait "$trainer"
rc=$?
echo "== train exited rc=${rc} =="
# the pod's exit ends the engine; SIGTERM first so it can release the slice cleanly
kill -TERM "$engine" 2>/dev/null || true
exit "$rc"
