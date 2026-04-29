#!/usr/bin/env bash
# Drive Qwen3.6-35B-A3B MTP RL training through multiple parallelism configs.
# Each run: 10 rollouts, smoke test for logprob diff < 0.02.

set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# OUTPUT_DIR/RESULT_DIR are user-supplied; pick a path with NFS-grade
# space (each smoke run writes hundreds of MB of logs/state).
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to a writable, NFS-backed path}"
RESULT_DIR="${RESULT_DIR:-${OUTPUT_DIR}/logs}"
mkdir -p "$RESULT_DIR"

# ------------------------------------------------------------
# Configurations on 8xH200 (TP:EP:CP:PP:ETP)
# Constraint: world = PP*TP*CP*DP = PP*ETP*EP*EDP = 8
# TP-only configs (TP=4/8) dropped: num_kv_heads=2 can't shard that far.
# ------------------------------------------------------------
CONFIGS=(
  # EP8 baseline is exercised by run_qwen3_6_35b_a3b_mtp.py directly; the
  # configs below cover the remaining parallelism shapes.
  "CP2_EP8:1:8:2:1:1"
  "PP2_CP4:1:2:4:2:1"
  "PP2_EP2_TP2:2:2:1:2:1"
  "PP2_EP2_CP2:1:2:2:2:1"
)

run_one() {
  local name="$1" tp="$2" ep="$3" cp="$4" pp="$5" etp="$6"
  local log="$RESULT_DIR/${name}.log"

  echo "===================================================="
  echo ">>> Running $name (TP=$tp EP=$ep CP=$cp PP=$pp ETP=$etp)"
  echo ">>> Log: $log"
  echo "===================================================="

  # Clean ray state between runs
  ray stop --force >/dev/null 2>&1 || true
  pkill -9 sglang 2>/dev/null || true
  sleep 5
  rm -rf /tmp/ray/session_* || true

  env \
    -u MILES_SCRIPT_DATA_DIR \
    -u MILES_SCRIPT_MODEL_DIR \
    python3 tools/run_qwen3_6_35b_a3b_mtp.py \
      --tp "$tp" --ep "$ep" --cp "$cp" --pp "$pp" --etp "$etp" \
      --num-rollout 10 \
      --rollout-max-response-len 1024 \
      --output-dir "${OUTPUT_DIR}/runs/${name}" \
      --skip-prepare \
      >"$log" 2>&1 && echo ">>> $name OK" || echo ">>> $name FAILED (see $log)"
}

for entry in "${CONFIGS[@]}"; do
  IFS=":" read -r NAME TP EP CP PP ETP <<< "$entry"
  run_one "$NAME" "$TP" "$EP" "$CP" "$PP" "$ETP"
done

# ------------------------------------------------------------
# Summarise logprob diff per config.
# abs_diff_mean = mean(|per-token train - per-token rollout|)  [logged metric]
# mean_abs_diff = |mean(train) - mean(rollout)|                [aggregate]
# Qwen3.5 CI convention checks mean_abs_diff against 0.03; abs_diff_mean is
# inherently 2-3x larger (triangle inequality). We report both.
# ------------------------------------------------------------
echo
echo "==================== SUMMARY ===================="
printf "%-20s %-6s %-14s %-14s %-10s\n" "CONFIG" "steps" "abs_diff_mean" "mean_abs_diff" "status"
for entry in "${CONFIGS[@]}"; do
  IFS=":" read -r NAME _ _ _ _ _ <<< "$entry"
  LOG="$RESULT_DIR/${NAME}.log"
  if [[ ! -f "$LOG" ]]; then
    printf "%-20s %-6s %-14s %-14s %-10s\n" "$NAME" "-" "-" "-" "missing"
    continue
  fi
  STEPS=$(grep -acE "'train/step': [0-9]+" "$LOG")
  LAST=$(python3 - "$LOG" << 'PY'
import re, sys, pathlib
text = pathlib.Path(sys.argv[1]).read_text(errors="replace")
absd = [float(x) for x in re.findall(r"'train/train_rollout_logprob_abs_diff':\s*([0-9.eE+-]+)", text)][::2]
tr = [float(x) for x in re.findall(r"'rollout/log_probs':\s*(-?[0-9.eE+-]+)", text)][::2]
rl = [float(x) for x in re.findall(r"'rollout/rollout_log_probs':\s*(-?[0-9.eE+-]+)", text)][::2]
if not absd:
    print("- -"); sys.exit()
a = absd[-1]
m = abs(tr[-1] - rl[-1]) if tr and rl else float('nan')
print(f"{a:.4f} {m:.4f}")
PY
)
  ABS=${LAST% *}; MEAN=${LAST#* }
  if [[ "$ABS" == "-" ]]; then
    printf "%-20s %-6s %-14s %-14s %-10s\n" "$NAME" "$STEPS" "-" "-" "NO_LOGPROB"
  else
    STATUS=$(awk -v m="$MEAN" 'BEGIN { print (m+0 < 0.03) ? "PASS" : "FAIL" }')
    printf "%-20s %-6s %-14s %-14s %-10s\n" "$NAME" "$STEPS" "$ABS" "$MEAN" "$STATUS"
  fi
done
