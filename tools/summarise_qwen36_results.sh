#!/usr/bin/env bash
# Summarise Qwen3.6-35B-A3B RL training logprob diff results.
#
# Two metrics reported per step:
#   abs_diff_mean   = mean(|per-token training logprob - per-token rollout logprob|)
#   mean_abs_diff   = |mean(training logprob) - mean(rollout logprob)|
#
# The user target ("logprobdiff training rollout < 0.02") matches the
# existing Qwen3.5 CI convention which checks mean_abs_diff against
# abs_tol=0.03. abs_diff_mean is inherently larger (triangle inequality).

RESULT_DIR="${RESULT_DIR:?set RESULT_DIR to the directory containing *.log run logs}"

printf "%-20s %-6s %-12s %-12s %-12s %-12s %-12s\n" \
  "CONFIG" "STEPS" "AVG_ABSDIFF" "MAX_ABSDIFF" "AVG_MEANDIFF" "MAX_MEANDIFF" "TARGET<0.02"
printf '%.0s-' {1..95}; echo

for log in "$RESULT_DIR"/*.log; do
  [[ -f "$log" ]] || continue
  name=$(basename "$log" .log)

  # per-step mean of absolute per-token diffs (train_rollout_logprob_abs_diff).
  # Regex covers scientific notation (e/E and ± exponent sign); we keep one
  # value per logged step (no `sort -u` — identical values from different
  # steps are real samples, not duplicates to collapse).
  absdiffs=$(grep -aoE "train/train_rollout_logprob_abs_diff': [0-9.eE+-]+" "$log" |
              awk -F': ' '{print $2}')
  # per-step log_probs / rollout_log_probs means
  mean_diffs=$(python3 - "$log" << 'PY'
import re, sys, pathlib
text = pathlib.Path(sys.argv[1]).read_text(errors="replace")
pairs = {}
for key in ("rollout/log_probs", "rollout/rollout_log_probs"):
    for m in re.finditer(rf"{re.escape(key)}['\"]?[: ]+(-?[0-9.eE+-]+)", text):
        pairs.setdefault(key, []).append(float(m.group(1)))
tr = pairs.get("rollout/log_probs", [])
rl = pairs.get("rollout/rollout_log_probs", [])
for a, b in zip(tr, rl):
    print(f"{abs(a-b):.6f}")
PY
)

  count=$(echo "$absdiffs" | grep -c .)
  if [[ "$count" == "0" ]]; then
    printf "%-20s %-6s %-12s %-12s %-12s %-12s %-12s\n" "$name" 0 "-" "-" "-" "-" "NO_DATA"
    continue
  fi
  abs_avg=$(echo "$absdiffs" | awk '{s+=$1; c++} END {printf "%.4f", s/c}')
  abs_max=$(echo "$absdiffs" | awk '{if($1>m) m=$1} END {printf "%.4f", m}')
  mean_avg=$(echo "$mean_diffs" | awk '{s+=$1; c++} END {if (c>0) printf "%.4f", s/c; else print "-"}')
  mean_max=$(echo "$mean_diffs" | awk '{if($1>m) m=$1} END {if (m>0) printf "%.4f", m; else print "-"}')
  status=$(awk -v m="$mean_avg" 'BEGIN { print (m+0 < 0.02) ? "PASS" : "FAIL" }')
  printf "%-20s %-6s %-12s %-12s %-12s %-12s %-12s\n" \
    "$name" "$count" "$abs_avg" "$abs_max" "$mean_avg" "$mean_max" "$status"
done
