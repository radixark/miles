#!/usr/bin/env bash
# Devbox: run e2e cases back to back in the background. Each case is LABEL=TEST_FILE.
#   tools/queue_e2e.sh sr-final=tests/e2e/megatron/test_glm5_2_744b_a40b_5layer_nvfp4_w4a16.py ...
# Per case: logs/e2e-LABEL.log, logs/e2e-LABEL.exit; queue progress in logs/queue.log.
set -euo pipefail
cases=("$@")
setsid nohup bash -c '
  for case in "$@"; do
    label=${case%%=*}; test_file=${case#*=}
    log=/hai-workspace/bf16-sr/logs/e2e-$label.log
    echo "$(date -u +%FT%TZ) start $label $test_file" >> /hai-workspace/bf16-sr/logs/queue.log
    ray stop --force >/dev/null 2>&1 || true
    (cd /root/miles && export PYTHONPATH=/root/miles:/root/Megatron-LM && date -u && git -C /root/miles rev-parse HEAD \
      && git -C /root/Megatron-LM rev-parse HEAD && start=$(date +%s) && { python "$test_file"; code=$?; } ; \
      echo "elapsed_s=$(($(date +%s) - start))"; echo $code > /hai-workspace/bf16-sr/logs/e2e-$label.exit; date -u) > "$log" 2>&1 < /dev/null
    echo "$(date -u +%FT%TZ) end $label exit=$(cat /hai-workspace/bf16-sr/logs/e2e-$label.exit)" >> /hai-workspace/bf16-sr/logs/queue.log
  done
' queue "${cases[@]}" > /dev/null 2>&1 < /dev/null &
echo "queued ${#cases[@]} cases"
