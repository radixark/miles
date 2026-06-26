#!/bin/bash
set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

RAY_ADDRESS="http://127.0.0.1:8265"
for i in $(seq 1 60); do
    if curl -fsS --max-time 2 "${RAY_ADDRESS}/api/version" >/dev/null 2>&1; then
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "Ray dashboard at ${RAY_ADDRESS} not ready after 60s" >&2
        exit 1
    fi
    echo "Waiting for Ray dashboard at ${RAY_ADDRESS} to be ready..."
    sleep 1
done

ray job submit --address="${RAY_ADDRESS}" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1"
     }
   }' \
   -- python3 examples/multi_lora/run_schedule.py \
   --multi-lora-dir "${SCRIPT_DIR}/adapters"
