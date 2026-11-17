#!/usr/bin/env bash
# Devbox: derive validation-only variants of the GLM-5.2 NVFP4 W4A16 e2e (outside the repo).
#   gpu-adam: GPU TE FusedAdam, no CPU offload, no precision-aware optimizer (distributed-optimizer hook)
#   rn:       the registered recipe with the flag set to 0 (round-to-nearest baseline)
set -euo pipefail
src=/root/miles/tests/e2e/megatron/test_glm5_2_744b_a40b_5layer_nvfp4_w4a16.py
out=/hai-workspace/bf16-sr/e2e
mkdir -p "$out"
sed -e '/"--optimizer-cpu-offload "/d' -e '/"--overlap-cpu-optimizer-d2h-h2d "/d' -e '/"--use-precision-aware-optimizer "/d' \
  "$src" > "$out/test_glm5_2_5layer_nvfp4_w4a16_sr_gpu_adam.py"
sed -e 's/"OPEN_TRAINING_BF16_STOCHASTIC_ROUNDING_FLAG": "1"/"OPEN_TRAINING_BF16_STOCHASTIC_ROUNDING_FLAG": "0"/' \
  "$src" > "$out/test_glm5_2_5layer_nvfp4_w4a16_rn.py"
for f in "$out"/*.py; do echo "== $f"; diff "$src" "$f" || true; done
