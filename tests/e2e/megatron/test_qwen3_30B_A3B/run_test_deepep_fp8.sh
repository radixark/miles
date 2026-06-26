#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"

export GITHUB_WORKSPACE=$PWD
export PYTHONPATH=$PWD

# mori EP MoE: dispatch cap must be >= chunked_prefill_size (auto = 16384 on
# MI355X HBM), otherwise server_args._handle_a2a_moe asserts at startup.
# SGLANG_USE_AITER routes the mori moe core through aiter.fused_moe (the
# per_128x128 path the fix targets). Both reach the SGLang subprocess only via
# _common.py's runtime_env passthrough, so they must be exported here.
export SGLANG_USE_AITER=1
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384
unset SGLANG_DEEPEP_BF16_DISPATCH

# Optional debug instrumentation (default off, not needed to reproduce or to
# apply the fix). When enabled, MoriEPMoE prints per-layer tensor stats and, on a
# zero-output MoE frame, dumps the failing fused_moe inputs to /tmp.
# export MILES_MORI_SELFCHECK=1
# export MILES_MORI_COMBINE_PROBE=1

# Clear any stale Ray cluster left by a previous failed run (port/GPU conflicts).
ray stop --force >/dev/null 2>&1 || true
sleep 2

python tests/e2e/megatron/test_qwen3_30B_A3B/test_deepep_fp8.py
