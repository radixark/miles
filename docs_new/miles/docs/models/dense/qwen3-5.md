---
title: Qwen3.5 (Dense)
description: Launch recipes for dense Qwen3.5 models.
---

# Qwen3.5 (Dense)

Qwen3.5 introduces an attention-output gate and tighter rotary configuration. Miles
handles these via the Qwen3.5-specific spec (`miles_plugins.models.qwen3_5`).

## Variants

| Model | Params | HF ID | Model config |
|---|---|---|---|
| Qwen3.5-4B | 4 B | `Qwen/Qwen3.5-4B` | `scripts/models/qwen3.5-4B.sh` |
| Qwen3.5-9B | 9 B | `Qwen/Qwen3.5-9B` | `scripts/models/qwen3.5-9B.sh` |
| Qwen3.5-27B | 27 B | `Qwen/Qwen3.5-27B` | `scripts/models/qwen3.5-27B.sh` |

!!! note "Architecture specifics"
    - Attention output gate enabled (`--attention-output-gate`).
    - Rotary base `10_000_000`, `--rotary-percent 0.25`.
    - Vocab size 248 320.
    - The `A_log` parameter must stay in FP32 — see
      [Backends Beyond Megatron: fp32 parameter handling](../../advanced/architecture-support.md#mixed-precision-keeping-fp32-parameters-fp32).

## Launch scripts

| Script | GPUs | Notes |
|---|---|---|
| `scripts/run-qwen3.5-4B.sh` | 8× H100 | Canonical |
| `scripts/run-qwen3.5-9B.sh` | 8× H100 | 9B variant |
| `scripts/run-qwen3.5-27B.sh` | 16× H100 | 27B multi-node-friendly |

## Convert weights

```bash
cd /root/miles
source scripts/models/qwen3.5-4B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /data/Qwen3.5-4B \
   --save          /data/Qwen3.5-4B_torch_dist
```

## Parallelism

| Size | TP | PP | CP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|
| 4 B | 2 | 1 | 1 | 8192 | 8 |
| 9 B | 2 | 1 | 2 | 4608 | 8 |
| 27 B | 4 | 2 | 2 | 4608 | 16 |

## Launch

```bash
bash scripts/run-qwen3.5-4B.sh
```

## Tuning

| Symptom | Fix |
|---|---|
| Precision drift on `A_log` | Confirm `mark_param_dtype(..., torch.float32)` hits the Qwen3.5 bridge |
| Tokenisation misalignment | Re-run the [chat template verifier](../../user-guide/agentic-chat-template.md) |
| Train OOM at 27 B | Bump PP to 2, reduce `max-tokens-per-gpu` |
