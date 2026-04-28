---
title: Qwen3.5 (Dense)
description: Launch recipes for dense Qwen3.5 models with gated attention and FP32 A_log.
---

# Qwen3.5 (Dense)

Qwen3.5 adds an attention-output gate, a tighter rotary configuration, and keeps `A_log` in FP32. Miles handles all three via the `miles_plugins.models.qwen3_5` spec — no code changes required beyond the launch script.

## Variants

| Model | Params | HF ID | Model config |
|---|---|---|---|
| Qwen3.5-4B | 4 B | `Qwen/Qwen3.5-4B` | `scripts/models/qwen3.5-4B.sh` |
| Qwen3.5-9B | 9 B | `Qwen/Qwen3.5-9B` | `scripts/models/qwen3.5-9B.sh` |
| Qwen3.5-27B | 27 B | `Qwen/Qwen3.5-27B` | `scripts/models/qwen3.5-27B.sh` |

## Quick start (Qwen3.5-4B, 8× H100)

```bash
cd /root/miles

# 1. Download weights
hf download Qwen/Qwen3.5-4B --local-dir /root/Qwen3.5-4B

# 2. Convert HF → Megatron dist checkpoint
source scripts/models/qwen3.5-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen3.5-4B \
   --save          /root/Qwen3.5-4B_torch_dist

# 3. Launch GRPO
bash scripts/run-qwen3.5-4B.sh
```

## Expected signal

Same `loss=… reward=… rollout=… train=…` trainer stdout as the other dense recipes. Specific to Qwen3.5: if you see precision drift or NaNs in the early iterations, check that `A_log` is staying in FP32 through `mark_param_dtype(..., torch.float32)` in the Qwen3.5 bridge — see the tuning knob below.

---

## Deep dive

### Architecture specifics

- Attention output gate enabled (`--attention-output-gate`).
- Rotary base `10_000_000`, `--rotary-percent 0.25`.
- Vocab size 248 320.
- The `A_log` parameter must stay in FP32 — see [Backends Beyond Megatron: fp32 parameter handling](../../advanced/architecture-support.md#mixed-precision-keeping-fp32-parameters-fp32).

### Launch scripts

| Script | GPUs | Notes |
|---|---|---|
| `scripts/run-qwen3.5-4B.sh` | 8× H100 | Canonical |
| `scripts/run-qwen3.5-9B.sh` | 8× H100 | 9 B variant |
| `scripts/run-qwen3.5-27B.sh` | 16× H100 | 27 B multi-node-friendly |

### Parallelism

| Size | TP | PP | CP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|
| 4 B | 2 | 1 | 1 | 8192 | 8 |
| 9 B | 2 | 1 | 2 | 4608 | 8 |
| 27 B | 4 | 2 | 2 | 4608 | 16 |

### Tuning knobs

| Symptom | Fix |
|---|---|
| Precision drift on `A_log` | Confirm `mark_param_dtype(..., torch.float32)` hits the Qwen3.5 bridge |
| Tokenisation misalignment | Re-run the [chat template verifier](../../user-guide/agentic-chat-template.md) |
| Train OOM at 27 B | Bump PP to 2, reduce `max-tokens-per-gpu` |
