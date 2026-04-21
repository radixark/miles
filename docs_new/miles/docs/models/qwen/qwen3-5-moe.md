---
title: Qwen3.5 MoE
description: Launch recipe for Qwen3.5 35B-A3B with MTP speculative decoding and R3.
---

# Qwen3.5 MoE

Qwen3.5-35B-A3B is Qwen3.5's MoE variant — 3 B active, 35 B total — with an MTP head baked in. Miles's recipe enables R3 (Rollout Routing Replay), unified FP8, and EAGLE-style speculative rollout by default, so the launch script is a single command once weights are converted.

## Variant

| Model | Active / Total | HF ID | Model config |
|---|---|---|---|
| Qwen3.5-35B-A3B | 3 B / 35 B | `Qwen/Qwen3.5-35B-A3B` | `scripts/models/qwen3.5-35B-A3B.sh` |

## Quick start (8× H100)

```bash
cd /root/miles

# 1. Download weights
hf download Qwen/Qwen3.5-35B-A3B --local-dir /root/Qwen3.5-35B-A3B

# 2. Convert HF → Megatron dist checkpoint — note --mtp-num-layers 1
source scripts/models/qwen3.5-35B-A3B.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen3.5-35B-A3B \
   --save          /root/Qwen3.5-35B-A3B_torch_dist \
   --mtp-num-layers 1

# 3. Launch GRPO with MTP speculative rollout
bash scripts/run-qwen3.5-35B-A3B-mtp.sh
```

`--mtp-num-layers 1` is mandatory so the MTP layer survives the HF → Megatron conversion.

## Expected signal

The trainer emits the usual `loss=… reward=…` lines. MoE-specific signals:

- `router/replay_hit_rate` climbs to ≈ 1.0 quickly — R3 is active.
- Speculative decoding logs report a draft acceptance rate from EAGLE; see [Speculative Decoding](../../advanced/speculative-decoding.md) for tuning.

---

## Deep dive

### Launch scripts

| Script | Notes |
|---|---|
| `scripts/run-qwen3.5-35B-A3B-mtp.sh` | Canonical recipe with MTP speculative rollout, R3, unified FP8 |

### Parallelism

| TP | EP | PP | CP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|
| 4 | 8 | 1 | 1 | 4608 | 8 |

### Architecture notes

- Qwen3.5 keeps `A_log` in FP32 — see [Mixed precision](../../advanced/architecture-support.md#mixed-precision-keeping-fp32-parameters-fp32).
- Attention output gate is enabled (`--attention-output-gate`).
- See [Miles Router (R3)](../../advanced/miles-router.md) for the routing-replay rationale.
