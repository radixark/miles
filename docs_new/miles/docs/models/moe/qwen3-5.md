---
title: Qwen3.5 MoE
description: Launch recipe for Qwen3.5 35B-A3B with MTP speculative decoding.
---

# Qwen3.5 MoE

## Variant

| Model | Active / Total | HF ID | Model config |
|---|---|---|---|
| Qwen3.5-35B-A3B | 3 B / 35 B | `Qwen/Qwen3.5-35B-A3B` | `scripts/models/qwen3.5-35B-A3B.sh` |

## Launch scripts

| Script | Notes |
|---|---|
| `scripts/run-qwen3.5-35B-A3B-mtp.sh` | Canonical recipe with MTP speculative rollout |

## Convert weights

```bash
cd /root/miles
source scripts/models/qwen3.5-35B-A3B.sh

PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /data/Qwen3.5-35B-A3B \
   --save          /data/Qwen3.5-35B-A3B_torch_dist \
   --mtp-num-layers 1
```

Pass `--mtp-num-layers 1` so the MTP layer makes it into the `torch_dist` checkpoint.

## Parallelism

| TP | EP | PP | CP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|
| 4 | 8 | 1 | 1 | 4608 | 8 |

## Launch

```bash
bash scripts/run-qwen3.5-35B-A3B-mtp.sh
```

The script turns on R3, unified FP8, and EAGLE speculative rollout by default. See
[Miles Router](../../advanced/miles-router.md) and
[Speculative Decoding](../../advanced/speculative-decoding.md).

## Notes

- Qwen3.5 keeps `A_log` in FP32 — see
  [Mixed precision](../../advanced/architecture-support.md#mixed-precision-keeping-fp32-parameters-fp32).
- Attention output gate is enabled (`--attention-output-gate`).
