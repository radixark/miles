---
title: GLM4 (Dense)
description: Launch recipes for dense GLM4 models (9 B, 32 B — the Zhipu "Z1" reasoning line).
---

# GLM4 (Dense)

GLM4 Dense is the Zhipu "Z1" reasoning line — GLM-Z1-9B-0414 and GLM-Z1-32B-0414. Miles ships a single-node GRPO recipe for the 9 B and a 2-node recipe for the 32 B, both using the same DAPO-Math-17k dataset as the Qwen recipes.

## Variants

| Model | Params | HF ID | Model config |
|---|---|---|---|
| GLM4-9B (GLM-Z1-9B-0414) | 9 B | `zai-org/GLM-Z1-9B-0414` | `scripts/models/glm4-9B.sh` |
| GLM4-32B | 32 B | `zai-org/GLM-Z1-32B-0414` | `scripts/models/glm4-32B.sh` |

## Quick start (GLM4-9B, 8× H100)

```bash
cd /root/miles

# 1. Download weights
hf download zai-org/GLM-Z1-9B-0414 --local-dir /root/GLM-Z1-9B-0414

# 2. Convert HF → Megatron dist checkpoint
source scripts/models/glm4-9B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/GLM-Z1-9B-0414 \
   --save          /root/GLM-Z1-9B-0414_torch_dist

# 3. Launch GRPO
bash scripts/run-glm4-9B.sh
```

## Expected signal

Trainer stdout follows the standard `loss=… reward=…` pattern. GLM-specific things to watch:

- GLM4 needs CP=2 at 16 K context; if you drop to CP=1 the rollout will stall.
- KL divergence on the Z1 line can climb without `--use-tis` — enable it if you see KL creep in the first 100 iterations.

---

## Deep dive

### Launch scripts

| Script | GPUs | Notes |
|---|---|---|
| `scripts/run-glm4-9B.sh` | 8× H100 | Canonical recipe |
| `scripts/run-glm4-9B-4xgpu-radixtree.sh` | 4× H100 | Variant with radix-tree middleware |

### Parallelism

| Size | TP | PP | CP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|
| 9 B | 2 | 1 | 2 | 4608 | 8 |
| 32 B | 4 | 2 | 2 | 4608 | 16 |

### Tuning knobs

| Symptom | Fix |
|---|---|
| Rollout slow at 16 K context | Keep `cp=2`; don't drop to 1 |
| Gradient norm spikes | Drop `--eps-clip-high` to 0.20 |
| KL divergence climbing | Enable `--use-tis` |
