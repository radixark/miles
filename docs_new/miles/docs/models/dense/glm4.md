---
title: GLM4 (Dense)
description: Launch recipes for dense GLM4 models.
---

# GLM4 (Dense)

Miles supports dense GLM4 sizes from 9 B to 32 B.

## Variants

| Model | Params | HF ID | Model config |
|---|---|---|---|
| GLM4-9B (GLM-Z1-9B-0414) | 9 B | `zai-org/GLM-Z1-9B-0414` | `scripts/models/glm4-9B.sh` |
| GLM4-32B | 32 B | `zai-org/GLM-Z1-32B-0414` | `scripts/models/glm4-32B.sh` |

## Launch scripts

| Script | GPUs | Notes |
|---|---|---|
| `scripts/run-glm4-9B.sh` | 8× H100 | Canonical recipe |
| `scripts/run-glm4-9B-4xgpu-radixtree.sh` | 4× H100 | Variant with radix-tree middleware |

## Convert weights

```bash
cd /root/miles
source scripts/models/glm4-9B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /data/GLM-Z1-9B-0414 \
   --save          /data/GLM-Z1-9B-0414_torch_dist
```

## Parallelism

| Size | TP | PP | CP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|
| 9 B | 2 | 1 | 2 | 4608 | 8 |
| 32 B | 4 | 2 | 2 | 4608 | 16 |

## Launch

```bash
bash scripts/run-glm4-9B.sh
```

## Tuning

| Symptom | Fix |
|---|---|
| Rollout slow at 16 K context | Keep `cp=2`; don't drop to 1 |
| Gradient norm spikes | Drop `--eps-clip-high` to 0.20 |
| KL divergence climbing | Enable `--use-tis` |
