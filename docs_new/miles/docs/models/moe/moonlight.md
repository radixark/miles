---
title: Moonlight
description: Launch recipe for Moonlight 16B-A3B.
---

# Moonlight 16B-A3B

Moonlight is a 16 B-parameter MoE with 3 B activation per token — a compact target for
MoE RL experimentation on a single node.

## Variant

| Model | Active / Total | HF ID | Model config |
|---|---|---|---|
| Moonlight-16B-A3B | 3 B / 16 B | `moonshotai/Moonlight-16B-A3B` | `scripts/models/moonlight.sh` |

## Launch scripts

| Script | GPUs |
|---|---|
| `scripts/run-moonlight-16B-A3B.sh` | 8× H100 |

## Convert weights

```bash
cd /root/miles
source scripts/models/moonlight.sh

PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /data/Moonlight-16B-A3B \
   --save          /data/Moonlight-16B-A3B_torch_dist
```

## Parallelism

| TP | EP | PP | CP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|
| 4 | 8 | 1 | 1 | 4608 | 8 |

On a single node, add CPU Adam if needed:

```bash
OPTIMIZER_ARGS+=(
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)
```

## Launch

```bash
bash scripts/run-moonlight-16B-A3B.sh
```

## Related

Enable R3 for stability — see [Miles Router](../../advanced/miles-router.md).
