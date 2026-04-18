---
title: GLM4.7 Flash
description: Launch recipe for GLM4.7 Flash — a compact MoE with 64 routed experts.
---

# GLM4.7 Flash

GLM4.7 Flash is a compact MoE: 64 routed experts, top-4 activation, one shared expert.

## Architecture

| Property | Value |
|---|---|
| Routed experts | 64 |
| Active experts per token | 4 |
| Shared experts | 1 |
| MoE layers | 46 (1 dense head) |
| Hidden size | 2048 |
| Heads | 20 |

Full config lives in `scripts/models/glm4.7-flash.sh`.

## Launch scripts

| Script | Entry point |
|---|---|
| `scripts/run-glm4.7-flash.sh` | Bash launcher |
| `scripts/run_glm47_flash.py` | Python launcher |

## Prerequisites

GLM4.7 Flash requires a current `transformers` with the corresponding model entry:

```bash
pip install git+https://github.com/huggingface/transformers.git@76732b4e7120808ff989edbd16401f61fa6a0afa
```

## Convert weights

```bash
cd /root/miles
source scripts/models/glm4.7-flash.sh

PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /data/GLM-4.7-Flash \
   --save          /data/GLM-4.7-Flash_torch_dist
```

## Parallelism

| TP | EP | PP | CP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|
| 4 | 8 | 1 | 2 | 8192 | 8 |

## Launch

```bash
bash scripts/run-glm4.7-flash.sh
```

R3 + unified FP8 are recommended — see [Miles Router](../../advanced/miles-router.md)
and [FP8](../../advanced/fp8-low-precision.md).
