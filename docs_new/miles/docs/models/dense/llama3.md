---
title: Llama 3 (Dense)
description: Launch recipes for Llama 3 models.
---

# Llama 3

Miles supports Llama 3.1 and Llama 3.2 on both NVIDIA and AMD.

## Variants

| Model | Params | HF ID | Model config |
|---|---|---|---|
| Llama-3.2-3B-Instruct | 3 B | `meta-llama/Llama-3.2-3B-Instruct` | `scripts/models/llama3.2-3B-Instruct.sh` |
| Llama-3.1-8B-Instruct | 8 B | `meta-llama/Llama-3.1-8B-Instruct` | `scripts/models/llama3.1-8B-Instruct.sh` |

## Launch scripts

| Script | Platform |
|---|---|
| `scripts/run-llama3.2-3B-Instruct-amd.sh` | AMD MI300X |

Miles on NVIDIA uses the Qwen3 / GLM4 script template — copy `run-qwen3-4B.sh` and swap
the `source` line to your Llama config.

## Convert weights

```bash
cd /root/miles
source scripts/models/llama3.2-3B-Instruct.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /data/Llama-3.2-3B-Instruct \
   --save          /data/Llama-3.2-3B-Instruct_torch_dist
```

## Parallelism

| Size | TP | PP | CP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|
| 3 B | 2 | 1 | 1 | 12 288 | 8 |
| 8 B | 2 | 1 | 1 | 8192 | 8 |

## AMD

Use the ROCm image and the `llama3.2-3B-Instruct-amd.sh` config. See
[Platforms: AMD MI300X](../../platforms/amd.md) for the container setup.
