---
title: Qwen3-Next
description: Launch recipe for Qwen3-Next 80B-A3B using the HuggingFace-wrapped backend.
---

# Qwen3-Next 80B-A3B

Qwen3-Next introduces a Gated-Delta-Net attention variant. Miles runs it through the
HuggingFace-wrapped Megatron backend — see
[Backends Beyond Megatron](../../advanced/architecture-support.md) for how that works.

## Variant

| Model | Active / Total | HF ID | Model config |
|---|---|---|---|
| Qwen3-Next-80B-A3B | 3 B / 80 B | `Qwen/Qwen3-Next-80B-A3B` | `scripts/models/qwen3-next-80B-A3B.sh` |

## Launch scripts

| Script | GPUs | Notes |
|---|---|---|
| `scripts/run-qwen3-next-80B-A3B.sh` | 16× H100 | Canonical, Megatron backend |
| `scripts/run-qwen3-next-80B-A3B-8gpus.sh` | 8× H100 | Tight-fit variant |
| `scripts/run-qwen3-next-80B-A3B-fsdp.sh` | 16× H100 | FSDP backend |

## Prerequisites

The scripts require two env vars:

```bash
export BASE_FOLDER=/shared/checkpoints        # reachable from all nodes
export MASTER_ADDR=<head node IP>
```

## Convert weights

```bash
cd /root/miles
source scripts/models/qwen3-next-80B-A3B.sh

PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint $BASE_FOLDER/Qwen3-Next-80B-A3B \
   --save          $BASE_FOLDER/Qwen3-Next-80B-A3B_torch_dist
```

## Parallelism

| TP | EP | PP | CP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|
| 4 | 8 | 2 | 1 | 4608 | 16 |

With the HF-wrapped backend, **TP inside the Attention module** isn't supported (see the
[limitations](../../advanced/architecture-support.md#current-limitations)). If you need
TP there, fall back to a native Megatron implementation.

## Launch

```bash
bash scripts/run-qwen3-next-80B-A3B.sh
```

## FSDP variant

FSDP skips the weight conversion step:

```bash
bash scripts/run-qwen3-next-80B-A3B-fsdp.sh
```
