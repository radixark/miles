---
title: GPT-OSS 20B
description: Launch recipe for GPT-OSS 20B on NVIDIA GPUs.
---

# GPT-OSS 20B

## Variant

| Model | Params | HF ID | Model config |
|---|---|---|---|
| GPT-OSS-20B | 20 B | `gpt-oss/gpt-oss-20b` | `scripts/models/gpt-oss-20b.sh` |

## Launch scripts

| Script | Backend |
|---|---|
| `scripts/run-gpt-oss-20b-bf16.sh` | Megatron, BF16 |
| `scripts/run-gptoss-20b-fsdp.sh` | FSDP (no weight conversion) |

## Convert weights (Megatron path)

```bash
cd /root/miles
source scripts/models/gpt-oss-20b.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /data/gpt-oss-20b \
   --save          /data/gpt-oss-20b_torch_dist
```

## Parallelism

| TP | PP | CP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|
| 4 | 1 | 2 | 4608 | 8 |

## Launch

=== "Megatron BF16"

    ```bash
    bash scripts/run-gpt-oss-20b-bf16.sh
    ```

=== "FSDP"

    ```bash
    bash scripts/run-gptoss-20b-fsdp.sh
    ```

For FSDP's flag mapping, see [Training backends: FSDP](../../user-guide/usage.md#fsdp-pytorch-fsdp2).
