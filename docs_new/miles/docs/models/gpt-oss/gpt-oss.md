---
title: GPT-OSS 20B
description: Launch recipe for GPT-OSS 20B on NVIDIA — Megatron BF16 and FSDP paths.
---

# GPT-OSS 20B

GPT-OSS 20B is a dense 20 B model with two ready-to-run Miles recipes: a Megatron BF16 path and an FSDP path. The FSDP variant skips weight conversion entirely — useful when you're iterating on the trainer and don't want to re-run the HF → Megatron converter.

## Variant

| Model | Params | HF ID | Model config |
|---|---|---|---|
| GPT-OSS-20B | 20 B | `gpt-oss/gpt-oss-20b` | `scripts/models/gpt-oss-20b.sh` |

## Quick start (8× H100)

=== "Megatron BF16"

    ```bash
    cd /root/miles

    # 1. Download weights
    hf download gpt-oss/gpt-oss-20b --local-dir /root/gpt-oss-20b

    # 2. Convert HF → Megatron torch_dist
    source scripts/models/gpt-oss-20b.sh
    PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
       ${MODEL_ARGS[@]} \
       --hf-checkpoint /root/gpt-oss-20b \
       --save          /root/gpt-oss-20b_torch_dist

    # 3. Launch
    bash scripts/run-gpt-oss-20b-bf16.sh
    ```

=== "FSDP (no conversion)"

    ```bash
    cd /root/miles
    hf download gpt-oss/gpt-oss-20b --local-dir /root/gpt-oss-20b
    bash scripts/run-gptoss-20b-fsdp.sh
    ```

## Expected signal

Standard `loss=… reward=…` trainer stdout. The Megatron path runs through CP=2 to fit 20 B at `max_tokens_per_gpu=4608`; FSDP is simpler but trades some throughput.

---

## Deep dive

### Launch scripts

| Script | Backend |
|---|---|
| `scripts/run-gpt-oss-20b-bf16.sh` | Megatron, BF16 |
| `scripts/run-gptoss-20b-fsdp.sh` | FSDP (no weight conversion) |

### Parallelism (Megatron path)

| TP | PP | CP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|
| 4 | 1 | 2 | 4608 | 8 |

For FSDP's flag mapping, see [Training backends: FSDP](../../user-guide/usage.md#fsdp-pytorch-fsdp2).
