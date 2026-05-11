---
title: LoRA Training and Serving
description: Train LoRA adapters with miles SFT or RL recipes and serve them through SGLang from the same checkpoint.
---

# LoRA Training and Serving

Miles supports LoRA adapters for both SFT and RL recipes. Adapters trained by
miles load directly into SGLang for rollout, so there is no separate merge or
conversion step in the training-serving loop.

This page is a stub; the full LoRA tutorial is being written. In the meantime,
the pieces below are enough to get a recipe running.

## Example launchers

The canonical LoRA recipes live under
[`examples/lora/`](https://github.com/radixark/miles/tree/main/examples/lora) in
the miles repo:

- `examples/lora/run-qwen2.5-0.5B-megatron-lora.sh` — small dense, single GPU.
- `examples/lora/run-qwen3-4B-megatron-lora.sh` — Qwen3-4B, RL with LoRA.
- `examples/lora/run-gpt-oss-20B-megatron-moe-lora.sh` — MoE example.

## Key flags

| Flag | Purpose |
|---|---|
| `--lora-rank` | LoRA rank. Typical values: 8, 16, 32, 64. |
| `--lora-alpha` | LoRA alpha. Usually 2 x rank. |
| `--lora-dropout` | Dropout on the LoRA path. Set to `0.0` for RL training. |

## Internals

The bridge between Megatron's LoRA path and SGLang adapter loading is in:

- `miles/backends/megatron_utils/lora_utils.py`
- `miles/backends/megatron_utils/bridge_lora_helpers.py`

A worked tutorial covering checkpoint conversion, SGLang adapter loading, and
LoRA-specific evaluation will land here in a future doc pass.
