---
title: Supported Models
description: Per-family recipes covering weight conversion, launch flags, and parallelism choices.
---

# Supported Models

Miles ships ready-to-run recipes for every model family listed below. Each page covers
weight conversion, parallelism, and the launch script in the order you'd actually run
them.

## By family

Each model name links to its recipe page.

| Family | Models |
|---|---|
| **DeepSeek** | [DeepSeek-V4 Pro](deepseek/deepseek-v4-pro.md)<br/>[DeepSeek-V4 Flash](deepseek/deepseek-v4-flash.md)<br/>[DeepSeek-R1](deepseek/deepseek.md)<br/>[DeepSeek-V3](deepseek/deepseek.md) |
| **Qwen** | [Qwen3.6 MoE](qwen/qwen3-6-moe.md)<br/>[Qwen3.6](qwen/qwen3-6.md)<br/>[Qwen3.5-35B-A3B](qwen/qwen3-5-moe.md)<br/>[Qwen3.5-4B / 9B / 27B](qwen/qwen3-5.md)<br/>[Qwen3-Next-80B-A3B-Thinking](qwen/qwen3-next.md)<br/>[Qwen3-30B-A3B / 235B-A22B](qwen/qwen3-moe.md)<br/>[Qwen3-0.6B / 1.7B / 4B / 8B / 14B / 32B](qwen/qwen3.md) |
| **GLM** | [GLM-5.1](glm/glm5.md)<br/>[GLM-5](glm/glm5.md)<br/>[GLM-4.7-Flash](glm/glm4-7-flash.md)<br/>[GLM-4.5](glm/glm4-5.md)<br/>[GLM-Z1-9B-0414](glm/glm4.md) |
| **Kimi** | [Kimi-K2.6](kimi/kimi-k2.5.md)<br/>[Kimi-K2.5](kimi/kimi-k2.5.md)<br/>[Kimi-K2-Instruct / Thinking](kimi/kimi-k2.md)<br/>[Moonlight-16B-A3B](kimi/moonlight.md) |
| **Nemotron** | [Nemotron-3-Super-120B-A12B-FP8](nemotron/nemotron-3-super.md)<br/>[Nemotron-3-Nano MoE](nemotron/nemotron-3-nano-moe.md)<br/>[Nemotron-3-Nano](nemotron/nemotron-3-nano.md) |
| **MiMo** | [MiMo-7B-RL](mimo/mimo.md) |
| **GPT-OSS** | [gpt-oss-20b](gpt-oss/gpt-oss.md) |

## How a recipe is structured

Every recipe page follows the same six sections:

1. **Model Introduction** — what the model is and why miles supports it.
2. **Supported Variants** — model sizes + HF links.
3. **Environment Setup** — env vars, downloads, and HF → Megatron conversion.
4. **Launch** — the `scripts/run-<family>.sh` (or `run_<family>.py`) invocation.
5. **Recipe Configuration** — parallelism, algorithm, rollout/SGLang, optimizer.
6. **Pairs Well With** — links to the advanced features that complement this recipe.

## Adding a new model

Miles's plugin architecture lets you wrap a HuggingFace implementation as a Megatron
module without patching Megatron core. See
[Backends Beyond Megatron](../advanced/architecture-support.md) for the workflow.
