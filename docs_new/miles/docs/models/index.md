---
title: Supported Models
description: Per-family recipes covering weight conversion, launch flags, and parallelism choices.
---

# Supported Models

Miles ships ready-to-run recipes for every model family listed below. Each page covers
weight conversion, parallelism, and the launch script in the order you'd actually run
them.

## Dense

| Model family | Sizes | Recipe |
|---|---|---|
| **Qwen3** | 0.6 B, 1.7 B, 4 B, 8 B, 14 B, 32 B | [qwen3](dense/qwen3.md) |
| **Qwen3.5** | 4 B, 9 B, 27 B | [qwen3-5](dense/qwen3-5.md) |
| **GLM4** | 9 B, 32 B | [glm4](dense/glm4.md) |
| **Llama 3** | 3 B, 8 B | [llama3](dense/llama3.md) |
| **MiMo** | 7 B | [mimo](dense/mimo.md) |
| **GPT-OSS** | 20 B | [gpt-oss](dense/gpt-oss.md) |

## Mixture-of-Experts

| Model family | Active / Total | Recipe |
|---|---|---|
| **Qwen3 MoE** | 3 B / 30 B, 22 B / 235 B | [qwen3](moe/qwen3.md) |
| **Qwen3.5 MoE** | 3 B / 35 B | [qwen3-5](moe/qwen3-5.md) |
| **Qwen3-Next** | 3 B / 80 B | [qwen3-next](moe/qwen3-next.md) |
| **GLM4.5** | 12 B / 106 B, 32 B / 355 B | [glm4-5](moe/glm4-5.md) |
| **GLM4.7 Flash** | 4 × 64 experts × 1 shared | [glm4-7-flash](moe/glm4-7-flash.md) |
| **GLM5** | 40 B / 744 B | [glm5](moe/glm5.md) |
| **DeepSeek R1 / V3** | 37 B / 671 B | [deepseek](moe/deepseek.md) |
| **Kimi K2 / K2-Thinking** | 32 B / 1 T | [kimi-k2](moe/kimi-k2.md) |
| **Moonlight** | 3 B / 16 B | [moonlight](moe/moonlight.md) |

## How a recipe is structured

Every recipe page has the same shape:

1. **Supported variants** — which sizes + checkpoints this family covers.
2. **Model configuration** — the `MODEL_ARGS` sourced from `scripts/models/<family>.sh`.
3. **Download + convert** — the exact commands for weights + datasets.
4. **Parallelism** — recommended TP / PP / EP / CP per size.
5. **Launch** — the `scripts/run-<family>.sh` invocation with any env vars required.
6. **Knobs to tune** — what to adjust first when the run is slow or unstable.

Dense pages are short; MoE pages go into extra detail on EP, expert balance, and R3
(see [Miles Router](../advanced/miles-router.md)).

## Adding a new model

Miles's plugin architecture lets you wrap a HuggingFace implementation as a Megatron
module without patching Megatron core. See
[Backends Beyond Megatron](../advanced/architecture-support.md) for the workflow.
