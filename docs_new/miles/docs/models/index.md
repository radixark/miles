---
title: Supported Models
description: Per-family recipes covering weight conversion, launch flags, and parallelism choices.
---

# Supported Models

Miles ships ready-to-run recipes for every model family listed below. Each page covers
weight conversion, parallelism, and the launch script in the order you'd actually run
them.

## By family

| Family | Class | Sizes | Recipes |
|---|---|---|---|
| **Qwen** | Dense & MoE | 0.6 B – 235 B | [overview](qwen/index.md) · [qwen3](qwen/qwen3.md) · [qwen3-moe](qwen/qwen3-moe.md) · [qwen3-5](qwen/qwen3-5.md) · [qwen3-5-moe](qwen/qwen3-5-moe.md) · [qwen3-next](qwen/qwen3-next.md) |
| **GLM** | Dense & MoE | 9 B – 744 B | [overview](glm/index.md) · [glm4](glm/glm4.md) · [glm4-5](glm/glm4-5.md) · [glm4-7-flash](glm/glm4-7-flash.md) · [glm5](glm/glm5.md) |
| **DeepSeek** | MoE | 37 B / 671 B | [deepseek](deepseek/deepseek.md) |
| **Kimi** | MoE | 3 B / 16 B, 32 B / 1 T | [kimi-k2](kimi/kimi-k2.md) · [moonlight](kimi/moonlight.md) |
| **Llama** | Dense | 3 B, 8 B | [llama3](llama/llama3.md) |
| **MiMo** | Dense | 7 B | [mimo](mimo/mimo.md) |
| **GPT-OSS** | Dense | 20 B | [gpt-oss](gpt-oss/gpt-oss.md) |

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
