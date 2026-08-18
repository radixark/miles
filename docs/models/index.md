---
title: Supported Models
sidebarTitle: Overview
description: Per-family recipes covering weight conversion, launch flags, and parallelism choices.
---
Miles ships ready-to-run recipes for every model family listed below. Each page covers
weight conversion, parallelism, and the launch script in the order you'd actually run
them.

## Language Models

Each model name links to its recipe page.

| Family | Models |
|---|---|
| **DeepSeek** | [DeepSeek-V4 Pro](/models/deepseek/deepseek-v4-pro)<br/>[DeepSeek-V4 Flash](/models/deepseek/deepseek-v4-flash)<br/>[DeepSeek-V3.2](/models/deepseek/deepseek-v3-2)<br/>[DeepSeek-V3](/models/deepseek/deepseek) |
| **Thinking Machines** | [Inkling](/models/thinkingmachines/inkling)<br/>[Inkling-Small](/models/thinkingmachines/inkling-small) |
| **Qwen** | [Qwen3.8-27B](/models/qwen/qwen3-8)<br/>[Qwen3.6 MoE](/models/qwen/qwen3-6-moe)<br/>[Qwen3.6](/models/qwen/qwen3-6)<br/>[Qwen3.5-35B-A3B](/models/qwen/qwen3-5-moe)<br/>[Qwen3.5-4B / 9B / 27B](/models/qwen/qwen3-5)<br/>[Qwen3-Next-80B-A3B-Thinking](/models/qwen/qwen3-next)<br/>[Qwen3-30B-A3B / 235B-A22B](/models/qwen/qwen3-moe)<br/>[Qwen3-0.6B / 1.7B / 4B / 8B / 14B / 32B](/models/qwen/qwen3) |
| **GLM** | [GLM-5.2](/models/glm/glm5-2)<br/>[GLM-5.1](/models/glm/glm5)<br/>[GLM-5](/models/glm/glm5)<br/>[GLM-4.7-Flash](/models/glm/glm4-7-flash)<br/>[GLM-4.5](/models/glm/glm4-5) |
| **Kimi** | [Kimi-K3](/models/kimi/kimi-k3)<br/>[Kimi-K2.6](/models/kimi/kimi-k2.5)<br/>[Kimi-K2.5](/models/kimi/kimi-k2.5)<br/>[Kimi-K2-Instruct / Thinking](/models/kimi/kimi-k2) |
| **Nemotron** | [Nemotron-3-Ultra-550B-A55B](/models/nemotron/nemotron-3-ultra)<br/>[Nemotron-3-Super-120B-A12B-FP8](/models/nemotron/nemotron-3-super)<br/>[Nemotron-3-Nano MoE](/models/nemotron/nemotron-3-nano-moe)<br/>[Nemotron-3-Nano](/models/nemotron/nemotron-3-nano) |
| **Gemma** | [Gemma-4 26B-A4B](/models/gemma/gemma-4)<br/>[Gemma-4 31B](/models/gemma/gemma-4) |
| **JoyAI** | [JoyAI-LLM-Flash](https://github.com/radixark/miles/blob/main/scripts/run_joy_ai_llm_flash.py) |
| **GPT-OSS** | [gpt-oss-20b](/models/gpt-oss/gpt-oss) |

## Diffusion

| Family | Models |
|---|---|
| **Stable Diffusion** | [SD3.5](/diffusion/models/sd3/sd3) |
| **Qwen-Image** | [Qwen-Image](/diffusion/models/qwen-image/qwen-image) |
| **Wan** | [Wan2.2-T2V-A14B](/diffusion/models/wan/wan2-2) |
| **LTX** | [LTX-2.3](/diffusion/models/ltx/ltx2) |
| **Cosmos** | [Cosmos3-Nano](/diffusion/models/cosmos/cosmos3) |
| **MiniMax** | [MiniMax H3](/diffusion/models/h3/h3) |

## How a recipe is structured

Every recipe page follows the same six sections:

1. **Model Introduction** — what the model is and why miles supports it.
2. **Supported Variants** — model sizes + HF links.
3. **Environment Setup** — env vars, downloads, and HF → Megatron conversion.
4. **Launch** — the `scripts/run_<family>.py` invocation.
5. **Recipe Configuration** — parallelism, algorithm, rollout/SGLang, optimizer.
6. **Pairs Well With** — links to the advanced features that complement this recipe.

## Adding a new model

Miles's plugin architecture lets you wrap a HuggingFace implementation as a Megatron
module without patching Megatron core. See
[Backends Beyond Megatron](/advanced/architecture-support) for the workflow.
