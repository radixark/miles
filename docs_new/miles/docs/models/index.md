---
title: Supported Models
description: Per-family recipes covering weight conversion, launch flags, and parallelism choices.
---

# Supported Models

Miles ships ready-to-run recipes for every model family listed below. Each page covers
weight conversion, parallelism, and the launch script in the order you'd actually run
them.

## By family

| Family | Class | Sizes |
|---|---|---|
| **DeepSeek** | MoE | 37 B / 671 B |
| **Qwen** | Dense & MoE | 0.6 B – 235 B |
| **GLM** | Dense & MoE | 9 B – 744 B | 
| **Kimi** | MoE | 3 B / 16 B, 32 B / 1 T |
| **MiMo** | Dense | 7 B | 
| **GPT-OSS** | Dense | 20 B | 

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
