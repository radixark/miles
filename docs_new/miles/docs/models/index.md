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
