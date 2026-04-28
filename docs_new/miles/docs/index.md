---
title: Miles Documentation
---

# Miles

Miles is a high-performance, enterprise-ready reinforcement learning (RL) framework specifically optimized for **Large-Scale model Post-Training**. It
couples [SGLang](https://github.com/sgl-project/sglang) for high-throughput rollout with
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [FSDP](https://docs.pytorch.org/docs/stable/fsdp.html)
for scalable training, and ships the precision, stability, and observability features
needed to run RL at trillion-parameter scale.


*"A journey of a thousand miles begins with a single rollout."* — Miles focuses on the low-level system optimizations that make large-scale RL stable, efficient, and reproducible.

## Core features

- **Unified low-precision training.** FP8 sampling and FP8 training share a single
  quantisation path, so the rollout policy and the training policy are bit-identical.
- **Rollout Routing Replay (R3).** For MoE models, expert routing captured during
  inference is replayed during the trainer's forward pass, eliminating the mismatch that
  destabilises large-scale MoE RL.
- **Speculative rollout with online MTP-SFT.** Miles keeps the draft model's acceptance
  rate high through training by fine-tuning MTP layers on-policy.
- **Fault tolerance.** Rank-level recovery, step-level replay, and RDMA P2P weight
  sync let weeks-long runs survive routine hardware faults.
- **First-class agentic rollout.** Tool use, multi-turn dialogue, search, code
  execution, and multi-agent co-evolution are all supported through clean Python
  extension points.
- **Minimal core, maximal extension.** Twenty-plus plug-points let you replace the
  rollout, reward, loss, or filter without forking the trainer.

## Supported models

Dense — Qwen3.5, Qwen3, GLM4, MiMo, GPT-OSS.

Mixture-of-Experts — DeepSeek-V4, DeepSeek V3, Qwen3.5 MoE, Qwen3-Next, Qwen3 MoE, GLM5, GLM4.7, GLM4.5, Kimi K2, Moonlight.

See [Models](models/index.md) for per-family recipes.

## Supported hardware

NVIDIA H100, H200, B100, B200, A100. AMD MI300X, MI325, MI350, MI355X via ROCm. See
[Platforms](platforms/index.md).

## Latest updates

- **[2026/02]** Complete argument reference. [Server arguments](user-guide/cli-reference.md)
- **[2026/01]** INT4 W4A16 QAT. [Low-precision guide](advanced/int4-qat.md)
- **[2026/01]** Unified VLM/LLM multi-turn rollout. [Multi-agent walkthrough](examples/multi-agent.md)
- **[2025/12]** Rollout Routing Replay (R3) for MoE. [Design doc](advanced/miles-router.md)
- **[2025/11]** Unified FP8 pipeline generally available. [Post](advanced/fp8-low-precision.md)
- **[2025/11]** Speculative decoding with online MTP-SFT. [Docs](advanced/speculative-decoding.md)

## Start here

1. **[Installation](getting-started/installation.md)** — Docker, bare metal, AMD.
2. **[Quick Start](getting-started/quick-start.md)** — a working training run in under an hour.
3. **[Core concepts](user-guide/concepts.md)** — the four objects in every Miles job.
4. **[Training backends](user-guide/usage.md)** — choosing between Megatron and FSDP.
5. **[Training script walkthrough](user-guide/training-script-walkthrough.md)** — every
   argument group in a launch script, annotated.

## Contribute

- GitHub: [github.com/radixark/miles](https://github.com/radixark/miles)
- Slack: [slack.sglang.ai](https://slack.sglang.ai), channel `#miles`
- Contributing: [developer guide](developer/contributing.md)
