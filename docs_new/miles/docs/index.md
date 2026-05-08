---
title: Miles Documentation
---

# Miles

Miles is a high-performance, enterprise-ready reinforcement learning (RL) framework specifically optimized for **Large-Scale model Post-Training**. It
couples [SGLang](https://github.com/sgl-project/sglang) for high-throughput rollout with
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) for scalable training, and ships the precision, stability, and observability features
needed to run RL at trillion-parameter scale.


*"A journey of a thousand miles begins with a single rollout."* — Miles focuses on the low-level system optimizations that make large-scale RL stable, efficient, and reproducible.

## Core features

- **Fast and stable support for the latest models.** Day-0 enablement of frontier
  releases such as DeepSeek V4, with rapid follow-on support for new architectures
  including GLM-5, Qwen 3.6, and Nemotron-3-Super.
- **Unified low-precision training.** Customisable precision across the rollout and
  training engines, with unified **BF16**, **FP8**, **MXFP8**, and **INT4 QAT** recipes
  available now and an **NVFP4** training recipe in progress.
- **Efficient Rollout Routing Replay (R3).** For MoE models, expert routing captured
  during inference is replayed during the trainer's forward pass, eliminating the
  mismatch that destabilises large-scale MoE RL. Optimised with a routing-result cache
  and overlapped device-to-host (D2H) copy to reduce overhead in both single-turn and
  multi-turn RL.
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

- **NVIDIA**: GB300, GB200, B200, B100, H200, H100, A100.
- **AMD**: MI300X, MI325, MI350, MI355X (via ROCm).

See [Platforms](platforms/index.md).

## Latest updates

- **[2026/02]** Complete argument reference. [CLI Reference](user-guide/cli-reference.md)
- **[2026/01]** INT4 W4A16 QAT. [INT4 Quantization-Aware Training](advanced/int4-qat.md)
- **[2026/01]** Unified VLM/LLM multi-turn rollout. [Multi-Agent Co-Evolution](examples/multi-agent.md)
- **[2025/12]** Rollout Routing Replay (R3) for MoE. [Rollout Routing Replay (R3)](advanced/miles-router.md)
- **[2025/11]** Unified FP8 pipeline generally available. [FP8 and Low Precision](advanced/fp8-low-precision.md)
- **[2025/11]** Speculative decoding with online MTP-SFT. [Speculative Decoding](advanced/speculative-decoding.md)

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
