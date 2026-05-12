---
title: Miles Documentation
---
Miles is a high-performance, enterprise-ready reinforcement learning (RL) framework specifically optimized for **Large-Scale model Post-Training**. It
couples [SGLang](https://github.com/sgl-project/sglang) for high-throughput rollout with
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) for scalable training, and ships the precision, stability, and observability features
needed to run RL at trillion-parameter scale.


*"A journey of a thousand miles begins with a single rollout."* — Miles focuses on the low-level system optimizations that make large-scale RL stable, efficient, and reproducible.

## Core features

- **Fast and stable support for the latest models.** Day-0 enablement of frontier
  releases such as DeepSeek-V4, with rapid follow-on support for new architectures
  including GLM-5, Qwen 3.6, and Nemotron-3-Super.
- **Unified low-precision training.** Customizable precision across the rollout and
  training engines, with unified **BF16**, **FP8**, **MXFP8**, and **INT4 QAT** recipes
  available now and an **NVFP4** training recipe in progress.
- **Efficient Rollout Routing Replay (R3).** For MoE models, expert routing captured
  during inference is replayed during the trainer's forward pass, eliminating the
  mismatch that destabilizes large-scale MoE RL. Optimized with a routing-result cache
  and overlapped device-to-host (D2H) copy to reduce overhead in both single-turn and
  multi-turn RL.
- **Speculative rollout with online MTP-SFT.** Miles keeps the draft model's acceptance
  rate high through training by fine-tuning MTP layers on-policy.
- **LoRA training and serving.** Both SFT and RL recipes support LoRA adapters,
  and the same adapters load directly into SGLang for rollout — no separate
  merge or conversion step.
- **Native agentic rollout.** Tool use, multi-turn dialogue, search, code
  execution, and multi-agent co-evolution are all supported through clean Python
  extension points.
- **Minimal core, maximal extension.** Twenty-plus plug-points let you replace the
  rollout, reward, loss, or filter without forking the trainer.
- **Broad hardware support.** First-class on NVIDIA Hopper (H100, H200) and
  Blackwell (B100, B200, GB200, GB300), with AMD MI300X / MI325 / MI350 /
  MI355X also supported via ROCm.

## Supported models

Each model name links to its recipe page.

| Family | Models |
|---|---|
| **DeepSeek** | [DeepSeek-V4 Pro](/miles/docs/models/deepseek/deepseek-v4-pro)<br/>[DeepSeek-V4 Flash](/miles/docs/models/deepseek/deepseek-v4-flash)<br/>[DeepSeek-R1](/miles/docs/models/deepseek/deepseek)<br/>[DeepSeek-V3](/miles/docs/models/deepseek/deepseek) |
| **Qwen** | [Qwen3.6 MoE](/miles/docs/models/qwen/qwen3-6-moe)<br/>[Qwen3.6](/miles/docs/models/qwen/qwen3-6)<br/>[Qwen3.5-35B-A3B](/miles/docs/models/qwen/qwen3-5-moe)<br/>[Qwen3.5-4B / 9B / 27B](/miles/docs/models/qwen/qwen3-5)<br/>[Qwen3-Next-80B-A3B-Thinking](/miles/docs/models/qwen/qwen3-next)<br/>[Qwen3-30B-A3B / 235B-A22B](/miles/docs/models/qwen/qwen3-moe)<br/>[Qwen3-0.6B / 1.7B / 4B / 8B / 14B / 32B](/miles/docs/models/qwen/qwen3) |
| **GLM** | [GLM-5.1](/miles/docs/models/glm/glm5)<br/>[GLM-5](/miles/docs/models/glm/glm5)<br/>[GLM-4.7-Flash](/miles/docs/models/glm/glm4-7-flash)<br/>[GLM-4.5](/miles/docs/models/glm/glm4-5)<br/>[GLM-Z1-9B-0414](/miles/docs/models/glm/glm4) |
| **Kimi** | [Kimi-K2.6](/miles/docs/models/kimi/kimi-k2.5)<br/>[Kimi-K2.5](/miles/docs/models/kimi/kimi-k2.5)<br/>[Kimi-K2-Instruct / Thinking](/miles/docs/models/kimi/kimi-k2)<br/>[Moonlight-16B-A3B](/miles/docs/models/kimi/moonlight) |
| **Nemotron** | [Nemotron-3-Super-120B-A12B-FP8](/miles/docs/models/nemotron/nemotron-3-super)<br/>[Nemotron-3-Nano MoE](/miles/docs/models/nemotron/nemotron-3-nano-moe)<br/>[Nemotron-3-Nano](/miles/docs/models/nemotron/nemotron-3-nano) |
| **MiMo** | [MiMo-7B-RL](/miles/docs/models/mimo/mimo) |
| **GPT-OSS** | [gpt-oss-20b](/miles/docs/models/gpt-oss/gpt-oss) |

See [Models](/miles/docs/models/index) for exact conversion commands, launch scripts, and
parallelism settings.

## Supported hardware

- **NVIDIA**: GB300, GB200, B200, B100, H200, H100, A100.
- **AMD**: MI300X, MI325, MI350, MI355X (via ROCm).

See [Platforms](/miles/docs/platforms/index).

## Latest updates

- **[2026/02]** Complete argument reference. [CLI Reference](/miles/docs/user-guide/cli-reference)
- **[2026/01]** INT4 W4A16 QAT. [INT4 Quantization-Aware Training](/miles/docs/advanced/int4-qat)
- **[2026/01]** Unified VLM/LLM multi-turn rollout. [Multi-Agent Co-Evolution](/miles/docs/examples/multi-agent)
- **[2025/12]** Rollout Routing Replay (R3) for MoE. [Rollout Routing Replay (R3)](/miles/docs/advanced/miles-router)
- **[2025/11]** Unified FP8 pipeline generally available. [FP8 and Low Precision](/miles/docs/advanced/fp8-low-precision)
- **[2025/11]** Speculative decoding with online MTP-SFT. [Speculative Decoding](/miles/docs/advanced/speculative-decoding)

## Start here

1. **[Installation](/miles/docs/getting-started/installation)** — Docker, bare metal, AMD.
2. **[Quick Start](/miles/docs/getting-started/quick-start)** — a working training run in under an hour.
3. **[Core concepts](/miles/docs/user-guide/concepts)** — the four objects in every Miles job.
4. **[Training backend](/miles/docs/user-guide/usage)** — Megatron-LM, parallelism, checkpoints, and hooks.
5. **[Training script walkthrough](/miles/docs/user-guide/training-script-walkthrough)** — every
   argument group in a launch script, annotated.

## Contribute

- GitHub: [github.com/radixark/miles](https://github.com/radixark/miles)
- Slack: [slack.sglang.ai](https://slack.sglang.ai), channel `#miles`
- Contributing: [developer guide](/miles/docs/developer/contributor-guide)
