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
  releases such as DeepSeek-V4, with rapid follow-on support for new architectures
  including GLM-5, Qwen 3.6, and Nemotron-3-Super.
- **Unified low-precision training.** Customisable precision across the rollout and
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

<CardGroup cols={2}>

  <Card title="Dense">

    - **Qwen**: <span className="model-list">[Qwen3.6](models/qwen/qwen3-6.md), [Qwen3.5](models/qwen/qwen3-5.md), [Qwen3](models/qwen/qwen3.md)</span>
    - **GLM**: <span className="model-list">[GLM4](models/glm/glm4.md)</span>
    - **Nemotron**: <span className="model-list">[Nemotron-3-Nano](models/nemotron/nemotron-3-nano.md)</span>
    - **MiMo**: <span className="model-list">[MiMo](models/mimo/mimo.md)</span>
    - **GPT-OSS**: <span className="model-list">[GPT-OSS](models/gpt-oss/gpt-oss.md)</span>

  </Card>

  <Card title="Mixture of Experts">

    - **DeepSeek**: <span className="model-list">[DeepSeek-V4 Pro](models/deepseek/deepseek-v4-pro.md), [DeepSeek-V4 Flash](models/deepseek/deepseek-v4-flash.md), [DeepSeek-V3 / R1](models/deepseek/deepseek.md)</span>
    - **Qwen**: <span className="model-list">[Qwen3.6 MoE](models/qwen/qwen3-6-moe.md), [Qwen3.5 MoE](models/qwen/qwen3-5-moe.md), [Qwen3-Next](models/qwen/qwen3-next.md), [Qwen3 MoE](models/qwen/qwen3-moe.md)</span>
    - **GLM**: <span className="model-list">[GLM5 / GLM5.1](models/glm/glm5.md), [GLM4.7](models/glm/glm4-7-flash.md), [GLM4.5](models/glm/glm4-5.md)</span>
    - **Kimi**: <span className="model-list">[Kimi K2.5 / K2.6](models/kimi/kimi-k2.5.md), [Kimi K2](models/kimi/kimi-k2.md), [Moonlight](models/kimi/moonlight.md)</span>
    - **Nemotron**: <span className="model-list">[Nemotron-3-Nano MoE](models/nemotron/nemotron-3-nano-moe.md), [Nemotron-3-Super](models/nemotron/nemotron-3-super.md)</span>

  </Card>

</CardGroup>

See [Models](models/index.md) for exact conversion commands, launch scripts, and
parallelism settings.

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
4. **[Training backend](user-guide/usage.md)** — Megatron-LM, parallelism, checkpoints, and hooks.
5. **[Training script walkthrough](user-guide/training-script-walkthrough.md)** — every
   argument group in a launch script, annotated.

## Contribute

- GitHub: [github.com/radixark/miles](https://github.com/radixark/miles)
- Slack: [slack.sglang.ai](https://slack.sglang.ai), channel `#miles`
- Contributing: [developer guide](developer/contributing.md)
