---
title: Miles Documentation
---
Miles is a high-performance, enterprise-ready reinforcement learning (RL) framework specifically optimized for **Large-Scale model Post-Training**. It
couples [SGLang](https://github.com/sgl-project/sglang) for high-throughput rollout with
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) for scalable training, and ships the precision, stability, and observability features
needed to run RL at trillion-parameter scale.


*"A journey of a thousand miles begins with a single rollout."* — Miles focuses on the low-level system optimizations that make large-scale RL stable, efficient, and reproducible.

## Core features

### Efficiency & stability

- **Fully async RL.** Rollout and training workers are decoupled, with configurable
  on- and off-policy schedules, an optimized pipeline with fewer bubbles, and
  customizable async rollout and eval modes. See
  [Fully Async RL](/user-guide/fully-async).
- **Fast agentic rollout.** High-throughput generation on
  [SGLang](https://github.com/sgl-project/sglang), optimized for multi-turn
  agentic workloads.
- **Fast weight updates.** Updated weights sync back in-loop without pausing
  rollout — under 10 seconds for a model with 1 T parameters — with
  [P2P RDMA](/advanced/p2p-weight-transfer) as a fast path for disaggregated setups.
- **Unified low-precision training.** [MXFP8 and NVFP4](/advanced/fp8-low-precision)
  training with a numerically stable RL recipe that reduces precision-induced
  divergence; FP8, [INT4 QAT](/advanced/int4-qat), BF16, and FP16 are also supported.
- **Token-in-token-out (TITO).** Supported for
  [all models and all black-box agent harnesses](/user-guide/agentic-chat-template) —
  no detokenize/retokenize round-trips between rollout and training.
- **Rollout Routing Replay (R3).** Expert routing recorded during rollout is
  [replayed in the trainer's forward pass](/advanced/miles-router), eliminating the
  routing mismatch that destabilizes large-scale MoE RL, with compute and
  communication overlapped to minimize overhead.
- **LoRA and multi-LoRA.** [Low-rank adapters](/advanced/lora) train frontier-scale
  models on a fraction of the GPUs, and the same adapters load directly into SGLang
  for rollout — no separate merge or conversion step.
- **Fault tolerance.** When an SGLang engine dies, Miles
  [recovers it and resumes the run in place](/advanced/fault-tolerance) — no
  restart, no pause.
- **Day-0 model support.** Day-0 enablement of frontier releases such as
  DeepSeek-V4, Kimi-K3, GLM-5.2, Inkling, and Nemotron — and beyond day-0, nearly
  all frontier models (see [Supported models](#supported-models)).

### Design, support & user experience

- **Coding-agent sandboxes and examples.** [Harbor](/user-guide/harbor),
  [OpenEnv](/user-guide/openenv), and [NeMo Gym](/user-guide/nemo-gym) integrations,
  running local CPU sandboxes or per-episode sandboxes on
  [Daytona](https://www.daytona.io/), [E2B](https://e2b.dev/), and self-hosted
  [AgentENV](https://github.com/kvcache-ai/AgentENV) — see
  [Environments](/user-guide/environments) for the support matrix.
- **Highly customizable pipeline.** Shape every workload through
  [twenty-plus plug-points](/user-guide/customization), from reward computation to
  the full rollout function.
- **Megatron or FSDP.**
  [Switch training backends](/developer/experimental-features#fsdp-backend) without
  rewriting your training loop.
- **Wide recipe support.** RL (GRPO, PPO), SFT, and on-policy distillation.
- **Verified on multiple hardware generations.** GB300, GB200, B300, B200, H200,
  H100, and AMD MI355X / MI300X — see [Platforms](/platforms/index).
- **Comprehensive CI.** Unit suites run on every pull request, and tag-triggered
  end-to-end GPU training tests cover the supported model families on both NVIDIA
  and AMD runners.
- **[Miles dashboard](/user-guide/dashboard).** A self-hosted web UI for a run's
  training dynamics and compute efficiency: what every GPU was doing during a step,
  and what each trajectory contained at the token level.

## Supported models

Each model name links to its recipe page or launch script. The table is not
exhaustive — it highlights recent releases; many more models run on Miles out
of the box, including older generations of the families below.

| Family | Models |
|---|---|
| **DeepSeek** | [DeepSeek-V4 Pro](/models/deepseek/deepseek-v4-pro)<br/>[DeepSeek-V4 Flash](/models/deepseek/deepseek-v4-flash) |
| **Thinking Machines** | [Inkling](/models/thinkingmachines/inkling)<br/>[Inkling-Small](/models/thinkingmachines/inkling-small) |
| **Qwen** | [Qwen3.6 MoE](/models/qwen/qwen3-6-moe)<br/>[Qwen3.6](/models/qwen/qwen3-6)<br/>[Qwen3.5-35B-A3B](/models/qwen/qwen3-5-moe)<br/>[Qwen3.5-4B / 9B / 27B](/models/qwen/qwen3-5) |
| **GLM** | [GLM-5.2](/models/glm/glm5-2)<br/>[GLM-5.1](/models/glm/glm5)<br/>[GLM-5](/models/glm/glm5)<br/>[GLM-4.7-Flash](/models/glm/glm4-7-flash) |
| **Kimi** | [Kimi-K3](https://github.com/radixark/miles/pull/1825)<br/>[Kimi-K2.6](/models/kimi/kimi-k2.5)<br/>[Kimi-K2.5](/models/kimi/kimi-k2.5) |
| **Nemotron** | [Nemotron-3-Ultra-550B-A55B](https://github.com/radixark/miles/blob/main/scripts/run_nemotron_3_ultra_550b_a55b.py)<br/>[Nemotron-3-Super-120B-A12B-FP8](/models/nemotron/nemotron-3-super)<br/>[Nemotron-3-Nano MoE](/models/nemotron/nemotron-3-nano-moe)<br/>[Nemotron-3-Nano](/models/nemotron/nemotron-3-nano) |
| **Gemma** | [Gemma-4 26B-A4B](https://github.com/radixark/miles/blob/main/scripts/run_gemma_4_26b_a4b.py)<br/>[Gemma-4 31B](https://github.com/radixark/miles/blob/main/scripts/run_gemma_4_31b.py) |
| **JoyAI** | [JoyAI-LLM-Flash](https://github.com/radixark/miles/blob/main/scripts/run_joy_ai_llm_flash.py) |

See [Models](/models/index) for exact conversion commands, launch scripts, and
parallelism settings.

## Supported hardware

- **NVIDIA**: GB300, GB200, B200, B100, H200, H100, A100.
- **AMD**: MI300X, MI325, MI350, MI355X (via ROCm).

See [Platforms](/platforms/index).

## Latest updates

- **[2026/02]** Complete argument reference. [CLI Reference](/user-guide/cli-reference)
- **[2026/01]** INT4 W4A16 QAT. [INT4 Quantization-Aware Training](/advanced/int4-qat)
- **[2026/01]** Unified VLM/LLM multi-turn rollout. [Multi-Agent Co-Evolution](/examples/multi-agent)
- **[2025/12]** Rollout Routing Replay (R3) for MoE. [Rollout Routing Replay (R3)](/advanced/miles-router)
- **[2025/11]** Unified FP8 pipeline generally available. [FP8 and Low Precision](/advanced/fp8-low-precision)
- **[2025/11]** Speculative decoding with online MTP-SFT. [Speculative Decoding](/advanced/speculative-decoding)

## Start here

1. **[Installation](/getting-started/installation)** — Docker, bare metal, AMD.
2. **[Quick Start](/getting-started/quick-start)** — a working training run in under an hour.
3. **[Core concepts](/user-guide/concepts)** — the four objects in every Miles job.
4. **[Training backend](/user-guide/usage)** — Megatron-LM, parallelism, checkpoints, and hooks.
5. **[Training script walkthrough](/user-guide/training-script-walkthrough)** — every
   argument group in a launch script, annotated.

## Contribute

- GitHub: [github.com/radixark/miles](https://github.com/radixark/miles)
- Slack: [slack.sglang.ai](https://slack.sglang.ai), channel `#miles`
- Contributing: [developer guide](/developer/contributor-guide)
