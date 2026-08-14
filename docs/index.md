---
title: Welcome
sidebarTitle: Overview
description: Miles is an open-source RL framework for large-scale LLM post-training, pairing SGLang rollout with Megatron-LM training at trillion-parameter scale.
---
Miles is a high-performance, enterprise-ready reinforcement learning framework for
**large-scale model post-training**. It pairs [SGLang](https://github.com/sgl-project/sglang)
for high-throughput rollout with [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) for
scalable training, and ships the precision, stability and observability features an RL run
needs at trillion-parameter scale. A PyTorch FSDP2 backend is available for runs that would
rather train the HuggingFace implementation as-is, though the recipes, the parallelism and
the largest models all live on Megatron-LM. See
[Training Backends](/user-guide/training-backend).

> *"A journey of a thousand miles begins with a single rollout."*

## Core features

### Performance

- **Fully async RL.** Rollout and training workers are decoupled, with configurable on- and
  off-policy schedules, a pipeline tuned for fewer bubbles, and customizable async rollout
  and eval modes. See [Fully Async RL](/user-guide/fully-async).
- **Fast agentic rollout.** Generation runs on [SGLang](https://github.com/sgl-project/sglang)
  behind a router that spreads requests across engines, preserves per-request metadata and
  health-checks the fleet. Tuned for multi-turn agentic workloads.
- **Fast weight updates.** New weights reach the engines in-loop in seconds, even on a
  trillion-parameter model such as Kimi-K2.6, with
  [P2P RDMA](/advanced/p2p-weight-transfer) as the fast path for disaggregated setups.
- **Low-precision training.** [MXFP8 and NVFP4](/advanced/low-precision) training with a
  numerically stable RL recipe that reduces precision-induced divergence. FP8,
  [INT4 QAT](/advanced/int4-qat), BF16 and FP16 are also supported.
- **LoRA and multi-LoRA.** [Low-rank adapters](/advanced/lora) train frontier-scale models
  on a fraction of the GPUs, and the same adapters load straight into SGLang for rollout.

### Correctness and resilience

- **Token-in-token-out (TITO).** Supported for
  [every model and every black-box harness](/user-guide/agentic-rollout), with no
  detokenize and retokenize round-trip between rollout and training.
- **Rollout Routing Replay (R3).** Expert routing recorded during rollout is
  [replayed in the trainer's forward pass](/advanced/miles-router), removing the MoE routing
  mismatch that destabilizes large runs, with compute and communication overlapped to keep
  the cost down.
- **Fault tolerance.** When an SGLang engine dies, Miles
  [recovers it and resumes the run in place](/advanced/fault-tolerance): no restart, no
  pause.
- **Miles dashboard.** A self-hosted web UI for a run's
  [training dynamics and compute efficiency](/user-guide/dashboard): what every GPU was
  doing during a step, and what each trajectory contained at the token level.

### What Miles runs

- **Day-0 model support.** DeepSeek-V4, Kimi-K3, GLM-5.2, Inkling and Nemotron landed on
  release day. Beyond day 0, nearly every frontier model runs on Miles, including Kimi-K2.6
  and Qwen3.5. See [Supported models](#supported-models).
- **Extensive hardware support.** NVIDIA from H100 through GB300, and AMD MI300X through
  MI355X via ROCm. See [Supported hardware](#supported-hardware).
- **Wide recipe support.** GRPO, GSPO, PPO and REINFORCE++ for RL, plus SFT and
  [on-policy distillation](/advanced/on-policy-distillation).
- **Agentic environments.** Train coding and computer-use agents through connectors for
  Harbor, HUD, NeMo Gym, OpenEnv, Verifiers and more, each plugging into the rollout
  layer that fits it, with task sandboxes on AgentENV, Daytona, E2B or Modal. See
  [Agentic Environments](/user-guide/environments).
- **Comprehensive CI.** Unit suites run on every pull request, and tag-triggered end-to-end
  GPU training tests cover the supported model families on both NVIDIA and AMD runners.

## Supported models

Each model name links to its recipe page or launch script. The table is not
exhaustive — it highlights recent releases; many more models run on Miles out
of the box, including older generations of the families below.

| Family | Models |
|---|---|
| **DeepSeek** | [DeepSeek-V4 Pro](/models/deepseek/deepseek-v4-pro)<br/>[DeepSeek-V4 Flash](/models/deepseek/deepseek-v4-flash)<br/>[DeepSeek-V3.2](/models/deepseek/deepseek-v3-2)<br/>[DeepSeek-V3](/models/deepseek/deepseek) |
| **Thinking Machines** | [Inkling](/models/thinkingmachines/inkling)<br/>[Inkling-Small](/models/thinkingmachines/inkling-small) |
| **Qwen** | [Qwen3.6 MoE](/models/qwen/qwen3-6-moe)<br/>[Qwen3.6](/models/qwen/qwen3-6)<br/>[Qwen3.5-35B-A3B](/models/qwen/qwen3-5-moe)<br/>[Qwen3.5-4B / 9B / 27B](/models/qwen/qwen3-5) |
| **GLM** | [GLM-5.2](/models/glm/glm5-2)<br/>[GLM-5.1](/models/glm/glm5)<br/>[GLM-5](/models/glm/glm5)<br/>[GLM-4.7-Flash](/models/glm/glm4-7-flash) |
| **Kimi** | [Kimi-K3](/models/kimi/kimi-k3)<br/>[Kimi-K2.6](/models/kimi/kimi-k2.5)<br/>[Kimi-K2.5](/models/kimi/kimi-k2.5) |
| **Nemotron** | [Nemotron-3-Ultra-550B-A55B](/models/nemotron/nemotron-3-ultra)<br/>[Nemotron-3-Super-120B-A12B-FP8](/models/nemotron/nemotron-3-super)<br/>[Nemotron-3-Nano MoE](/models/nemotron/nemotron-3-nano-moe)<br/>[Nemotron-3-Nano](/models/nemotron/nemotron-3-nano) |
| **Gemma** | [Gemma-4 26B-A4B](/models/gemma/gemma-4)<br/>[Gemma-4 31B](/models/gemma/gemma-4) |
| **JoyAI** | [JoyAI-LLM-Flash](https://github.com/radixark/miles/blob/main/scripts/run_joy_ai_llm_flash.py) |

See [Models](/models/index) for exact conversion commands, launch scripts, and
parallelism settings.

## Supported hardware

- **NVIDIA**: GB300, GB200, B300, B200, H200, H100, A100.
- **AMD**: MI300X, MI325, MI350, MI355X (via ROCm).

See [Installation](/getting-started/installation#hardware-requirements) for per-GPU status
and the container images for each.

## News

- [2026/07] Towards Blackwell-Native 8-bit and 4-bit RL: End-to-End MXFP8 and NVFP4 RL in Miles ([blog](https://www.lmsys.org/blog/2026-07-29-mxfp8-nvfp4-rl)).
- [2026/07] 🔥 SGLang and Miles add day-0 support for Kimi K3 ([blog](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support)).
- [2026/07] On-policy distillation lands in Miles ([blog](https://www.lmsys.org/blog/2026-07-18-opd-support-in-miles)).
- [2026/07] 🔥 SGLang and Miles add day-0 support for Inkling, a frontier multimodal model ([blog](https://www.lmsys.org/blog/2026-07-15-inkling-day0-support)).
- [2026/07] DeepSeek-V4 Flash RL training comes to AMD Instinct MI355X with Miles ([blog](https://www.lmsys.org/blog/2026-07-10-rocm-miles-dsv4)).
- [2026/06] SGLang and Miles add day-0 support for NVIDIA Nemotron 3 Ultra ([blog](https://www.lmsys.org/blog/2026-06-04-nvidia-run-nemotron-3-ultra)).
- [2026/05] No token left behind: token-in-token-out in Miles ([blog](https://www.lmsys.org/blog/2026-05-13-no-token-left-behind)).
- [2026/04] Updating 1 T parameters in seconds: P2P weight transfer in large-scale distributed RL ([blog](https://www.lmsys.org/blog/2026-04-29-p2p-update)).
- [2026/04] 🔥 DeepSeek-V4 on day 0: from fast inference to verified RL with SGLang and Miles ([blog](https://www.lmsys.org/blog/2026-04-25-deepseek-v4)).

## Start here

1. **[Installation](/getting-started/installation)** — Docker, bare metal, AMD.
2. **[Quick Start](/getting-started/quick-start)** — a working training run in under an hour.
3. **[Core concepts](/user-guide/concepts)** — the four objects in every Miles job.
4. **[Launch script](/user-guide/launch-script)** — what `python scripts/run_*.py` does
   and how to override a recipe.
5. **[Training backends](/user-guide/training-backend)** — Megatron-LM and FSDP: parallelism,
   checkpoints, and hooks.

## Contribute

- GitHub: [github.com/radixark/miles](https://github.com/radixark/miles)
- Slack: [slack.sglang.ai](https://slack.sglang.ai), channel `#miles`
- Contributing: [developer guide](/developer/contributor-guide)
