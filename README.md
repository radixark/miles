aaaaaaaa

<div align="center">

<img src="https://raw.githubusercontent.com/radixark/miles/main/docs/assets/images/brand/miles_logo.png" alt="Miles Logo" width="340">

### **Enterprise-Grade Reinforcement Learning for Large-Scale Model Post-Training**

[![Website](https://img.shields.io/badge/website-miles.radixark.com-d55816)](https://miles.radixark.com/)
[![GitHub Repo](https://img.shields.io/badge/github-radixark%2Fmiles-black?logo=github)](https://github.com/radixark/miles)
[![Docs](https://img.shields.io/badge/docs-miles.radixark.com%2Fdocs-d55816)](https://miles.radixark.com/docs)
[![License](https://img.shields.io/github/license/radixark/miles)](LICENSE)
[![Slack](https://img.shields.io/badge/slack-%23miles--rl-brightgreen.svg)](https://slack.sglang.ai)

| [**Website**](https://miles.radixark.com/) | [**Documentation**](https://miles.radixark.com/docs) | [**Quick Start**](https://miles.radixark.com/docs/getting-started/quick-start) | [**Supported Models**](https://miles.radixark.com/docs/models) | [**Miles Diffusion**](https://github.com/radixark/miles_diffusion) | [**Blog**](https://www.lmsys.org/blog?filter=miles) | [**Slack**](https://slack.sglang.ai) (`#miles-rl`) |

</div>

--------------------------------------------------------------------------------

## News

- [2026/08] 🔥 Miles v0.1 is released! Read the blog post here: [Miles v0.1: Production-level Post-training](https://www.lmsys.org/blog/2026-08-18-miles-v0-1).
- [2026/07] Towards Blackwell-Native 8-bit and 4-bit RL: End-to-End MXFP8 and NVFP4 RL in Miles ([blog](https://www.lmsys.org/blog/2026-07-29-mxfp8-nvfp4-rl)).
- [2026/07] 🔥 SGLang and Miles add day-0 support for Kimi K3 ([blog](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support)).
- [2026/07] On-policy distillation lands in Miles ([blog](https://www.lmsys.org/blog/2026-07-18-opd-support-in-miles)).
- [2026/07] 🔥 SGLang and Miles add day-0 support for Inkling, a frontier multimodal model ([blog](https://www.lmsys.org/blog/2026-07-15-inkling-day0-support)).
- [2026/07] DeepSeek-V4 Flash RL training comes to AMD Instinct MI355X with Miles ([blog](https://www.lmsys.org/blog/2026-07-10-rocm-miles-dsv4)).
- [2026/06] SGLang and Miles add day-0 support for NVIDIA Nemotron 3 Ultra ([blog](https://www.lmsys.org/blog/2026-06-04-nvidia-run-nemotron-3-ultra)).
- [2026/05] No token left behind: token-in-token-out in Miles ([blog](https://www.lmsys.org/blog/2026-05-13-no-token-left-behind)).
- [2026/04] Updating 1 T parameters in seconds: P2P weight transfer in large-scale distributed RL ([blog](https://www.lmsys.org/blog/2026-04-29-p2p-update)).
- [2026/04] 🔥 DeepSeek-V4 on day 0: from fast inference to verified RL with SGLang and Miles ([blog](https://www.lmsys.org/blog/2026-04-25-deepseek-v4)).

## About

Miles is a high-performance, enterprise-ready reinforcement learning framework for
**large-scale model post-training**. It pairs [SGLang](https://github.com/sgl-project/sglang)
for high-throughput rollout with [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) for
scalable training, and ships the precision, stability, and observability features an RL run
needs at trillion-parameter scale. A PyTorch FSDP2 backend is available for runs that would
rather train the HuggingFace implementation as-is, though the recipes, the parallelism, and
the largest models all live on Megatron-LM. See
[Training Backends](https://miles.radixark.com/docs/user-guide/training-backend).

> *"A journey of a thousand miles begins with a single rollout."*

### Performance

- **Fully async RL.** Rollout and training workers are decoupled, with configurable on- and
  off-policy schedules, a pipeline tuned for fewer bubbles, and customizable async rollout
  and eval modes. See [Fully Async RL](https://miles.radixark.com/docs/user-guide/fully-async).
- **Fast agentic rollout.** Generation runs on [SGLang](https://github.com/sgl-project/sglang)
  behind a router that spreads requests across engines, preserves per-request metadata, and
  health-checks the fleet. Tuned for multi-turn agentic workloads.
- **Fast weight updates.** New weights reach the engines in-loop in seconds, even on a
  trillion-parameter model such as Kimi-K2.6, with
  [P2P RDMA](https://miles.radixark.com/docs/advanced/p2p-weight-transfer) as the fast path
  for disaggregated setups.
- **Low-precision training.** [MXFP8 and NVFP4](https://miles.radixark.com/docs/advanced/low-precision)
  training with a numerically stable RL recipe that reduces precision-induced divergence.
  FP8, [INT4 QAT](https://miles.radixark.com/docs/advanced/int4-qat), BF16, and FP16 are also
  supported.
- **LoRA and multi-LoRA.** [Low-rank adapters](https://miles.radixark.com/docs/advanced/lora)
  train frontier-scale models on a fraction of the GPUs, and the same adapters load straight
  into SGLang for rollout.

### Correctness and resilience

- **Token-in-token-out (TITO).** Supported for
  [every model and every black-box harness](https://miles.radixark.com/docs/user-guide/agentic-rollout),
  with no detokenize/retokenize round-trip between rollout and training.
- **Rollout Routing Replay (R3).** Expert routing recorded during rollout is
  [replayed in the trainer's forward pass](https://miles.radixark.com/docs/advanced/miles-router),
  removing the MoE routing mismatch that destabilizes large runs, with compute and
  communication overlapped to keep the cost down.
- **Fault tolerance.** When an SGLang engine dies, Miles
  [recovers it and resumes the run in place](https://miles.radixark.com/docs/advanced/fault-tolerance):
  no restart, no pause.

### What Miles runs

- **Day-0 model support.** DeepSeek-V4, Kimi-K3, GLM-5.2, Inkling, and Nemotron landed on
  release day. Beyond day 0, nearly every frontier model runs on Miles, including Kimi-K2.6
  and Qwen3.5. See [Models](https://miles.radixark.com/docs/models).
- **Extensive hardware support.** NVIDIA GB300, GB200, B300, B200, H200, H100, and A100, and
  AMD MI300X, MI325, MI350, and MI355X via ROCm. See
  [Installation](https://miles.radixark.com/docs/getting-started/installation#hardware-requirements)
  for per-GPU status and the container image for each.
- **Wide recipe support.** GRPO, GSPO, PPO, and REINFORCE++ for RL, plus SFT and
  [on-policy distillation](https://miles.radixark.com/docs/advanced/on-policy-distillation).
- **Agentic environments.** Train coding and computer-use agents through connectors for
  Harbor, HUD, NeMo Gym, OpenEnv, Verifiers, and more, each plugging into the rollout
  layer that fits it, with task sandboxes on AgentENV, Daytona, E2B, or Modal. See
  [Agentic Environments](https://miles.radixark.com/docs/user-guide/environments).
- **Diffusion models.** Flow-GRPO, DiffusionNFT and SFT on an sglang-diffusion rollout
  engine and an FSDP2 trainer, in
  [Miles-diffusion](https://github.com/radixark/miles_diffusion).

## Getting Started

- [Install Miles](https://miles.radixark.com/docs/getting-started/installation)
- [Quick Start](https://miles.radixark.com/docs/getting-started/quick-start)
- [Supported Models](https://miles.radixark.com/docs/models)
- [Core Concepts](https://miles.radixark.com/docs/user-guide/concepts)
- [Launch Script Walkthrough](https://miles.radixark.com/docs/user-guide/launch-script)
- [Training Backends](https://miles.radixark.com/docs/user-guide/training-backend)
- [Contribution Guide](https://miles.radixark.com/docs/developer/contributor-guide)

## Acknowledgment

Miles was forked from [slime](https://github.com/THUDM/slime), and integrates
[SGLang](https://github.com/sgl-project/sglang),
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM), and
[torch_memory_saver](https://github.com/fzyzcjy/torch_memory_saver).

Miles is shaped by the teams that build on it and support its development,
from hardware and cloud to model labs, agent infrastructure, and academia:

<div align="center">

<img src="https://raw.githubusercontent.com/radixark/miles/main/docs/assets/images/acknowledgment.png" alt="Organizations building on, contributing to, and collaborating with Miles" width="900">

</div>

## Citation

If Miles is useful in your research or your product, please cite it:

```bibtex
@misc{miles2026,
  title        = {Miles: Enterprise-Grade Reinforcement Learning for Large-Scale Model Post-Training},
  author       = {Miles Team},
  year         = {2026},
  howpublished = {\url{https://github.com/radixark/miles}}
}
```
