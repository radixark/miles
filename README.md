<div align="center">

<img src="https://raw.githubusercontent.com/radixark/miles/main/docs/assets/images/brand/miles_logo.png" alt="Miles Logo" width="550">

### **Enterprise-Grade Reinforcement Learning for Large-Scale Model Post-Training**

[![GitHub Repo](https://img.shields.io/badge/github-radixark%2Fmiles-black?logo=github)](https://github.com/radixark/miles)
[![Docs](https://img.shields.io/badge/docs-miles.radixark.com-d55816)](https://miles.radixark.com/docs)
[![License](https://img.shields.io/github/license/radixark/miles)](LICENSE)
[![Slack](https://img.shields.io/badge/slack-join-brightgreen.svg)](https://slack.sglang.ai)

</div>

--------------------------------------------------------------------------------

| [**Documentation**](https://miles.radixark.com/docs) | [**Quick Start**](https://miles.radixark.com/docs/getting-started/quick-start) | [**Models**](https://miles.radixark.com/docs/models) | [**Blog**](https://www.lmsys.org/blog) | [**Slack**](https://slack.sglang.ai) |

## News

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
scalable training, and ships the precision, stability and observability features an RL run
needs at trillion-parameter scale.

> *"A journey of a thousand miles begins with a single rollout."*

The core features include:

- **Fast rollout and weight updates**: fully async RL with configurable on- and off-policy
  schedules, high-throughput agentic generation on SGLang, and in-loop weight sync that
  moves 1 T parameters in under 10 seconds, with
  [P2P RDMA](https://miles.radixark.com/docs/advanced/p2p-weight-transfer) as the fast path
  for disaggregated setups.
- **Correctness at scale**: [token-in-token-out](https://miles.radixark.com/docs/user-guide/agentic-chat-template)
  for every model and every black-box agent harness,
  [rollout routing replay](https://miles.radixark.com/docs/advanced/miles-router) to remove
  the MoE routing mismatch that destabilizes large runs, and true-on-policy alignment
  between the trainer and the engine.
- **Low-precision training**: end-to-end
  [MXFP8 and NVFP4](https://miles.radixark.com/docs/advanced/low-precision) on Blackwell,
  plus FP8, [INT4 QAT](https://miles.radixark.com/docs/advanced/int4-qat), BF16 and FP16.
- **Broad model support**: day-0 enablement of frontier releases including DeepSeek-V4,
  Kimi-K3, GLM-5.2, Qwen3.6, Inkling, Nemotron 3 and Gemma 4, covering dense, MoE, hybrid
  attention and multimodal architectures. See
  [Models](https://miles.radixark.com/docs/models).
- **Extensive hardware support**: NVIDIA GB300, GB200, B300, B200, H200, H100 and A100, and
  AMD MI300X, MI325, MI350 and MI355X via ROCm.
- **Recipes and environments**: RL (GRPO, GSPO, PPO), SFT,
  [on-policy distillation](https://miles.radixark.com/docs/advanced/on-policy-distillation)
  and [LoRA](https://miles.radixark.com/docs/advanced/lora), with
  [Harbor, OpenEnv and NeMo Gym](https://miles.radixark.com/docs/user-guide/environments)
  integrations for coding-agent sandboxes.
- **Built to keep running**: [fault tolerance](https://miles.radixark.com/docs/advanced/fault-tolerance)
  that recovers a dead SGLang engine in place, twenty-plus
  [plug-points](https://miles.radixark.com/docs/user-guide/customization) for custom Python,
  and a [dashboard](https://miles.radixark.com/docs/user-guide/dashboard) showing what every
  GPU did during a step and what every trajectory contained at the token level.

## Getting Started

- [Install Miles](https://miles.radixark.com/docs/getting-started/installation)
- [Quick Start](https://miles.radixark.com/docs/getting-started/quick-start)
- [Core Concepts](https://miles.radixark.com/docs/user-guide/concepts)
- [Launch Script Walkthrough](https://miles.radixark.com/docs/user-guide/launch-script)
- [Training Backends](https://miles.radixark.com/docs/user-guide/training-backend)
- [Contribution Guide](https://miles.radixark.com/docs/developer/contributor-guide)

Docker is the recommended way in, since Miles pins patched builds of SGLang, Megatron-LM
and a few CUDA kernels:

```bash
docker pull radixark/miles:latest
docker run --rm --gpus all --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 --network=host \
  -it radixark/miles:latest /bin/bash
```

From there the [Quick Start](https://miles.radixark.com/docs/getting-started/quick-start)
takes one node of 8 GPUs from a fresh container to a GRPO run with the reward climbing, and
ends on a single command:

```bash
python scripts/run_qwen3_dense.py --model-name Qwen3-4B
```

## Contact Us

For enterprise adoption, collaboration or support, reach us at
[miles@radixark.ai](mailto:miles@radixark.ai). For questions and discussion, join the
`#miles` channel on [Slack](https://slack.sglang.ai), or open an
[issue](https://github.com/radixark/miles/issues).

## Acknowledgment

<!-- TODO: acknowledgment figure -->

Miles builds on the work of [slime](https://github.com/THUDM/slime),
[SGLang](https://github.com/sgl-project/sglang),
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM),
[mbridge](https://github.com/ISEEKYAN/mbridge) and
[torch_memory_saver](https://github.com/fzyzcjy/torch_memory_saver).
