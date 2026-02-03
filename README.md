<div align="center">

<img src="https://raw.githubusercontent.com/radixark/miles/main/imgs/miles_logo.png" alt="Miles Logo" width="550">

### **Enterprise-Grade Reinforcement Learning for Large-Scale Model Training**
### **High-Performance Rollout • Low Precision Training • Production Stability**

[![GitHub Repo](https://img.shields.io/badge/github-radixark%2Fmiles-black?logo=github)](https://github.com/radixark/miles)
[![License](https://img.shields.io/github/license/radixark/miles)](LICENSE)
[![Slack](https://img.shields.io/badge/slack-join-brightgreen.svg)](https://slack.sglang.ai)

[**Latest Updates**](#latest-updates) | [**Quick Start**](#quick-start) | [**Key Features**](#key-features) | [**Documentation**](docs/en/get_started/quick_start.md)

</div>

---


## Latest Updates

*   **[2026/01]** 💎 **INT4 Quantization-Aware Training (QAT)**: Inspired by the Kimi K2-Thinking report, Miles now features a full-stack INT4 W4A16 QAT pipeline. This allows 1TB-scale models to fit into single-machine VRAM (e.g., NVIDIA H200), doubling rollout efficiency by eliminating cross-node bottlenecks while maintaining BF16-equivalent accuracy. [Blog](https://lmsys.org/blog/2026-01-26-int4-qat/)
*   **[2026/01]** 💎 **Unified VLM/LLM Multi-Turn Training**: We provided an implementation for the VLM multi-turn sampling paradigm. Developers only need to write a customized `rollout` function to easily start multi-turn RL for VLM, just like training LLM. [Blog](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/vlm-multi-turn/readme-en.md)
*   **[2026/01]** 🤖 **Multi-Agent Co-Evolution**: Miles now supports **MrlX**, a novel asynchronous co-evolutionary framework for Multi-Agent RL. Achieve superior performance in complex tasks like Doctor-Patient simulations and DeepResearch pipelines by enabling specialized agents to evolve together symbiotically. [[Link]](https://github.com/AQ-MedAI/MrlX)
*   **[2025/12]** 🔄 **Rollout Routing Replay (R3)**: In collaboration with SGLang, we've launched R3 to solve MoE RL instability. R3 records inference routing decisions and replays them during training, effectively eliminating the "training-inference mismatch" and preventing training collapse in large MoE models like Qwen3 and DeepSeek-V3. [[Paper]](https://arxiv.org/pdf/2510.11370)
*   **[2025/11]** 🔥 **Unified FP8 Release**: Solves the stability issues in MoE RL by ensuring training and inference use the exact same FP8 quantization logic. [[Blog]](https://lmsys.org/blog/2025-11-25-fp8-rl/)
*   **[2025/11]** ⚡ **Speculative Decoding in RL**: Integrated speculative rollout with online SFT for draft models, achieving massive throughput gains. [[Blog]](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/spec/readme-en.md)
*   **[2025/11]** 🎉 **Miles Project Launch**: A joint effort by InfiXAI, Ant Group, SGLang RL Team, and the Miles community. [[Announcement]](https://lmsys.org/blog/2025-11-19-miles/)

## What is Miles?

**Miles** is a high-performance, enterprise-ready reinforcement learning (RL) framework specifically optimized for **Large-Scale model Post-Training**. Built as a powerful fork of **[slime](https://github.com/THUDM/slime)**, Miles bridges the gap between research-grade RL and production-grade reliability by integrating **SGLang** for high-throughput rollout and **Megatron-LM** for scalable training.

> *"A journey of a thousand miles begins with a single step."* — Miles focuses on the low-level system optimizations that make large-scale RL stable, efficient, and reproducible.

---


## Key Features

### 🌪️ Advanced MoE & Low-Precision Training

*   **Unified FP8 Pipeline**: The first framework to implement end-to-end FP8 sampling and training. By unifying precision across rollout and training, Miles eliminates the quantization-induced discrepancy that causes RL collapse in large MoE models.
*   **Rollout Routing Replay (R3)**: Records expert routing decisions during SGLang inference and replays them during training to ensure bit-wise expert alignment.
*   **INT4 QAT Support**: Recommendation for 1TB+ models to enable single-machine (e.g., H200) deployment by significantly reducing memory footprint.

### 🛡️ Eliminating Train-Inference Mismatch

*   **Bit-wise Identical Training and Inference Log Probs**: System-level solution achieving deterministic forward/backward passes through kernel-level optimization (FlashAttention-3, DeepGEMM).
*   **Algorithmic Correction (TIS/MIS)**: When mismatch is unavoidable, Miles provides **Truncated Importance Sampling (TIS)** and **Masked Importance Sampling (MIS)** to mitigate off-policy bias and prevent training divergence.

### ⚡ Extreme Performance & Efficiency

*   **Speculative RL Training**: Achieve **25%+ rollout speedup** by using an **Online SFT Draft Model**. Unlike frozen draft models, Miles updates the draft policy during RL to prevent policy drift.
*   **Zero-Copy Weight Sync**: Optimized weight refit via **CUDA IPC zero-copy mapping**, async tensor gathering, and bucketed flattening. Sync time reduced by 50% compared to standard HTTP/RPC transfers.
*   **Partial Rollout & Over-Sampling**: Handles the "Long-Tail Effect" in multi-turn RL by over-sampling requests and recycling half-finished trajectories to maximize GPU utilization.

## Model Support & Training Diversity

### 🏗️ Supported Models
Miles supports a wide range of state-of-the-art architectures, with a special emphasis on **DeepSeek, Qwen, Llama** and mainstream models.

| Family | Supported Models |
| :--- | :--- |
| **DeepSeek** | **R1, V3, V3.2** |
| **Qwen** | **Qwen 2, 2.5, 3** |
| **Llama** | **Llama 3, 3.1, 3.3, 4** |
| **Gemma** | **Gemma 2, 3, 3N** |
| **GLM** | **GLM-4.5, GLM-4.6, GLM-4.7** |
| **MiniMax** | **M2, M2.1** |
| **Others** | **Mistral, Mixtral, Phi, gpt-oss and any model supported by SGLang and Megatron** |

### 🧩 Diverse Training Scenarios
Miles is designed to handle the complexity of modern RL workloads across various dimensions:
*   **Multi-Turn Interaction**: Optimized for complex, multi-round conversations and tool-use scenarios.
*   **VLM & LLM Support**: Unified framework for both Vision-Language and pure Text models.
*   **Reasoning & Coding**: Specific recipes and optimizations for **Reasoning (Math/Logic)** and **Coding Agent** tasks.
*   **Multi-Agent Training**: Support for advanced co-training and collaborative multi-agent reinforcement learning.

---

## Quick Start

### Installation

We recommend using our official Docker image for the best performance and compatibility:

```bash
# Pull the latest image
docker pull radixark/miles:latest

# Or install from source
pip install -r requirements.txt
pip install -e .
```

### Launch Training

Miles provides a unified entry point for complex RL tasks. Here is an example of FP8 GRPO training for Qwen3:

```bash
python train.py \
    --advantage-estimator grpo \
    --model-name qwen3-30b-a3b \
    --hf-checkpoint /path/to/qwen3-30b-a3b-hf \
    --rollout-batch-size 512 \
    --n-samples-per-prompt 8
```

For comprehensive guides on environment setup and custom reward functions, see the [Quick Start Guide](docs/en/get_started/quick_start.md).

---

## Roadmap

### ✅ Completed

- [x] **Unified FP8** E2E Training & Rollout
- [x] **INT4 Quantization-Aware Training (QAT)**: Single-machine 1TB models
- [x] **Speculative RL** with Online SFT
- [x] **Multi-Agent RL** (Co-evolutionary frameworks like [MrlX](https://github.com/AQ-MedAI/MrlX))
- [x] **Support DeepSeek V3.2 Models**
- [x] **VLM Multi-Turn Training**
- [x] **Aligning SGLang with Megatron in Dense Models**
- [x] **Rollout Routing Replay (R3)**

### 🏗️ In Progress & Planned

- [ ] **Zero mismatch for MoE RL**
- [ ] **Aligning SGLang with Megatron in MoE Models**
- [ ] **Diffusion RL** Support
- [ ] **Omni RL** Support
- [ ] **Diffusion LLM RL** Support
- [ ] **Elastic Resource Scheduling**: Dynamic scaling of rollout vs. training workers



---

## Acknowledgements

Miles is built upon the shoulders of giants in the LLM infrastructure ecosystem:
*   **[slime](https://github.com/THUDM/slime)**: The core modular architecture and inspiration.
*   **[SGLang](https://github.com/sgl-project/sglang)**: The high-performance inference engine.
*   **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)**: Robust large-scale training components.

Special thanks to **InfiXAI Team**, **Ant Group AQ Team**, **SGLang RL Team**, and the **Miles Team**. We also thank **DataCrunch** for compute sponsorship and **NVIDIA** for technical support on Transformer Engine (TE).

---

## Links

*   **GitHub**: [https://github.com/radixark/miles](https://github.com/radixark/miles)
*   **Slime Project**: [https://github.com/THUDM/slime](https://github.com/THUDM/slime)
*   **Developer Guide**: Check the `docs/` and `examples/` directories for in-depth technical notes.

<div align="center">

**Give Miles a ⭐️ Star if it helps your RL journey!**

</div>
