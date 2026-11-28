<div align="center">

<img src="https://raw.githubusercontent.com/radixark/miles/main/docs/assets/images/brand/miles_logo.png" alt="Miles Logo" width="340">

### **面向超大规模模型后训练的企业级强化学习框架**

[![Website](https://img.shields.io/badge/website-miles.radixark.com-d55816)](https://miles.radixark.com/)
[![GitHub Repo](https://img.shields.io/badge/github-radixark%2Fmiles-black?logo=github)](https://github.com/radixark/miles)
[![Docs](https://img.shields.io/badge/docs-miles.radixark.com%2Fdocs-d55816)](https://miles.radixark.com/docs)
[![License](https://img.shields.io/github/license/radixark/miles)](LICENSE)
[![Slack](https://img.shields.io/badge/slack-%23miles--rl-brightgreen.svg)](https://slack.sglang.ai)

| [**官方网站**](https://miles.radixark.com/) | [**官方文档**](https://miles.radixark.com/docs) | [**快速上手**](https://miles.radixark.com/docs/getting-started/quick-start) | [**支持模型**](https://miles.radixark.com/docs/models) | [**Miles 扩散模型版**](https://github.com/radixark/miles_diffusion) | [**技术博客**](https://www.lmsys.org/blog?filter=miles) | [**Slack 社区**](https://slack.sglang.ai) (`#miles-rl`) |

<br />

[English](./README.md) | **简体中文**

</div>

--------------------------------------------------------------------------------

## 最新动态 (News)

- [2026/09] 🔥 SGLang 与 Miles 首日（Day-0）支持 DeepSeek-V4.1（[博客](https://www.lmsys.org/blog/2026-09-10-deepseek-v41)）。
- [2026/09] 🔥 Miles v0.1 技术报告正式发布：[Miles v0.1: Production-Level Post-Training](https://arxiv.org/abs/2609.08368)。
- [2026/09] 使用 Miles 进行后训练，理解并生成多模态物理世界（[博客](https://miles.radixark.com/blogs/post-training-with-miles-to-understand-and-generate-the-multimodal-world)）。
- [2026/08] 🔥 Miles v0.1 正式发布！阅读官方博客：[Announcing Miles v0.1](https://www.radixark.com/blog/announcing-miles-v0-1) 与 [Miles v0.1: Production-level Post-training](https://www.lmsys.org/blog/2026-08-18-miles-v0-1)。
- [2026/08] SGLang 与 Miles 首日支持 Qwen3.8（[博客](https://www.lmsys.org/blog/2026-08-12-qwen3-8-day0-support)）。
- [2026/07] 迈向 Blackwell 原生 8 位与 4 位强化学习：Miles 实现端到端 MXFP8 与 NVFP4 强化学习（[博客](https://www.lmsys.org/blog/2026-07-29-mxfp8-nvfp4-rl)）。
- [2026/07] 🔥 SGLang 与 Miles 首日支持 Kimi K3（[博客](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support)）。
- [2026/07] 在策略蒸馏（On-policy Distillation）正式合入 Miles（[博客](https://www.lmsys.org/blog/2026-07-18-opd-support-in-miles)）。
- [2026/07] 🔥 SGLang 与 Miles 首日支持前沿多模态大模型 Inkling（[博客](https://www.lmsys.org/blog/2026-07-15-inkling-day0-support)）。
- [2026/07] 依托 Miles，AMD Instinct MI355X 现已支持 DeepSeek-V4 Flash 强化学习训练（[博客](https://www.lmsys.org/blog/2026-07-10-rocm-miles-dsv4)）。
- [2026/06] Miles：专为超大规模 LLM 强化学习后训练打造的 PyTorch 原生技术栈（[PyTorch 官方博客](https://pytorch.org/blog/miles-a-pytorch-native-stack-for-large-scale-llm-rl-post-training/)）。
- [2026/06] SGLang 与 Miles 首日支持 NVIDIA Nemotron 3 Ultra（[博客](https://www.lmsys.org/blog/2026-06-04-nvidia-run-nemotron-3-ultra)）。
- [2026/05] 一个 Token 都不能少：Miles 中的 Token-in-Token-out 零损耗机制（[博客](https://www.lmsys.org/blog/2026-05-13-no-token-left-behind)）。
- [2026/04] 秒级完成 1 万亿参数权重更新：大规模分布式强化学习中的 P2P 权重传输（[博客](https://www.lmsys.org/blog/2026-04-29-p2p-update)）。
- [2026/04] 🔥 DeepSeek-V4 首日支持：借助 SGLang 与 Miles 实现从极速推理到验证强化学习（[博客](https://www.lmsys.org/blog/2026-04-25-deepseek-v4)）。

## 项目简介 (About)

Miles 是一个专为**大规模模型后训练**打造的高性能、企业级强化学习框架。它深度协同 [SGLang](https://github.com/sgl-project/sglang) 提供高吞吐的经验采样生成（Rollout），结合 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 实现高可扩展的模型训练，并在万亿参数规模下提供强化学习所需的极致数值精度、系统稳定性与全链路可观测性。框架同时支持 FSDP2 与 [torchtitan](https://github.com/pytorch/torchtitan) 训练后端，详见[训练后端指南](https://miles.radixark.com/docs/user-guide/training-backend)。

> *"千里之行，始于足下（A journey of a thousand miles begins with a single rollout）。"*

### 极致性能 (Performance)

- **完全异步强化学习 (Fully async RL)**：将 Rollout 采样与训练 Worker 彻底解耦，提供可灵活配置的在策略（On-policy）与离策略（Off-policy）调度策略、专门优化以减少空泡的流水线，以及可定制的异步采样与评估模式。详见[完全异步 RL 指南](https://miles.radixark.com/docs/user-guide/fully-async)。
- **高速智能体采样 (Fast agentic rollout)**：采样生成运行在 [SGLang](https://github.com/sgl-project/sglang) 之上，前端配备智能路由器负责在多引擎间均匀分发请求、透传保留单请求元数据，并对集群节点进行健康检查。针对长程多轮 Agent 工作负载深度调优。
- **极速权重更新 (Fast weight updates)**：新权重在训练循环中数秒内即可同步至推理引擎，即使在像 Kimi-K2.6 这样的万亿参数模型上亦能如此；在分离式部署中提供 [P2P RDMA](https://miles.radixark.com/docs/advanced/p2p-weight-transfer) 作为极速直连通道。
- **低精度量化训练 (Low-precision training)**：支持 [MXFP8 与 NVFP4](https://miles.radixark.com/docs/advanced/low-precision) 训练，配合数值稳定的强化学习配方，显著降低由低精度导致的训练发散风险。同时全面支持 FP8、[INT4 QAT](https://miles.radixark.com/docs/advanced/int4-qat)、BF16 与 FP16。
- **LoRA 与多 LoRA 训练 (LoRA and multi-LoRA)**：支持通过[低秩自适应（LoRA）](https://miles.radixark.com/docs/advanced/lora)以极少量的 GPU 算力微调前沿规模大模型，生成的 Adapter 可直接无缝加载至 SGLang 中用于 Rollout 采样。

### 正确性与系统韧性 (Correctness and resilience)

- **Token 进出直通 (Token-in-token-out / TITO)**：面向[所有模型与黑盒测试评估环境](https://miles.radixark.com/docs/user-guide/agentic-rollout)原生支持，彻底消除 Rollout 采样与训练阶段之间 Detokenize（反分词）/ Retokenize（重新分词）来回转换的语义丢失与开销。
- **Rollout 路由重放 (Rollout Routing Replay / R3)**：Rollout 采样期间记录的专家路由路径会在[训练的前向传播中完全重放](https://miles.radixark.com/docs/advanced/miles-router)，彻底解决因 MoE 动态路由不匹配而导致的大规模训练不稳定的问题，并充分重叠计算与通信以压降系统开销。
- **高容错自愈 (Fault tolerance)**：当某个 SGLang 推理引擎异常崩溃退出时，Miles 能够[原地将其自愈恢复并无缝继续当前训练运行](https://miles.radixark.com/docs/advanced/fault-tolerance)：无需集群重启，无需全局暂停。

### Miles 支持与运行的场景 (What Miles runs)

- **首日（Day-0）前沿模型即刻支持**：DeepSeek-V4.1、DeepSeek-V4、Kimi-K3、Qwen3.8、GLM-5.3、GLM-5.2、MiMo-V2.6-Flash、Inkling 与 Nemotron 均在发布当天完成首日适配。除首日模型外，几乎所有业界前沿大模型均可在 Miles 上稳定运行，涵盖 Kimi-K2.6 与 Qwen3.5。详见[支持模型列表](https://miles.radixark.com/docs/models)。
- **广泛的异构硬件生态**：全面支持 NVIDIA GB300、GB200、B300、B200、H200、H100 与 A100，以及 AMD MI355X、MI350X、MI325X 与 MI300X。各 GPU 适配状态请参见[安装指南](https://miles.radixark.com/docs/getting-started/installation#hardware-requirements)，ROCm 容器镜像请参见 [AMD ROCm 指南](https://miles.radixark.com/docs/hardware-platforms/amd-gpus)。
- **丰富的强化学习算法配方**：全面支持 GRPO、GSPO、PPO 与 REINFORCE++ 等强化学习算法，并支持[得分中心化（Score Centering）](https://miles.radixark.com/docs/examples/infra-features/score-centering)、SFT 监督微调以及[在策略蒸馏（On-policy Distillation）](https://miles.radixark.com/docs/advanced/on-policy-distillation)。
- **Ray 或 Kubernetes 云原生部署**：相同的启动脚本既可在 Ray 集群上直接执行，亦可作为 Helm Release 安装，由 Kubernetes 统筹调度整个训练任务中的每个 Worker。详见 [Ray 与 Kubernetes 集群后端](https://miles.radixark.com/docs/advanced/cluster-backend)。
- **丰富的智能体（Agent）沙盒环境**：通过 Harbor、HUD、NeMo Gym、OpenEnv、Verifiers 等连接器训练代码生成与电脑操作（Computer-Use）智能体，各连接器适配最贴合的采样层，并支持接入 AgentENV、Daytona、E2B 或 Modal 等任务沙盒。详见[智能体环境指南](https://miles.radixark.com/docs/user-guide/environments)。
- **扩散模型支持 (Diffusion models)**：通过 [Miles-diffusion](https://github.com/radixark/miles_diffusion) 仓库，支持在 sglang-diffusion 采样引擎与 FSDP2 训练器上执行 Flow-GRPO、DiffusionNFT 以及 SFT 训练。

## 快速上手 (Getting Started)

- [安装 Miles](https://miles.radixark.com/docs/getting-started/installation)
- [快速上手教程](https://miles.radixark.com/docs/getting-started/quick-start)
- [支持模型矩阵](https://miles.radixark.com/docs/models)
- [核心概念解析](https://miles.radixark.com/docs/user-guide/concepts)
- [启动脚本完全指南](https://miles.radixark.com/docs/user-guide/launch-script)
- [训练后端详解](https://miles.radixark.com/docs/user-guide/training-backend)
- [开发者贡献指引](https://miles.radixark.com/docs/developer/contributor-guide)

## 致谢 (Acknowledgment)

Miles 项目 Fork 自 [slime](https://github.com/THUDM/slime)，并深度集成了 [SGLang](https://github.com/sgl-project/sglang)、[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 以及 [torch_memory_saver](https://github.com/fzyzcjy/torch_memory_saver)。

Miles 的蓬勃发展离不开在其之上构建应用与支持其研发的各大团队，涵盖硬件厂商、云服务商、前沿模型实验室、智能体基础设施提供商与学术科研机构：

<div align="center">

<img src="https://raw.githubusercontent.com/radixark/miles/main/docs/assets/images/acknowledgment.png" alt="使用、贡献并与 Miles 协同合作的企业与机构" width="900">

</div>

## 学术引用 (Citation)

如果 Miles 在你的学术研究或生产产品中发挥了作用，欢迎引用本项目：

技术报告：

```bibtex
@misc{radixark2026milesv01productionlevelposttraining,
  title         = {Miles v0.1: Production-Level Post-Training},
  author        = {RadixArk},
  year          = {2026},
  eprint        = {2609.08368},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2609.08368}
}
```

开源代码仓库：

```bibtex
@misc{miles2026,
  title        = {Miles: Enterprise-Grade Reinforcement Learning for Large-Scale Model Post-Training},
  author       = {Miles Team},
  year         = {2026},
  howpublished = {\url{https://github.com/radixark/miles}}
}
```

---

> **免责声明**：本简体中文文档由社区志愿者出于学习交流目的进行翻译维护。英文原版文档始终作为项目最权威的单一事实来源（Single Source of Truth）；如中文翻译与英文原版存在任何语义或版本脱节，请以 [英文原版 README](https://github.com/radixark/miles/blob/main/README.md) 为准。
