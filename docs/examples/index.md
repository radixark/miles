---
title: "Examples"
sidebarTitle: "Overview"
description: "These examples are runnable starting points for your own RL workflow."
# Generated from examples/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
A few are purely demonstrative, but most are verifiable against a concrete performance score.

## Recipes

End-to-end training workflows — the place to start.

- **[geo3k_vlm](/examples/geo3k-vlm)**: Training VLMs with FSDP using GRPO on the GEO3K dataset.
  - **[multi_turn](/examples/geo3k-vlm/multi-turn)**: The same dataset over multiple turns, with the model cropping images through an interactive environment.
- **[lora](https://github.com/radixark/miles/tree/main/examples/lora)**: LoRA fine-tuning with the Megatron backend.
- **[multi_lora](/examples/multi-lora)**: Fully-async multi-adapter LoRA training with a slot-keyed adapter page table.
- **[multi_policy](/examples/multi-policy)**: Two policies in one run — a solver answering gsm8k, and a verifier scored on ruling correctly about the solver's answers.
- **[on_policy_distillation](/examples/on-policy-distillation)**: Teacher–student distillation on the student's own rollouts, run inside the on-policy training loop.
  - **[qwen3_5_35b_selfdistill](/examples/on-policy-distillation/qwen3-5-35b-selfdistill)**: Two-phase self-distillation of Qwen3.5-35B-A3B on one 8xH200 node, with an in-process Megatron teacher.
- **[ppo](/examples/ppo)**: Actor-critic PPO with GAE advantages, where the critic shares the actor's train GPUs.
- **[retool_v2](/examples/retool-v2)**: Tool-enabled language model generation with sandboxed Python code execution interleaved with thinking.
- **[swe-agent-harbor-docker](/examples/swe-agent-harbor-docker)**: Trains coding and terminal agents with Harbor-managed local Docker sandboxes and verifier rewards.

## [Infra Features](/examples/infra-features)

Runtime and infrastructure plumbing rather than training recipes — how miles moves
data and weights around.

- **[fully_async](/examples/infra-features/fully-async)**: Demonstrates fully asynchronous rollout generation for higher efficiency.
- **[hot_restart](/examples/infra-features/hot-restart)**: Replaces the orchestration script and rollout executor of a live run, keeping trainers and engines up.
- **[low_precision](/examples/infra-features/low-precision)**: Examples of FP8 training and inference, plus INT4 QAT, for improved throughput and stability.
- **[p2p_weight_transfer](/examples/infra-features/p2p-weight-transfer)**: Point-to-point weight transfer between training and rollout engines.
- **[random_async](/examples/infra-features/random-async)**: Dataset-free stress test of the async rollout ↔ trainer loop.
- **[split_deployment](/examples/infra-features/split-deployment)**: Installs one run as several helm releases — trainer, engines and orchestration script apart.
- **[train_infer_mismatch_helper](/examples/infra-features/train-infer-mismatch-helper)**: Algorithmic methods for rollout correction (e.g., TIS, MIS).
- **[true_on_policy](/examples/infra-features/true-on-policy)**: Ensures strictly equal log probabilities between inference (SGLang) and training engines.

## [Experimental](https://github.com/radixark/miles/tree/main/examples/experimental)

Not fully verified — for experimental and development use.

- **[agentenv](https://github.com/radixark/miles/tree/main/examples/experimental/agentenv)**: Rollouts against AgentENV, a self-hosted platform running agent sandboxes on Firecracker microVMs.
- **[DrGRPO](https://github.com/radixark/miles/tree/main/examples/experimental/DrGRPO)**: Custom reducer for Dr.GRPO algorithm.
- **[eval](https://github.com/radixark/miles/tree/main/examples/experimental/eval)**: Documentation and setup for evaluation environments using NeMo-Skills.
- **[eval_multi_task](https://github.com/radixark/miles/tree/main/examples/experimental/eval_multi_task)**: Example for supporting OOD evaluation tasks, e.g., GPQA, IFBench.
- **[formal_math](https://github.com/radixark/miles/tree/main/examples/experimental/formal_math)**: Examples related to formal math reasoning tasks, including a single round demo.
- **[multi_agent](https://github.com/radixark/miles/tree/main/examples/experimental/multi_agent)**: Example of running multi-agent RL with `miles`.
- **[nemo-gym](https://github.com/radixark/miles/tree/main/examples/experimental/nemo-gym)**: SWE-agent training with NVIDIA NeMo Gym as the environment ecosystem.
- **[openenv](https://github.com/radixark/miles/tree/main/examples/experimental/openenv)**: Rollouts against OpenEnv-hosted environments.
- **[reproducibility](https://github.com/radixark/miles/tree/main/examples/experimental/reproducibility)**: Guides on achieving bitwise experiment reproduction using deterministic modes.
- **[search-r1](https://github.com/radixark/miles/tree/main/examples/experimental/search-r1)**: A minimal reproduction of Search-R1, featuring multi-turn conversation and tool-calling.
- **[strands_sglang](https://github.com/radixark/miles/tree/main/examples/experimental/strands_sglang)**: Integration example with the Strands-Agents scaffolding framework.
- **[swe-agent-harbor-daytona](https://github.com/radixark/miles/tree/main/examples/experimental/swe-agent-harbor-daytona)**: The `swe-agent-harbor-docker` pipeline with task sandboxes hosted on Daytona instead of local Docker.
- **[tau-bench](https://github.com/radixark/miles/tree/main/examples/experimental/tau-bench)**: Training in an agentic multi-turn tool use environment (Tau-bench).
- **[verifiers](https://github.com/radixark/miles/tree/main/examples/experimental/verifiers)**: Training on a Prime Intellect Verifiers environment instead of a Miles prompt dataset.
