# Examples

These examples are runnable starting points for your own RL workflow. A few are purely demonstrative, but most are verifiable against a concrete performance score.

## Recipes

End-to-end training workflows — the place to start.

- **[geo3k_vlm](./geo3k_vlm)**: Training VLMs with FSDP using GRPO on the GEO3K dataset.
  - **[multi_turn](./geo3k_vlm/multi_turn)**: The same dataset over multiple turns, with the model cropping images through an interactive environment.
- **[lora](./lora)**: LoRA fine-tuning with the Megatron backend.
- **[multi_lora](./multi_lora)**: Fully-async multi-adapter LoRA training with a slot-keyed adapter page table.
- **[on_policy_distillation](./on_policy_distillation)**: Teacher–student distillation on the student's own rollouts, run inside the on-policy training loop.
  - **[qwen3_5_35b_selfdistill](./on_policy_distillation/qwen3_5_35b_selfdistill)**: Two-phase self-distillation of Qwen3.5-35B-A3B on one 8xH200 node, with an in-process Megatron teacher.
- **[ppo](./ppo)**: Actor-critic PPO with GAE advantages, where the critic shares the actor's train GPUs.
- **[retool_v2](./retool_v2)**: Tool-enabled language model generation with sandboxed Python code execution interleaved with thinking.
- **[swe-agent-harbor-docker](./swe-agent-harbor-docker)**: Trains coding and terminal agents with Harbor-managed local Docker sandboxes and verifier rewards.

## [Infra Features](./infra_features)

Runtime and infrastructure plumbing rather than training recipes — how miles moves
data and weights around.

- **[fully_async](./infra_features/fully_async)**: Demonstrates fully asynchronous rollout generation for higher efficiency.
- **[low_precision](./infra_features/low_precision)**: Examples of FP8 training and inference, plus INT4 QAT, for improved throughput and stability.
- **[p2p_weight_transfer](./infra_features/p2p_weight_transfer)**: Point-to-point weight transfer between training and rollout engines.
- **[random_async](./infra_features/random_async)**: Dataset-free stress test of the async rollout ↔ trainer loop.
- **[train_infer_mismatch_helper](./infra_features/train_infer_mismatch_helper)**: Algorithmic methods for rollout correction (e.g., TIS, MIS).
- **[true_on_policy](./infra_features/true_on_policy)**: Ensures strictly equal log probabilities between inference (SGLang) and training engines.

## [Experimental](./experimental)

Not fully verified — for experimental and development use.

- **[agentenv](./experimental/agentenv)**: Rollouts against AgentENV, a self-hosted platform running agent sandboxes on Firecracker microVMs.
- **[DrGRPO](./experimental/DrGRPO)**: Custom reducer for Dr.GRPO algorithm.
- **[eval](./experimental/eval)**: Documentation and setup for evaluation environments using NeMo-Skills.
- **[eval_multi_task](./experimental/eval_multi_task)**: Example for supporting OOD evaluation tasks, e.g., GPQA, IFBench.
- **[formal_math](./experimental/formal_math)**: Examples related to formal math reasoning tasks, including a single round demo.
- **[multi_agent](./experimental/multi_agent)**: Example of running multi-agent RL with `miles`.
- **[nemo-gym](./experimental/nemo-gym)**: SWE-agent training with NVIDIA NeMo Gym as the environment ecosystem.
- **[openenv](./experimental/openenv)**: Rollouts against OpenEnv-hosted environments.
- **[reproducibility](./experimental/reproducibility)**: Guides on achieving bitwise experiment reproduction using deterministic modes.
- **[search-r1](./experimental/search-r1)**: A minimal reproduction of Search-R1, featuring multi-turn conversation and tool-calling.
- **[strands_sglang](./experimental/strands_sglang)**: Integration example with the Strands-Agents scaffolding framework.
- **[swe-agent-harbor-daytona](./experimental/swe-agent-harbor-daytona)**: The `swe-agent-harbor-docker` pipeline with task sandboxes hosted on Daytona instead of local Docker.
- **[tau-bench](./experimental/tau-bench)**: Training in an agentic multi-turn tool use environment (Tau-bench).
- **[verifiers](./experimental/verifiers)**: Training on a Prime Intellect Verifiers environment instead of a Miles prompt dataset.

<!-- docs:exclude:start -->
## These READMEs are the documentation site

Every README outside `experimental/` is mirrored onto
[miles.radixark.com/docs/examples](https://miles.radixark.com/docs/examples) by
`scripts/tools/sync_example_docs.py`, which pre-commit runs for you. The docs site is
generated from this directory and never edited directly, so a new example needs nothing
beyond its README and an entry in the list above — the sync fails if either is missing.
Three things that list controls: the
level-1 heading of each README becomes the page title, the one-line description becomes
the page's meta description (keep it under 160 characters), and the bullet order is the
sidebar order. Content between
`docs:exclude:start` / `docs:exclude:end` HTML comments (like this section) stays on
GitHub but is left out of the site.
<!-- docs:exclude:end -->
