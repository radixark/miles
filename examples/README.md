# Examples

These examples provide concrete examples to leverage miles in your own RL workflow. Some examples are just demonstrative, but most of them are verifiable with a concrete performance score.

## Directory Structure

Training recipes live at the top level. Two subdirectories group everything else:
`infra_features/` for runtime and infrastructure plumbing, and `experimental/` for
recipes that are not fully verified.

### Recipes

- **[fully_async](./fully_async)**: Demonstrates fully asynchronous rollout generation for higher efficiency.
- **[geo3k_vlm](./geo3k_vlm)**: Training VLMs with FSDP using GRPO on the GEO3K dataset, single-turn and [multi-turn](./geo3k_vlm/multi_turn).
- **[lora](./lora)**: LoRA fine-tuning with the Megatron backend.
- **[multi_lora](./multi_lora)**: Fully-async multi-adapter LoRA training with a slot-keyed adapter page table.
- **[on_policy_distillation](./on_policy_distillation)**: Example implementation for on-policy distillation, extending the reinforcement learning pipeline to support teacher–student distillation directly within on-policy training.
- **[retool_v2](./retool_v2)**: Tool-enabled language model generation with sandboxed Python code execution interleaved with thinking.
- **[swe-agent](./swe-agent)**: Trains coding and terminal agents with Harbor-managed sandboxes and verifier rewards.

### [infra_features](./infra_features)

- **[low_precision](./infra_features/low_precision)**: Examples of FP8 training and inference, plus INT4 QAT, for improved throughput and stability.
- **[p2p_weight_transfer](./infra_features/p2p_weight_transfer)**: Point-to-point weight transfer between training and rollout engines.
- **[random_async](./infra_features/random_async)**: Dataset-free stress test of the async rollout ↔ trainer loop.
- **[train_infer_mismatch_helper](./infra_features/train_infer_mismatch_helper)**: Algorithmic methods for rollout correction (e.g., TIS, MIS).
- **[true_on_policy](./infra_features/true_on_policy)**: Ensures strictly equal log probabilities between inference (SGLang) and training engines.

### [experimental](./experimental)

Not fully verified — for experimental and development use.

- **[DrGRPO](./experimental/DrGRPO)**: Custom reducer for Dr.GRPO algorithm.
- **[eval](./experimental/eval)**: Documentation and setup for evaluation environments using NeMo-Skills.
- **[eval_multi_task](./experimental/eval_multi_task)**: Example for supporting OOD evaluation tasks, e.g., GPQA, IFBench.
- **[formal_math](./experimental/formal_math)**: Examples related to formal math reasoning tasks, including a single round demo.
- **[multi_agent](./experimental/multi_agent)**: Example of running multi-agent RL with `miles`.
- **[openenv](./experimental/openenv)**: Rollouts against OpenEnv-hosted environments.
- **[reproducibility](./experimental/reproducibility)**: Guides on achieving bitwise experiment reproduction using deterministic modes.
- **[search-r1](./experimental/search-r1)**: A minimal reproduction of Search-R1, featuring multi-turn conversation and tool-calling.
- **[strands_sglang](./experimental/strands_sglang)**: Integration example with the Strands-Agents scaffolding framework.
- **[tau-bench](./experimental/tau-bench)**: Training in an agentic multi-turn tool use environment (Tau-bench).
