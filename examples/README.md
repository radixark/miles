# Examples

These examples provide concrete examples to leverage Miles in your own RL workflow. Some examples are just demonstrative, but most of them are verifiable with a concrete performance score.

## Directory Structure

| Example | Description | W&B |
| :--- | :--- | :--- |
| **[DrGRPO](./DrGRPO)** | Custom reducer for Dr.GRPO algorithm. | |
| **[eval](./eval)** | Documentation and setup for evaluation environments using NeMo-Skills. | [link](https://wandb.ai/zijie_xia-n-a/miles-eval) |
| **[eval_multi_task](./eval_multi_task)** | Example for supporting OOD evaluation tasks, e.g., GPQA, IFBench. | [link](https://wandb.ai/zijie_xia-n-a/miles-eval-multi-task) |
| **[formal_math](./formal_math)** | Examples related to formal math reasoning tasks, including a single round demo. | [link](https://wandb.ai/zijie_xia-n-a/miles-formal-math-run-minimal) |
| **[fully_async](./fully_async)** | Demonstrates fully asynchronous rollout generation for higher efficiency. | [link](https://wandb.ai/zijie_xia-n-a/miles-fully-async) |
| **[geo3k_vlm](./geo3k_vlm)** | Training VLMs with FSDP on a single-turn reasoning task using GRPO on the GEO3K dataset. | [link](https://wandb.ai/zijie_xia-n-a/miles-geo3k-vlm) |
| **[geo3k_vlm_multi_turn](./geo3k_vlm_multi_turn)** | VLM multi-turn training (FSDP backend) on Geo3k dataset. | [link](https://wandb.ai/zijie_xia-n-a/miles-geo3k-vlm-multi-turn) |
| **[low_precision](./low_precision)** | Examples of FP8 training and inference for improved throughput and stability. | [link](https://wandb.ai/zijie_xia-n-a/miles-low-precision) |
| **[multi_agent](./multi_agent)** | Example of running multi-agent RL with miles. | [link](https://wandb.ai/zijie_xia-n-a/miles-multi-agent) |
| **[on_policy_distillation](./on_policy_distillation)** | Example implementation for on-policy distillation, extending the reinforcement learning pipeline to support teacher–student distillation directly within on-policy training. | [link](https://wandb.ai/zijie_xia-n-a/miles-on-policy-distillation) |
| **[reproducibility](./reproducibility)** | Guides on achieving bitwise experiment reproduction using deterministic modes. | [link](https://wandb.ai/zijie_xia-n-a/miles-reproducibility) |
| **[retool](./retool)** | Demonstrates the retool functionality for tool-enabled language model generation. | [link](https://wandb.ai/zijie_xia-n-a/miles-retool) |
| **[search-r1](./search-r1)** | A minimal reproduction of Search-R1, featuring multi-turn conversation and tool-calling. | [link](https://wandb.ai/zijie_xia-n-a/miles-search-r1) |
| **[strands-agents](./strands-agents)** | Integration example with the Strands-Agents scaffolding framework. | [link](https://wandb.ai/zijie_xia-n-a/miles-strands-agents) |
| **[train_infer_mismatch_helper](./train_infer_mismatch_helper)** | Algorithmic methods for rollout correction (e.g., TIS, MIS). | [link](https://wandb.ai/zijie_xia-n-a/miles-train-infer-mismatch) |
| **[true_on_policy](./true_on_policy)** | Ensures strictly equal log probabilities between inference (SGLang) and training engines. | [link](https://wandb.ai/zijie_xia-n-a/miles-true-on-policy) |
| **[true_on_policy_vlm](./true_on_policy_vlm)** | "True On-Policy" training demonstration for VLM (Qwen3-VL). | |
