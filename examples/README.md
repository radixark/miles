# Miles Examples

This directory contains example configurations and scripts demonstrating various Miles capabilities for reinforcement learning post-training.

## Quick Reference

| Example | Description | Status |
|---------|-------------|--------|
| [eval](#eval) | External evaluation framework with NeMo Skills | Verified |
| [eval_multi_task](#eval_multi_task) | Multi-dataset evaluation configuration | - |
| [formal_math](#formal_math) | Formal math verification with Lean | - |
| [fully_async](#fully_async) | Fully asynchronous rollout generation | - |
| [geo3k_vlm](#geo3k_vlm) | Vision language model training on geometry | - |
| [low_precision](#low_precision) | FP8 training and inference | Verified |
| [multi_agent](#multi_agent) | Multi-agent reinforcement learning | - |
| [on_policy_distillation](#on_policy_distillation) | Knowledge distillation from teacher model | - |
| [reproducibility](#reproducibility) | Deterministic/reproducible training | Verified |
| [retool](#retool) | Tool-augmented generation with Python sandbox | - |
| [search-r1](#search-r1) | Multi-turn search and reasoning | - |
| [strands-agents](#strands-agents) | Strands agent framework integration | - |
| [tau-bench](#tau-bench) | Multi-turn task-oriented dialogue | Verified |
| [train_infer_mismatch_helper](#train_infer_mismatch_helper) | TIS/MIS for training-inference mismatch | - |
| [true_on_policy](#true_on_policy) | Bitwise-identical training and inference | - |
| [true_on_policy_vlm](#true_on_policy_vlm) | True on-policy for vision language models | - |

---

## eval

External evaluation framework that integrates with NeMo Skills for math problem evaluation.

**Key Features:**
- Multi-task evaluation support (AIME, GPQA, IFBench, HLE benchmarks)
- Docker-based distributed setup
- Configurable sampling parameters per dataset

**Files:**
- `eval_delegate.py` - Configuration dataclass for evaluation parameters
- `eval_delegate_rollout.py` - Proxy for delegating evaluation to external services
- `nemo_skills/skills_server.py` - HTTP server proxying to NeMo Skills
- `scripts/run-qwen3-4B.sh`, `scripts/run-qwen3-32B.sh` - Launch scripts

---

## eval_multi_task

Configuration and scripts for evaluating models on multiple datasets simultaneously.

**Key Features:**
- Unified configuration for multiple benchmarks
- Per-dataset parameter overrides
- Automatic environment setup for IFBench

**Datasets Supported:** AIME-2024, GPQA-Diamond, IFBench

**Files:**
- `multi_task.yaml` - Multi-dataset configuration
- `multi_task.sh` - Launch script

---

## formal_math

Training on formal math verification using interactive theorem proving with Lean.

**Key Features:**
- Integration with Kimina client for Lean 4
- Data filtering and curriculum learning
- Binary reward based on proof verification
- Supports both SFT and RL phases

**Files:**
- `single_round/run.py` - Main training orchestration
- `single_round/run_minimal.py` - Simplified version without Ray
- `single_round/reward_fn.py` - Custom reward for Lean verification

---

## fully_async

Fully asynchronous rollout generation where a persistent background worker continuously generates samples.

**Key Features:**
- Removes per-step synchronization barriers
- Thread-based persistent worker with asyncio event loop
- Configurable concurrency up to `rollout-batch-size`

**Usage:** Uses `train_async.py` instead of `train.py`

**Files:**
- `fully_async_rollout.py` - Core async worker implementation
- `run-qwen3-4b-fully_async.sh` - Launch script

---

## geo3k_vlm

Fine-tunes Qwen3-VL on geometry problem solving using FSDP.

**Key Features:**
- Multi-modal input handling (images + text)
- GRPO algorithm implementation
- Blackwell GPU support with configurable attention backends

**Models:** Qwen3-VL-2B/4B/8B-Instruct
**Dataset:** GEO3K (geometry problems with images)

**Files:**
- `run_geo3k_vlm.py` - Main training script

---

## low_precision

Demonstrates low-precision training and inference using FP8 format.

**Key Features:**
- FP8 format (`e4m3`) with blockwise scaling
- TransformerEngine integration
- Online quantization during forward/backward passes

**Limitations:**
- FP8 weights conflict with CPU Adam offload
- Requires checkpoint conversion post-training

**Files:**
- `run-qwen3-4b-fp8.sh` - Single-node FP8 training
- `run-qwen3-30b-a3b-fp8-two-nodes.sh` - Multi-node example

---

## multi_agent

Training with multiple agents solving the same problem independently.

**Key Features:**
- Parallel generation with configurable agent count
- Weighted reward adjustment based on correctness
- Random shuffling of solutions before training

**Configuration:**
```python
"num_parallel": 5,           # 5 agents per prompt
"incorrect_reward_weight": 0.8,
"correct_reward_weight": 1.2,
```

**Files:**
- `agent_system.py` - Multi-agent orchestration
- `rollout_with_multi_agents.py` - Generation with multiple agents
- `run-qwen3-30B-A3B-multi-agent.sh` - Launch script

---

## on_policy_distillation

On-policy knowledge distillation from a large teacher model to a smaller student.

**Key Features:**
- Teacher model server on separate GPU
- Asynchronous HTTP-based communication
- Token-level log probability extraction

**Architecture:** Teacher (Qwen3-32B) → Student (Qwen3-8B)

**Files:**
- `on_policy_distillation.py` - Async reward function with teacher log probs
- `run-qwen3-8B-opd.sh` - Training pipeline

---

## reproducibility

Demonstrates bitwise-reproducible training using deterministic modes.

**Key Features:**
- Bitwise identical results across runs
- SGLang deterministic inference + Megatron deterministic training

**Required Flags:**
```bash
--sglang-enable-deterministic-inference
--sglang-attention-backend flashinfer
--deterministic-mode
```

**Metric:** `train/train_rollout_logprob_abs_diff` should be exactly 0

**References:**
- [SGLang Deterministic Inference Blog](https://lmsys.org/blog/2025-09-22-sglang-deterministic/)

**Files:**
- `run-qwen2.5-0.5B-gsm8k.sh` - Training script with determinism flags
- `README.md` - Detailed setup instructions

---

## retool

Tool-augmented generation with full SFT-to-RL pipeline.

**Key Features:**
- Sandboxed Python code execution with memory/time limits
- Concurrent tool execution with semaphore control
- Two-phase training (SFT → RL)

**Tool Format:**
```xml
<tool_call>{"name": "function_name", "arguments": {...}}</tool_call>
```

**References:** [ReTool Paper](https://arxiv.org/abs/2504.11536)

**Files:**
- `generate_with_retool.py` - Multi-turn generation with tool support
- `tool_sandbox.py` - Safe Python execution sandbox
- `retool_qwen3_4b_sft.sh`, `retool_qwen3_4b_rl.sh` - Training scripts

---

## search-r1

Multi-turn conversation with web/retrieval search integration.

**Key Features:**
- Dual search backend support (local dense retrieval / Google)
- Trajectory Importance Sampling (TIS) support
- Local retrieval with FAISS and e5-base-v2 embeddings

**Search Backends:**
- **Local:** Dense retrieval (requires ~132GB disk space)
- **Google:** Serper.dev API-based

**Datasets:** NQ, HotpotQA, TriviaQA, POPQA, 2WikiMultihopQA, MuSiQue, BamBooGLE

**Files:**
- `generate_with_search.py` - Main generation with search
- `local_dense_retriever/` - Local retrieval server setup
- `README.md` - Detailed setup instructions

---

## strands-agents

Integration with the Strands-Agents scaffolding framework.

**Key Features:**
- OpenAI-compatible model wrapper from SGLang server
- Python code execution tool via CAMEL's SubprocessInterpreter
- Max 16 messages per interaction

**References:** [Strands-Agents SDK](https://github.com/strands-agents/sdk-python)

**Files:**
- `generate_with_strands.py` - Strands agent creation and execution
- `strands_qwen3_4b.sh` - Training script

---

## tau-bench

Multi-turn task-oriented dialogue training in interactive environments.

**Key Features:**
- Retail and airline booking environments
- Multiple agent types: tool-calling, act, react, few-shot
- User simulation with Gemini 2.0 Flash

**Configuration:**
```python
"env": "retail",  # or "airline"
"agent": "tool-calling",
"user_model": "gemini-2.0-flash-lite",
```

**Files:**
- `generate_with_tau.py` - Environment interaction
- `trainable_agents.py` - Agent factory
- `run_qwen3_4B.sh` - Training script

---

## train_infer_mismatch_helper

Algorithmic methods for handling training-inference mismatch using importance sampling.

**Algorithms:**
1. **Standard PPO** - Baseline
2. **Bypassing PPO** - Uses rollout logprobs directly
3. **Decoupled PPO** - 3-policy approach

**Importance Sampling Levels:**
- **Token:** Per-token weights (biased, simple)
- **Sequence:** Product of token weights (unbiased, high variance)
- **Geometric:** Geometric mean (low variance)

**References:**
- [Decoupled PPO Paper](https://arxiv.org/pdf/2110.00641)

**Files:**
- `mis.py` - Importance sampling implementation
- `mis.yaml` - Configuration parameters
- `run-qwen3-4b-mis.sh` - Training script

---

## true_on_policy

Bitwise-identical training and inference (true on-policy).

**Key Technologies:**
- Flash Attention 3 for bitwise-identical prefill/decode
- DeepGEMM with consistent tensor core instructions
- SGLang deterministic inference + Megatron deterministic kernels

**Metric:** `train/train_rollout_logprob_abs_diff` should be exactly 0.0

**References:**
- [Implementation PR #566](https://github.com/radixark/miles/pull/566)
- [SGLang PR #12058](https://github.com/sgl-project/sglang/pull/12058)

**Files:**
- `run_simple.py` - Minimal training script
- `README.md` - Detailed setup and visualization

---

## true_on_policy_vlm

Extends bitwise-identical training to Vision Language Models.

**Key Features:**
- Ensures ViT encoder matches between training and inference
- FSDP across 8 GPUs
- Coordinate alignment between SGLang and Transformers

**References:** [SGLang PR #14636](https://github.com/sgl-project/sglang/pull/14636)

**Models:** Qwen3-VL-2B/4B/8B-Instruct

**Files:**
- `run_simple.py` - VLM true on-policy training script

---

## Common Patterns

### Configuration
Most examples use a combination of:
- Shell scripts (`.sh`) for launch configuration
- YAML files for complex configurations
- Python scripts for custom logic

### Models
Commonly used models across examples:
- **Qwen3-4B/8B/32B** - Text models
- **Qwen3-30B-A3B** - MoE model
- **Qwen3-VL-2B/4B/8B** - Vision-language models
- **Qwen2.5-0.5B/3B** - Smaller models for testing

### Datasets
- **dapo-math-17k** - Math problem dataset
- **aime-2024** - AIME competition problems
- **gsm8k** - Grade school math
- **GEO3K** - Geometry problems with images

## Contributing

When adding new examples:
1. Create a dedicated subdirectory
2. Include a README.md with setup instructions
3. Provide working shell scripts for common configurations
4. Document any external dependencies or API keys needed
5. Update this README with a new entry

See [CONTRIBUTING.md](../CONTRIBUTING.md) for general contribution guidelines.
