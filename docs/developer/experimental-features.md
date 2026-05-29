---
title: Experimental Features
description: Backends and features that exist in tree but are not production-ready — opt-in at your own risk.
---

# Experimental Features

These features live in the Miles tree but are **not** production-ready. They typically
have rough edges, missing parallelism, or known bugs against current dependency
versions. Use them when you want to iterate quickly or co-develop a feature, not for
the long-running training jobs you'd publish results from.

## FSDP backend

A PyTorch FSDP2 training backend lives at `miles/backends/fsdp_utils/`.
It trades maximum throughput for **zero conversion overhead**: there is no
`torch_dist` step, Miles reads architecture information from the HuggingFace
`config.json`, and weights load directly via `AutoModelForCausalLM.from_pretrained()`.
The distributed optimizer is built into FSDP, and mixed precision falls out of standard
PyTorch.

<Warning>

**Status.** Experimental. Known bug after SGLang v0.5.10. No TP / PP / CP / EP — runs as
plain FSDP data parallel only. Suitable for fast iteration on small-to-mid dense
models, not for production runs.

</Warning>

### When to reach for it

- Iterating on a new model architecture and you don't want to write a Megatron spec yet.
- Small-to-mid dense workloads where the parallelism story doesn't matter.
- You want a HuggingFace-native checkpoint at every step with no conversion.

For large MoE models, multi-rack jobs, or anything where TP / PP / CP / EP matters,
use the production [Megatron-LM backend](../user-guide/usage.md#megatron-lm) instead.

### Enabling it

```bash
--train-backend fsdp
```

### Flag mapping vs. Megatron

Most RL-level flags carry over unchanged. Backend-specific differences:

| Concern | Megatron | FSDP |
|---|---|---|
| Model load | `--load` + architecture args | `--hf-checkpoint` *(single flag, required)* |
| Tensor parallel | `--tensor-model-parallel-size` | Not supported yet |
| Pipeline parallel | `--pipeline-model-parallel-size` | Not supported yet |
| Expert parallel | `--expert-model-parallel-size` | Not supported yet |
| Context parallel | `--context-parallel-size` | Not supported yet |
| Optimizer | `--use-distributed-optimizer` *(forced on by Miles)* | Built-in |
| Gradient checkpoint | `--recompute-granularity / method / num-layers` | `--gradient-checkpointing` *(boolean)* |
| CPU offload | Distributed optimizer | `--fsdp-cpu-offload` |
| CPU backend | *(in distributed optimizer)* | `--fsdp-cpu-backend` |
| Attention backend | Decided by Megatron Core | `--attn-implementation flash_attention_2 / sdpa / eager` |
| Mixed precision | `--fp16` / `--bf16` | `--fp16` *(bf16 inferred)* |
| Extra backend config | — | `--config <yaml>` |

### Quick start

```bash
# Optional: wandb
export WANDB_API_KEY=<key>

# Model + data
hf download Qwen/Qwen3-4B                                       --local-dir /root/Qwen3-4B
hf download --repo-type dataset BytedTsinghua-SIA/DAPO-Math-17K --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024              --local-dir /root/aime-2024

# Code
git clone https://github.com/radixark/miles.git && cd miles
pip install -e . --no-deps

# Launch — no conversion step
bash scripts/run-qwen3-4B-fsdp.sh
```
