---
title: Qwen3
description: Launch recipes for dense Qwen3 models (0.6 B – 32 B).
---

# Qwen3

## 1. Model Introduction

[Qwen3](https://github.com/QwenLM/Qwen3) is the latest generation of Alibaba's Qwen language model series, available in dense and MoE variants with both Instruct and reasoning-enhanced Thinking editions.

**Key highlights:**

- **Stronger general intelligence**: significant improvements in instruction following, logical reasoning, mathematics, science, coding, and tool usage over Qwen2.5.
- **Extended context length**: trained for 256 K-token contexts, useful for long-document reasoning and agentic workflows.
- **Flexible deployment options**: dense sizes from 0.6 B up to 32 B; this page covers the dense recipes (MoE recipes live in [qwen3-moe](qwen3-moe.md)).
- **Stronger agent interaction**: improved tool-use and search-based agent performance.

## 2. Supported Variants

| Model | HF ID |
|---|---|
| Qwen3-0.6B | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) |
| Qwen3-1.7B | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) |
| Qwen3-4B | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) |
| Qwen3-4B-Instruct-2507 | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| Qwen3-8B | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| Qwen3-14B | [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) |
| Qwen3-32B | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) |

## 3. Environment Setup

### 3.1 Download model + datasets

```bash
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/aime-2024
```

### 3.2 HF → Megatron `torch_dist` conversion

```bash
cd /root/miles
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen3-4B \
   --save          /root/Qwen3-4B_torch_dist
```

The converter auto-derives PP from `WORLD_SIZE`; for larger sizes drive it with `torchrun --nproc-per-node 8`. The FSDP launcher loads the HF checkpoint directly and skips this step.

## 4. Launch

### 4.1 Quick start

```bash
cd /root/miles
bash scripts/run-qwen3-4B.sh
```

Other variants follow the same pattern — replace the script name (`run-qwen3-32B.sh`, `run-qwen3-4B-fsdp.sh`, etc.) and the `qwen3-XB.sh` model config.

The Qwen3-4B-Instruct-2507 config (`scripts/models/qwen3-4B-Instruct-2507.sh`) just sets `MODEL_ARGS_ROTARY_BASE=5000000` and re-sources `qwen3-4B.sh` — source it when converting / launching the Instruct-2507 checkpoint.

## 5. Recipe Configuration

### 5.1 Parallelism

| Script | TP | PP | CP | EP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|
| `run-qwen3-4B.sh` | 2 | 1 | 1 | 1 | 9216 | 8 (1 × 8) |
| `run-qwen3-4B_4xgpu.sh` | 2 | 1 | 1 | 1 | 9216 | 4 (1 × 4) |
| `run-qwen3-32B.sh` | 8 | 1 | 1 | 1 | 20480 | 8 (1 × 8) |

`--sequence-parallel` is on whenever TP > 1.

### 5.2 Algorithm

GRPO baseline across all dense recipes:

```bash
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)
```

Rollout uses `--rm-type deepscaler` against `dapo-math-17k`. The SFT recipe (`run-qwen3-4B-base-sft.sh`) trains on `/root/openhermes2_5.parquet`.

### 5.3 Rollout & SGLang

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
)
```

`run-qwen3-32B.sh` additionally pins `--sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)`. The FSDP variant uses `--attn-implementation flash_attention_3`, SGLang attention backend `fa3`, and adds `--update-weight-buffer-size 536870912 --gradient-checkpointing`.

### 5.4 Optimizer

`run-qwen3-32B.sh` enables CPU Adam:

```bash
--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
```

The 4 B / 8 B / 14 B recipes leave Adam on GPU.

### 5.5 Notable quirks

- **BF16 train + FP8 inference**: `run-qwen3-4B.sh:30-31` ships a commented `--hf-checkpoint /root/Qwen3-4B-FP8` alternative — uncomment it (and download `Qwen/Qwen3-4B-FP8`) to swap rollout to FP8 while keeping BF16 training. See [Low Precision RL](../../advanced/fp8-low-precision.md).
- **FSDP backend**: `run-qwen3-4B-fsdp.sh` runs the same recipe with `--train-backend fsdp`; no Megatron `torch_dist` conversion needed.
- **AMD ROCm**: `scripts/amd/run-qwen3-4B-amd.sh` mirrors the recipe with `${NUM_GPUS}` resolved from the AMD environment.

## 6. Pairs Well With

- [Low Precision RL](../../advanced/fp8-low-precision.md)
- [Backends Beyond Megatron](../../advanced/architecture-support.md) — for the FSDP variant.
