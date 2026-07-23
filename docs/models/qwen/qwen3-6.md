---
title: Qwen3.6
description: Launch recipe for the dense Qwen3.6-27B with attention-output-gate.
---
## 1. Model Introduction

[Qwen3.6](https://github.com/QwenLM/Qwen3) is the next iteration of Alibaba's
Qwen3 line, focused on agentic-coding workflows and on preserving reasoning
context across long sessions. The family ships two variants — a sparse MoE
(Qwen3.6-35B-A3B) and a dense GDN-backbone model (Qwen3.6-27B) — both with
native hybrid reasoning (thinking by default), built-in tool calling, and
multimodal text / image / video input. Context windows reach 262 K and
extend past 1 M. Weights are Apache 2.0, available in BF16 and FP8.

The dense **Qwen3.6-27B** is the single-GPU-friendly variant. In miles it
reuses the Qwen3.5 Megatron spec
(`miles_plugins.models.qwen3_5.get_qwen3_5_spec`); architecturally it's a
wider, deeper Qwen3.5 with the gated-attention design preserved.

**Key highlights:**

- **Dense GDN backbone**: 27 B parameters, single-GPU friendly footprint.
- **Attention-output gate**: shared with Qwen3.5, trained alongside attention weights.
- **Extended rotary base**: `--rotary-base 10000000`, `--rotary-percent 0.25`.
- **Larger vocabulary**: 248320 tokens.
- **Shape**: `hidden-size 5120`, `ffn-hidden-size 17408`, 64 layers.
- **Long context**: 262 K tokens, extensible past 1 M.

## 2. Supported Variants

| Model | HF ID |
|---|---|
| Qwen3.6-27B | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) |

## 3. Environment Setup

### 3.1 Download model + datasets

```bash
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/aime-2024
```

### 3.2 HF → Megatron `torch_dist` conversion

```bash
cd /root/miles
source scripts/models/qwen3.6-27B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen3.6-27B \
   --save          /root/Qwen3.6-27B_torch_dist
```

## 4. Launch

### 4.1 Quick start

```bash
cd /root/miles
bash scripts/run-qwen3.6-27B.sh
```

The script targets 1 node × 8 GPU.

## 5. Recipe Configuration

### 5.1 Parallelism

| TP | PP | CP | EP | `max_tokens_per_gpu` | SGLang `mem-fraction-static` | CPU Adam | GPUs |
|---|---|---|---|---|---|---|---|
| 4 | 1 | 1 | 1 | 8192 | 0.5 | ✓ | 8 (1 × 8) |

`--sequence-parallel` is enabled. Activation checkpointing is on
(`--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`).

### 5.2 Algorithm

GRPO with low-variance KL:

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

### 5.3 Rollout & SGLang

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.5
)
```

`--rollout-num-gpus-per-engine 1` follows the Qwen3.5 line; SGLang TP > 1 has been
problematic on this family. If your SGLang version carries the fix for
[sglang#21039](https://github.com/sgl-project/sglang/issues/21039), you can raise it.

### 5.4 Optimizer

CPU Adam is enabled (`--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer`).

### 5.5 Notable quirks

From `scripts/models/qwen3.6-27B.sh`:

- `--spec miles_plugins.models.qwen3_5 get_qwen3_5_spec` — Qwen3.6 reuses the Qwen3.5 spec (gated attention, FP32 `A_log`).
- `--rotary-base 10000000`, `--rotary-percent 0.25`.
- `--vocab-size 248320`.
- `--apply-layernorm-1p`, `--qk-layernorm`, `--group-query-attention`.
- `--attention-output-gate`.

If the trainer OOMs on image-heavy batches, add `--defer-multimodal-cuda-transfer`:
image tensors stay on pinned CPU memory and only the active micro-batch is moved to
the GPU during collation. See [CLI Reference: Memory and throughput](/user-guide/cli-reference#memory-and-throughput).

See [Backends Beyond Megatron](/advanced/architecture-support) for how miles
preserves FP32 parameters like `A_log` through Megatron's mixed-precision pipeline.

## 6. Pairs Well With

- [Backends Beyond Megatron](/advanced/architecture-support)
- [FP8 & Low Precision](/advanced/fp8-low-precision)
