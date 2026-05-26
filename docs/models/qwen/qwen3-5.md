---
title: Qwen3.5
description: Launch recipes for Qwen3.5-4B / 9B / 27B with attention-output-gate.
---

# Qwen3.5

## 1. Model Introduction

[Qwen3.5](https://github.com/QwenLM/Qwen3) is the next iteration of the Qwen3 dense series, introducing the gated-attention architecture and an FP32-preserved `A_log` parameter.

**Key highlights:**

- **Attention-output gate**: a learned gate on the attention output, trained alongside attention weights for stronger long-context behaviour.
- **Extended rotary base**: `--rotary-base 10000000`, `--rotary-percent 0.25` — wider effective context than the original Qwen3.
- **Larger vocabulary**: 248320 tokens.
- **FP32 `A_log` preservation**: a parameter that must stay in FP32 through Megatron's mixed-precision pipeline; miles handles this via the bridge.

## 2. Supported Variants

| Model | HF ID |
|---|---|
| Qwen3.5-4B | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) |
| Qwen3.5-9B | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) |
| Qwen3.5-27B | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) |

## 3. Environment Setup

### 3.1 Download model + datasets

```bash
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/aime-2024
# Place the model checkpoint at /root/Qwen3.5-{4B,9B,27B}
```

### 3.2 HF → Megatron `torch_dist` conversion

```bash
cd /root/miles
source scripts/models/qwen3.5-4B.sh   # or qwen3.5-9B.sh / qwen3.5-27B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen3.5-4B \
   --save          /root/Qwen3.5-4B_torch_dist
```

## 4. Launch

### 4.1 Quick start

```bash
cd /root/miles
bash scripts/run-qwen3.5-4B.sh        # or run-qwen3.5-9B.sh / run-qwen3.5-27B.sh
```

All three scripts are 1 node × 8 GPU.

## 5. Recipe Configuration

### 5.1 Parallelism

| Script | TP | PP | CP | `max_tokens_per_gpu` | SGLang `mem-fraction-static` | CPU Adam | GPUs |
|---|---|---|---|---|---|---|---|
| `scripts/run-qwen3.5-4B.sh` | 2 | 1 | 1 | 9216 | 0.7 | – | 8 (1 × 8) |
| `scripts/run-qwen3.5-9B.sh` | 2 | 1 | 1 | 9216 | 0.6 | – | 8 (1 × 8) |
| `scripts/run-qwen3.5-27B.sh` | 4 | 1 | 1 | 8192 | 0.5 | ✓ | 8 (1 × 8) |

### 5.2 Algorithm

GRPO across all three sizes:

```bash
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)
```

### 5.3 Rollout & SGLang

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
)
```

All three scripts pin `--rollout-num-gpus-per-engine 1` because SGLang TP > 1 produced garbage output for Qwen3.5 on 0.5.9 (in-source comment, [sglang#21039](https://github.com/sgl-project/sglang/issues/21039)). If you bump the SGLang version past the fix, you can raise this back up.

### 5.4 Optimizer

Only the 27 B script enables CPU Adam (`--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer`). The 4 B and 9 B recipes leave Adam on GPU.

### 5.5 Notable quirks

From `scripts/models/qwen3.5-4B.sh` (and analogous configs for 9 B / 27 B):

- `--spec miles_plugins.models.qwen3_5 get_qwen3_5_spec` — attention-output gate, `A_log` parameter handling.
- `--rotary-base 10000000`, `--rotary-percent 0.25`.
- `--vocab-size 248320`.
- `--apply-layernorm-1p`, `--qk-layernorm`, `--group-query-attention`.
- `--attention-output-gate`.

See [Backends Beyond Megatron](../../advanced/architecture-support.md) for how miles preserves FP32 parameters like `A_log` through Megatron's mixed-precision pipeline.

## 6. Pairs Well With

- [Backends Beyond Megatron](../../advanced/architecture-support.md)
- [Low Precision RL](../../advanced/fp8-low-precision.md)
