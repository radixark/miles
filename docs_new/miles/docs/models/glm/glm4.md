---
title: GLM4
description: Launch recipes for GLM-Z1-9B-0414. The 32 B model config ships without a launcher.
---

# GLM4

## 1. Model Introduction

[GLM-Z1-9B-0414](https://huggingface.co/zai-org/GLM-Z1-9B-0414) is a dense reasoning-tuned model from Zhipu AI's GLM-4 series, sized for single-node experimentation.

**Key highlights:**

- **Dense 9 B architecture**: fits comfortably on a single 8-GPU node.
- **Reasoning-tuned**: post-trained for step-by-step reasoning under the GLM-Z1 line.
- **Compatible RL recipe**: GRPO with DAPO-style rollout, drop-in replacement for other dense Qwen / LLaMA-class workloads.

## 2. Supported Variants

| Model | HF ID |
|---|---|
| GLM-Z1-9B-0414 | [zai-org/GLM-Z1-9B-0414](https://huggingface.co/zai-org/GLM-Z1-9B-0414) |

## 3. Environment Setup

### 3.1 Download model + datasets

```bash
hf download zai-org/GLM-Z1-9B-0414 --local-dir /root/GLM-Z1-9B-0414
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/aime-2024
```

### 3.2 HF → Megatron `torch_dist` conversion

```bash
cd /root/miles
source scripts/models/glm4-9B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/GLM-Z1-9B-0414 \
   --save          /root/GLM-Z1-9B-0414_torch_dist
```

## 4. Launch

### 4.1 Quick start

```bash
cd /root/miles
bash scripts/run-glm4-9B.sh                    # 8 GPU
# or:
bash scripts/run-glm4-9B-4xgpu-radixtree.sh    # 4 GPU smoke test
```

## 5. Recipe Configuration

### 5.1 Parallelism

| Script | TP | PP | CP | EP | `max_tokens_per_gpu` | actor / rollout GPUs | GPUs |
|---|---|---|---|---|---|---|---|
| `run-glm4-9B.sh` | 2 | 1 | 2 | 1 | 4608 | 4 / 4 (non-colocate) | 8 (1 × 8) |
| `run-glm4-9B-4xgpu-radixtree.sh` | 2 | 1 | 1 | 1 | 2304 | 4 / 2 | 4 (1 × 4) |

### 5.2 Algorithm

GRPO across both scripts:

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
# run-glm4-9B.sh
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
)

# run-glm4-9B-4xgpu-radixtree.sh
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
)
```

### 5.4 Optimizer

CPU Adam is not enabled in either launcher.

### 5.5 Notable quirks

- `run-glm4-9B.sh` runs actor and rollout on disjoint GPUs (non-colocate).

## 6. Pairs Well With

- [Rollout Routing Replay (R3)](../../advanced/miles-router.md)
- [Low Precision RL](../../advanced/fp8-low-precision.md)
