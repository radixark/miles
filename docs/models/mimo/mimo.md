---
title: MiMo
description: Single-node GRPO + EAGLE speculative recipe with online MTP training.
---

# MiMo 7B

## 1. Model Introduction

[MiMo-7B-RL](https://huggingface.co/XiaomiMiMo/MiMo-7B-RL) is Xiaomi's dense reasoning RL model with a built-in MTP (Multi-Token Prediction) layer.

**Key highlights:**

- **Dense 7 B with built-in MTP head**: a convenient target for EAGLE-style speculative rollout with online MTP-SFT.
- **Strong reasoning**: matches o1-mini on math and code reasoning at 7 B scale.
- **Single-node recipe**: full RL flow fits on 1 × 8 GPU.
- **Online MTP training**: the recipe trains the MTP head jointly with the policy via `--enable-mtp-training`.

## 2. Supported Variants

| Model | HF ID |
|---|---|
| MiMo-7B-RL | [XiaomiMiMo/MiMo-7B-RL](https://huggingface.co/XiaomiMiMo/MiMo-7B-RL) |

## 3. Environment Setup

### 3.1 Download model + datasets

```bash
hf download XiaomiMiMo/MiMo-7B-RL --local-dir /root/MiMo-7B-RL
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/aime-2024
```

### 3.2 HF → Megatron `torch_dist` conversion

```bash
cd /root/miles
source scripts/models/mimo-7B-rl.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/MiMo-7B-RL \
   --save          /root/MiMo-7B-RL_torch_dist
```

## 4. Launch

### 4.1 Quick start

```bash
cd /root/miles
bash scripts/run-mimo-7B-rl-eagle.sh
```

## 5. Recipe Configuration

### 5.1 Parallelism

| TP | PP | CP | EP | expert-TP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|
| 2 | 1 | 1 | 1 | 1 | 9216 | 8 (1 × 8) |

`--use-dynamic-batch-size` + `--sequence-parallel` on; `--micro-batch-size` is commented out.

### 5.2 Algorithm

GRPO baseline:

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

Plus online MTP training:

```bash
SPEC_ARGS=(
   --enable-mtp-training
   --mtp-loss-scaling-factor 0.2
)
```

`--mtp-num-layers 1` lives in `MODEL_ARGS`, so you get it for free when you source the model config.

### 5.3 Rollout & SGLang

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7

   # for speculative decoding
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 4

   # sometimes flashinfer has IMA bugs. Use fa3 instead
   --sglang-attention-backend fa3
)
```

`--rollout-num-gpus-per-engine 1` is intentional — one SGLang engine per GPU. The `fa3` attention backend is chosen because flashinfer was hitting IMAs (in-source comment).

### 5.4 Optimizer

CPU Adam is **not** enabled.

### 5.5 Notable quirks

- `--save-interval` is **2000** here (much larger than the 20 used in most other launchers); the actor save dir has a `-mtp` suffix (`/root/MiMo-7B-RL-mtp_miles/`).
- R3 is **not** enabled.

## 6. Pairs Well With

- [Speculative Decoding](../../advanced/speculative-decoding.md)
