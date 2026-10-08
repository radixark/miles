---
title: GLM-5.3
description: GLM-5.3 (744 B / 40 B active) on the GLM-5.2 recipe — same architecture, Megatron config, and launchers.
---
## 1. Model Introduction

[GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) is the successor to GLM-5.2 in Zhipu AI's GLM series. It keeps the 744 B-parameter (40 B active) `glm_moe_dsa` architecture of GLM-5.2 unchanged — MoE plus DeepSeek Sparse Attention (DSA) with cross-layer index sharing — so Miles trains it with the GLM-5.2 Megatron model args, bridge, and recipe. Only the checkpoint differs.

GLM-5.3 is not GLM-5.3-Flash. The Flash model is a separate KDA + DSA hybrid architecture with its own recipe; see [GLM-5.3-Flash](/models/glm/glm5-3-flash).

## 2. Supported Variants

| Model | Active / Total | HF ID | Notes |
|---|---|---|---|
| GLM-5.3 (BF16) | 40 B / 744 B (78 layers) | [zai-org/GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16) | Training checkpoint |
| GLM-5.3 (FP8) | 40 B / 744 B (78 layers) | [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) | Official FP8 release, usable as the rollout checkpoint |

Training is BF16, so start from `GLM-5.3-BF16`. For smoke tests, use the 5-layer GLM-5.2 prune (`GLM-5.2_5layer`): the shapes are identical.

## 3. Launch

### 3.1 LoRA

`scripts/run_glm5_2_744b_a40b_lora.py` takes the checkpoint path through `--hf-checkpoint`, so GLM-5.3 runs on the `GLM-5.2` model entry (Megatron model type `glm5.2-744B-A40B_lora`). Stage the checkpoint and dataset yourself and call `train`: `prepare` and `full-train` download the repo named by `--model-name`, which would fetch GLM-5.2.

```bash
hf download zai-org/GLM-5.3-BF16 --local-dir /root/models/GLM-5.3-BF16
hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/datasets/gsm8k

python scripts/run_glm5_2_744b_a40b_lora.py train \
  --model-name GLM-5.2 \
  --hf-checkpoint /root/models/GLM-5.3-BF16
```

For `--task dapo-math`, download `zhuzilin/dapo-math-17k` to `/root/datasets/dapo-math-17k` instead.

FP8 rollout works as it does for GLM-5.2: training stays BF16 and SGLang serves `<hf_checkpoint>_fp8`. Point that at the official FP8 release (for example, download `zai-org/GLM-5.3` to `/root/models/GLM-5.3-BF16_fp8`) and add `--fp8-rollout`.

### 3.2 Full-parameter

`scripts/run_glm5_2_744b_a40b.py` does not take a checkpoint override yet: `prepare` downloads `<model-org>/<model-name>` and accepts only the `GLM-5.2` and `GLM-5.2_5layer` names. Use the LoRA launcher above for GLM-5.3 until the full-parameter launcher gains a GLM-5.3 entry.

## 4. Recipe Configuration

Every setting comes from the GLM-5.2 LoRA launcher unchanged: parallelism, DSA kernel backend (`--dsa-attention-backend`), LoRA targets, algorithm, and rollout flags. See [LoRA](/advanced/lora) for the GLM-5.2 Bridge LoRA details and validation results, and [GLM-5.2](/models/glm/glm5-2) for the DSA constraint that every pipeline stage must start on an indexer-computing layer.

## 5. Pairs Well With

- [GLM-5.2](/models/glm/glm5-2) — the recipe GLM-5.3 runs on.
- [LoRA](/advanced/lora) — the GLM-5 / 5.1 / 5.2 Bridge LoRA path also covers GLM-5.3.
- [Low Precision RL](/advanced/low-precision) — FP8 rollout from the official FP8 checkpoint.
