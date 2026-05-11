---
title: GLM4.7 Flash
description: Launch recipes for GLM-4.7-Flash — compact MLA + MoE with R3 enabled by default.
---

# GLM4.7 Flash

## 1. Model Introduction

[GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) is a lightweight, high-speed MoE model in the GLM-4.7 series from Zhipu AI, designed for single-GPU-node deployment.

**Key highlights:**

- **Compact MoE architecture**: 30 B total / 3 B active, sparse activation for efficient inference.
- **MLA attention**: Multi-head Latent Attention with q-LoRA rank 768 and kv-LoRA rank 512.
- **MTP head + EAGLE speculative**: built-in `--mtp-num-layers 1` and EAGLE rollout enabled by default.
- **R3 on by default**: both miles launchers enable `--use-rollout-routing-replay` out of the box.

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| GLM-4.7-Flash | 3 B / 30 B | [zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) |

## 3. Environment Setup

### 3.1 Download model + datasets

```bash
hf download zai-org/GLM-4.7-Flash --local-dir /root/shared/GLM-4.7-Flash
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/shared/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/shared/aime-2024
```

The bash launcher hardcodes `BASE_DIR=/root/shared` (L29). The Python launcher downloads `zhuzilin/dapo-math-17k` and `zhuzilin/aime-2024` automatically.

### 3.2 HF → Megatron `torch_dist` conversion

```bash
cd /root/miles
source scripts/models/glm4.7-flash.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/shared/GLM-4.7-Flash \
   --save          /root/shared/GLM-4.7-Flash_torch_dist
```

The Python launcher does the conversion automatically.

## 4. Launch

### 4.1 Quick start

```bash
# Bash launcher (1 node × 8 GPU)
cd /root/miles
bash scripts/run-glm4.7-flash.sh

# Python launcher (H200 only — `hardware` literal in the dataclass)
python scripts/run_glm47_flash.py
```

Defaults of the Python launcher (see `ScriptArgs`): `model_org=zai-org`, `model_name=GLM-4.7-Flash`, `num_gpus_per_node=8`, `hardware=H200`, `data_dir=/root/datasets`, `model_dir=/root/models`.

## 5. Recipe Configuration

### 5.1 Parallelism

| TP | PP | CP | EP | expert-TP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|
| 4 | 1 | 1 | 8 | 1 | 32768 | 8 (1 × 8) |

`--rollout-num-gpus-per-engine 4` (TP must divide 20 attention heads, so TP=4). The bash launcher's `SGLANG_ARGS` keeps `--sglang-enable-dp-attention` / `--sglang-dp-size` commented out — the in-source comment notes that DP-attention requires `tp_size % dp_size == 0`.

### 5.2 Algorithm

GRPO with `--eps-clip 0.2 --eps-clip-high 0.28 --use-kl-loss --kl-loss-coef 0.00`.

### 5.3 Rollout & SGLang

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.7

   # EAGLE speculative decoding (MTP)
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 2
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 3

   # R3 — on by default in this script
   --use-rollout-routing-replay
)
```

### 5.4 Optimizer

CPU Adam on:

```bash
--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
```

### 5.5 Notable quirks

- Megatron-side DeepEP / `flex` dispatcher are commented out by default in this recipe.
- R3 (`--use-rollout-routing-replay`) is enabled by default — atypical for the rest of the model lineup.

## 6. Pairs Well With

- [Rollout Routing Replay (R3)](../../advanced/miles-router.md) — already on by default.
- [Low Precision RL](../../advanced/fp8-low-precision.md)
