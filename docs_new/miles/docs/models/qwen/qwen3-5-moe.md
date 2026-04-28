---
title: Qwen3.5 MoE
description: Launch recipe for Qwen3.5-35B-A3B with MTP training and EAGLE speculative rollout.
---

# Qwen3.5 MoE

## 1. Model Introduction

[Qwen3.5-35B-A3B](https://github.com/QwenLM/Qwen3) is the MoE branch of the Qwen3.5 line — 3 B active / 35 B total — combining the gated-attention architecture with a built-in MTP head.

**Key highlights:**

- **Sparse MoE**: 3 B active out of 35 B total parameters.
- **Attention-output gate**: shared with the Qwen3.5 dense series, with FP32-preserved `A_log`.
- **Multi-Token Prediction (MTP)**: `--mtp-num-layers 1` baked into the model config; the recipe trains the MTP head and uses EAGLE speculative decoding at rollout.
- **Single-node footprint**: full recipe fits on 1 × 8 GPU.

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| Qwen3.5-35B-A3B | 3 B / 35 B | [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) |

## 3. Environment Setup

### 3.1 Download model + datasets

```bash
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/aime-2024
# Place the model checkpoint at /root/Qwen3.5-35B-A3B
```

### 3.2 HF → Megatron `torch_dist` conversion

```bash
cd /root/miles
source scripts/models/qwen3.5-35B-A3B.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen3.5-35B-A3B \
   --save          /root/Qwen3.5-35B-A3B_torch_dist \
   --mtp-num-layers 1
```

`--mtp-num-layers 1` during conversion preserves the MTP layer so it survives into Megatron format.

## 4. Launch

### 4.1 Quick start

```bash
cd /root/miles
bash scripts/run-qwen3.5-35B-A3B-mtp.sh
```

## 5. Recipe Configuration

### 5.1 Parallelism

| TP | PP | CP | EP | expert-TP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 8 | 1 | 8192 | 8 (1 × 8) |

### 5.2 Algorithm

GRPO with `--eps-clip 0.2 --eps-clip-high 0.28 --use-kl-loss --kl-loss-coef 0.00`. Plus MTP training:

```bash
MTP_ARGS=(
   --enable-mtp-training
   --mtp-num-layers 1
   --mtp-loss-scaling-factor 0.2
)
```

### 5.3 Rollout & SGLang

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-ep-size 8
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)

   # mtp speculative decoding
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 2
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 3

   --sglang-max-running-requests 512
)
```

### 5.4 Optimizer

CPU Adam is enabled (`--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer`).

### 5.5 Notable quirks

- The Megatron side uses `--moe-token-dispatcher-type flex`; DeepEP isn't enabled here, unlike Qwen3-Next.
- The model config (`scripts/models/qwen3.5-35B-A3B.sh`) reuses the Qwen3.5 spec: `--attention-output-gate`, `--rotary-base 10000000`, `--rotary-percent 0.25`, `A_log` kept in FP32 via the bridge. See [Backends Beyond Megatron](../../advanced/architecture-support.md).

## 6. Pairs Well With

- [Speculative Decoding](../../advanced/speculative-decoding.md)
- [Backends Beyond Megatron](../../advanced/architecture-support.md)
