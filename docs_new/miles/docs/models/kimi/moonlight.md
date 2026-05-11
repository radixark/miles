---
title: Moonlight
description: Single-node MoE recipe (8 GPU) — DAPO-style dynamic sampling and CPU Adam on by default.
---

# Moonlight

## 1. Model Introduction

[Moonlight](https://huggingface.co/moonshotai/Moonlight-16B-A3B) is Moonshot AI's compact MoE — 16 B total / 3 B active, trained with the Muon optimizer — and a useful single-node test target for MoE RL code changes before scaling to Kimi K2.

**Key highlights:**

- **Compact MoE**: 16 B total / 3 B active, 27 layers (1 dense + 26 MoE), 64 routed experts top-6 + 2 shared.
- **MLA attention**: Multi-head Latent Attention with `kv-LoRA rank 512`.
- **Single-node footprint**: full RL recipe fits on 1 × 8 H100.
- **Muon-trained base**: pretrained with the Muon optimizer; weight decay matters at scale.

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| Moonlight-16B-A3B | 3 B / 16 B | [moonshotai/Moonlight-16B-A3B](https://huggingface.co/moonshotai/Moonlight-16B-A3B) |

## 3. Environment Setup

### 3.1 Download model + datasets

```bash
hf download moonshotai/Moonlight-16B-A3B --local-dir /root/Moonlight-16B-A3B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/aime-2024
```

### 3.2 HF → Megatron `torch_dist` conversion

```bash
cd /root/miles
source scripts/models/moonlight.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Moonlight-16B-A3B \
   --save          /root/Moonlight-16B-A3B_torch_dist
```

## 4. Launch

### 4.1 Quick start

```bash
cd /root/miles
bash scripts/run-moonlight-16B-A3B.sh
```

## 5. Recipe Configuration

### 5.1 Parallelism

| TP | PP | CP | EP | expert-TP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|
| 4 | 1 | 1 | 8 | 1 | 8192 | 8 (1 × 8) |

### 5.2 Algorithm

GRPO with `--eps-clip 0.2 --eps-clip-high 0.28 --use-kl-loss --kl-loss-coef 0.00`. R3 is **not** enabled.

```bash
ROLLOUT_ARGS=(
   --rm-type math
   --num-rollout 3000
   --rollout-batch-size 128
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 1

   --over-sampling-batch-size 256
   --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

   --num-steps-per-rollout 4
   --balance-data
)
```

### 5.3 Rollout & SGLang

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)
```

Megatron-side DeepEP is on: `--moe-enable-deepep --moe-token-dispatcher-type flex`.

### 5.4 Optimizer

CPU Adam on:

```bash
--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
```

### 5.5 Notable quirks

- `--attention-backend flash` is **commented out** in this script (script comment: "need to comment this when using model with MLA").

## 6. Pairs Well With

- [Rollout Routing Replay (R3)](../../advanced/miles-router.md)
- [Low Precision RL](../../advanced/fp8-low-precision.md)
