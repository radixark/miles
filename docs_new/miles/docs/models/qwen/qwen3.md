---
title: Qwen3 (Dense)
description: Launch recipes for dense Qwen3 models from 0.6 B to 32 B.
---

# Qwen3 (Dense)

Dense Qwen3 models (0.6 B → 32 B) are the canonical starting point for Miles: one command loads the HuggingFace checkpoint, converts to a Megatron dist checkpoint, and launches GRPO on [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17K). The 4 B recipe fits on a single 8× H100 node and converges in a few thousand steps.

## Variants

| Model | Params | HF ID | Model config |
|---|---|---|---|
| Qwen3-0.6B | 0.6 B | `Qwen/Qwen3-0.6B` | `scripts/models/qwen3-0.6B.sh` |
| Qwen3-1.7B | 1.7 B | `Qwen/Qwen3-1.7B` | `scripts/models/qwen3-1.7B.sh` |
| Qwen3-4B | 4 B | `Qwen/Qwen3-4B` | `scripts/models/qwen3-4B.sh` |
| Qwen3-8B | 8 B | `Qwen/Qwen3-8B` | `scripts/models/qwen3-8B.sh` |
| Qwen3-14B | 14 B | `Qwen/Qwen3-14B` | `scripts/models/qwen3-14B.sh` |
| Qwen3-32B | 32 B | `Qwen/Qwen3-32B` | `scripts/models/qwen3-32B.sh` |

Instruct variants like `Qwen/Qwen3-4B-Instruct-2507` ship with `rotary_base=5_000_000` — pass `--rotary-base 5000000` during conversion.

## Quick start (Qwen3-4B, 8× H100)

```bash
cd /root/miles

# 1. Download weights + dataset
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
hf download --repo-type dataset BytedTsinghua-SIA/DAPO-Math-17K \
  --local-dir /root/dapo-math-17k

# 2. Convert HF → Megatron dist checkpoint
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen3-4B \
   --save          /root/Qwen3-4B_torch_dist

# 3. Launch GRPO
bash scripts/run-qwen3-4B.sh
```

## Expected signal

Trainer stdout looks like:

```
[trainer]  iter 1/3000 | loss=0.412 reward=0.61 rollout=18.4s train=22.1s
```

`loss` should drift down and `reward` should climb over the first few hundred iterations. Checkpoints land under `/root/Qwen3-4B_miles/` every 20 steps (see `--save-interval` in `scripts/run-qwen3-4B.sh`). The [Quick Start](../../getting-started/quick-start.md) walks through the same recipe step by step and shows what to watch for.

---

## Deep dive

### Launch scripts

| Script | GPUs | Notes |
|---|---|---|
| `scripts/run-qwen3-4B.sh` | 8× H100 | Canonical GRPO recipe |
| `scripts/run-qwen3-4B_4xgpu.sh` | 4× H100 | Half-node variant |
| `scripts/run-qwen3-4B_4xgpu-radixtree.sh` | 4× H100 | Variant with radix-tree middleware |
| `scripts/run-qwen3-4B-amd.sh` | 8× MI300X | AMD ROCm variant |
| `scripts/run-qwen3-4B-base-sft.sh` | 8× H100 | SFT (OpenHermes) |
| `scripts/run-qwen3-4B-fsdp.sh` | 8× H100 | FSDP backend, no weight conversion |
| `scripts/run-qwen3-32B.sh` | 16× H100 | 32 B dense |
| `scripts/run-qwen3-8B-amd.sh` | 8× MI300X | 8 B on AMD |

For 32 B and above, drive the HF → Megatron converter with `torchrun --nproc-per-node 8`.

### Parallelism

| Size | TP | PP | CP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|
| 0.6 B – 4 B | 2 | 1 | 1 | 9216 | 8 |
| 8 B | 2 | 1 | 1 | 6144 | 8 |
| 14 B | 2 | 1 | 2 | 4608 | 8 |
| 32 B | 4 | 1 | 2 | 4608 | 16 |

Turn on `--sequence-parallel` whenever TP > 1.

### FSDP variant

If you prefer FSDP over Megatron, use the `-fsdp` scripts. FSDP loads the HuggingFace checkpoint directly — no conversion required:

```bash
bash scripts/run-qwen3-4B-fsdp.sh
```

See [Training backends: FSDP](../../user-guide/usage.md#fsdp-pytorch-fsdp2) for the flag mapping.

### BF16 train + FP8 inference

Qwen3 ships FP8 checkpoints (`Qwen/Qwen3-4B-FP8`). Swap `--hf-checkpoint`:

```bash
CKPT_ARGS=(
   --hf-checkpoint /data/Qwen3-4B-FP8
   --ref-load      /data/Qwen3-4B_torch_dist   # BF16-derived dist ckpt
   ...
)
```

See [FP8 & Low Precision](../../advanced/fp8-low-precision.md) for the end-to-end FP8 path.

### Tuning knobs

| Symptom | Fix |
|---|---|
| Rollout >> train | Increase `rollout-batch-size` and `n-samples-per-prompt` |
| Train OOM | Drop `max-tokens-per-gpu` |
| Rollout OOM | Drop `--sglang-mem-fraction-static` to 0.6 |
| Reward stuck at 0 | Check `--rm-type` matches the dataset label format |
| Generations don't stop | Set `--rollout-stop-token-ids` explicitly |
