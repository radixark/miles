---
title: DeepSeek V4 Pro
description: Launch recipe for DeepSeek-V4-Pro (1.6 T) — V4-family architecture at Pro scale.
---

# DeepSeek V4 Pro

!!! note "Work in progress"
    This page is being filled in. The skeleton below mirrors the
    [V4-Flash recipe page](deepseek-v4-flash.md); sections marked **TBD** will
    receive Pro-specific content as the recipe lands. Tracking issue:
    [`radixark/miles#1046`](https://github.com/radixark/miles/issues/1046).

## 1. Model Introduction

TBD — Pro-specific architecture summary (active / total params, attention stack, KV
compressors, MoE topology, RoPE / context length, FP8 vs BF16 defaults).

**Key highlights:**

- TBD
- TBD
- TBD
- TBD

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| DeepSeek-V4-Pro | 49 B / 1.6 T | TBD |

## 3. Quick start

### 3.1 One-line launch

```bash
# Pull the image matching your cluster (TBD)
docker pull <image>

# Production Pro run, inside the container
cd /root/miles
python scripts/run_deepseek_v4.py full-train \
   --model-name DeepSeek-V4-Pro \
   --num-nodes 8 --num-gpus-per-node 256
```

TBD — describe what `full-train` chains for Pro and any Pro-specific stage skips.

### 3.2 Launcher path defaults

| Flag | Default | Use |
|---|---|---|
| `--data-dir` | `/root/datasets` | HF datasets (e.g. dapo-math-17k, …) |
| `--model-dir` | `/root/models` | parent directory holding the HF checkpoint and Megatron `_torch_dist` artifacts |
| `--model-local-dir` | `/root/local_data` | local NVMe path on each node; `prepare-cp` rsyncs the HF checkpoint and `_torch_dist` here so the trainer reads from local disk |
| `--save-dir` | `/root/models` | training checkpoints under `{save-dir}/{run-id}/checkpoints/` |

TBD — Pro-specific overrides or env-var notes.

## 4. Script breakdown

What `full-train` does under the hood, and how to drive each stage manually.

### 4.1 Download model + datasets

TBD

### 4.2 HF → Megatron `torch_dist` conversion

TBD — FP8 → BF16 cast (if applicable) and the distributed conversion command for Pro.

### 4.3 Multi-node fan-out

TBD — Ray bootstrapping, shared-filesystem expectations, `MILES_SCRIPT_EXTERNAL_RAY` /
`RAY_ADDRESS` handling for Pro.

### 4.4 Notable quirks

- TBD

## 5. Example Recipe Configuration

### 5.1 Parallelism

| Hardware | Nodes × GPUs | TP | PP | EP | expert-TP | `max_tokens_per_gpu` | Pipeline layout |
|---|---|---|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 5.2 Algorithm

TBD — GRPO / loss flags / routing-bias freeze constraints for Pro.

### 5.3 Rollout & SGLang

```bash
SGLANG_ARGS=(
   # TBD
)
```

TBD — required env vars and Megatron-side flags.

### 5.4 Optimizer

```bash
# TBD
```

## 6. Pairs Well With

- [FP8 & Low Precision](../../advanced/fp8-low-precision.md)
- [Architecture Support](../../advanced/architecture-support.md)
- [DeepSeek V4 Flash](deepseek-v4-flash.md) — sibling recipe; shares the V4-family architecture.
