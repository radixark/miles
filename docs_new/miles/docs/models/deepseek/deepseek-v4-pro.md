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
| DeepSeek-V4-Pro-FP8 | 49 B / 1.6 T | [sgl-project/DeepSeek-V4-Pro-FP8](https://huggingface.co/sgl-project/DeepSeek-V4-Pro-FP8) |

## 3. Quick start

### 3.1 One-line launch

```bash
# Pull the image matching your cluster (TBD)
docker pull <image>

# Production Pro run, inside the container
cd /root/miles
python scripts/run_deepseek_v4.py full-train \
   --model-name DeepSeek-V4-Pro-FP8 \
   --num-nodes 32 --num-gpus-per-node 8
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

The under-the-hood stages are essentially identical to V4-Flash — see the [V4-Flash Script breakdown](deepseek-v4-flash.md#4-script-breakdown) and substitute the Pro model name and path defaults shown above.

## 5. Example Recipe Configuration

### 5.1 Parallelism

| Hardware | Nodes × GPUs | TP | PP | EP | expert-TP | `max_tokens_per_gpu` | Pipeline layout |
|---|---|---|---|---|---|---|---|
| H200 | 32 × 8 = 256 | 8 | 8 | 32 | 1 | 2048 | first 7 / last 6 layers |

The launcher additionally flips two Pro-specific defaults on selection of
`--model-name DeepSeek-V4-Pro-FP8`: `optimizer_offload=True` (Adam states offloaded to
CPU to fit Pro on H200) and `enable_r3=False` (Rollout Routing Replay disabled).

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
