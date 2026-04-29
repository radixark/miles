---
title: FP8 and Low Precision
description: Train and infer at the same precision. The unified FP8 path for MoE RL.
---

# FP8 and Low Precision

A common failure mode in MoE RL is precision drift between training and
inference. Pipelines that train in BF16 and serve in FP8 accumulate per-layer
numerical disagreement, which compounds into divergent log-probabilities and
gradients pointing in unintended directions.

Miles supports a unified FP8 path where rollout and training share the same
quantisation logic on the forward pass. Backward passes and master weights stay
in BF16.

## What "unified" means

| Stage | Typical pipeline | Miles unified FP8 |
|---|---|---|
| Rollout (forward) | FP8 GEMM | FP8 GEMM |
| Trainer (forward) | BF16 GEMM | FP8 GEMM with matching quant config |
| Trainer (backward) | BF16 grads | BF16 backward (master weights in BF16) |
| Optimiser | BF16 master | BF16 master |

The forward pass in training matches rollout. The backward pass and master
weights remain BF16, which is what keeps the gradient signal stable.

## Three modes

### 1. BF16 train + FP8 inference

The lowest-friction path. SGLang loads FP8 weights while the trainer keeps a
BF16 `torch_dist` checkpoint. There is precision drift between the two paths;
on MoE workloads, pair this with R3 (and optionally TIS).

```bash
hf download Qwen/Qwen3-30B-A3B-FP8 --local-dir /root/Qwen3-30B-A3B-FP8

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-30B-A3B-FP8        # FP8 weights for SGLang
   --ref-load      /root/Qwen3-30B-A3B_torch_dist  # BF16 torch_dist for trainer
)
```

### 2. Unified FP8

Rollout and training share the same quantisation. The flags fall into three
groups:

```bash
# Megatron / TransformerEngine (PERF_ARGS)
--transformer-impl transformer_engine
--fp8-format e4m3
--fp8-recipe blockwise

# Router (separate group)
--use-miles-router

# Optional, for MoE numerical stability
--use-tis
```

| Flag | Effect |
|---|---|
| `--transformer-impl transformer_engine` | Megatron flag. Routes Megatron's forward through TransformerEngine so FP8 GEMM is actually engaged. |
| `--fp8-format e4m3` | Megatron flag. Forward FP8 format used by TransformerEngine. |
| `--fp8-recipe blockwise` | Megatron flag. Block-wise quantisation recipe; the SGLang side must serve weights in the matching layout. |
| `--use-miles-router` | Miles flag (`add_router_arguments`). Required for R3 and the radix-tree middleware. |
| `--use-tis` | Truncated Importance Sampling for residual precision drift. |

For MoE workloads, also consider `--use-rollout-routing-replay` (R3). The
canonical FP8 recipe leaves it commented out by default but the flag is
available for opt-in.

The canonical recipe is
[`examples/low_precision/run-qwen3-30b-a3b-fp8-two-nodes.sh`](https://github.com/radixark/miles/blob/main/examples/low_precision/run-qwen3-30b-a3b-fp8-two-nodes.sh).
Also enable `NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1` in the Ray runtime env to
use FP32 scales (`miles/ray/actor_group.py:56` already sets this in the actor
env).

### 3. Block-wise FP8 (DeepSeek-style)

For models that ship 128×128 block-wise FP8 weights (such as DeepSeek-R1 and
DeepSeek-V3), the same `--fp8-recipe blockwise` recipe applies. Point
`--hf-checkpoint` at the block-wise FP8 directory and let SGLang autodetect.

## Hardware support

| GPU | FP8 support | Notes |
|---|---|---|
| NVIDIA H100 / H200 | Native | Hardware FP8 GEMM via cuBLASLt |
| NVIDIA B100 / B200 | Native | Higher throughput, same code path |
| NVIDIA A100 | None | Skip FP8, use BF16 |
| AMD MI300X | Partial | FP8 forward only. See [AMD notes](../platforms/amd.md) |

## When BF16 is enough

* Dense models below ~30 B.
* A100 hardware (no FP8 GEMM).
* Bring-up of a new model architecture, where clean BF16 numerics simplify
  debugging.

## Reading order

1. [MilesRouter and R3](miles-router.md): get R3 working first.
2. This page: turn on FP8.
3. [INT4 QAT](int4-qat.md): push further when memory is the constraint.
