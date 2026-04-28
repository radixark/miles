---
title: FP8 & Low Precision
description: Train and infer at the same precision — the unified FP8 path that survives MoE RL.
---

# FP8 & Low Precision

The biggest source of MoE RL collapse — after the routing mismatch fixed by
[R3](miles-router.md) — is **precision drift between training and inference**. Most
pipelines train in BF16 and serve in FP8 / INT8; the tiny per-layer numerical
disagreements compound into divergent log-probabilities and gradients that point in the
wrong direction.

Miles ships a **unified FP8 pipeline**: rollout and training share the exact same FP8
quantisation logic.

## What "unified" means

| Stage | Most pipelines | Miles unified FP8 |
|---|---|---|
| Rollout (forward) | FP8 GEMM, per-tensor scales | FP8 GEMM, per-tensor scales |
| Trainer (forward) | BF16 GEMM | **FP8 GEMM, same scales** |
| Trainer (backward) | BF16 grads | FP8 fwd + BF16 backward (master weights kept in BF16) |
| Optimiser | BF16 master | BF16 master (same) |

The point is the **forward pass** in training matches rollout bit-for-bit. The
backward pass and master weights stay BF16 — that's what keeps the gradient signal sane.

## Three modes Miles supports

### 1. BF16 train + FP8 inference

Cheapest to enable. Useful as a stepping stone.

```bash
hf download Qwen/Qwen3-30B-A3B-FP8 --local-dir /root/Qwen3-30B-A3B-FP8

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-30B-A3B-FP8        # FP8 weights for SGLang
   --ref-load      /root/Qwen3-30B-A3B_torch_dist  # BF16 dist for trainer
   ...
)
```

SGLang serves in FP8 by directly casting BF16 weights at load time. This works but
introduces drift on MoE — fix it with R3 + TIS.

### 2. Unified FP8 (recommended)

```bash
SGLANG_ARGS+=(
   --use-miles-router
)

PERF_ARGS+=(
   --transformer-impl transformer_engine
   --fp8-format e4m3
   --fp8-recipe blockwise
)

GRPO_ARGS+=(
   --use-miles-router
   --use-rollout-routing-replay
   --use-tis
)
```

What each flag does:

| Flag | Effect |
|---|---|
| `--fp8-format e4m3` | Forward FP8 format used by Transformer Engine |
| `--fp8-recipe blockwise` | Block-wise quantisation recipe (matches SGLang's block-wise serve) |
| `--use-rollout-routing-replay` | R3 (essential for MoE) |
| `--use-tis` | Truncated importance sampling guard |

The canonical recipe is [`examples/low_precision/run-qwen3-30b-a3b-fp8-two-nodes.sh`](https://github.com/radixark/miles/blob/main/examples/low_precision/run-qwen3-30b-a3b-fp8-two-nodes.sh) — also enable `NVTE_FP8_BLOCK_SCALING_FP32_SCALES` in the Ray runtime env to use FP32 scales.

### 3. Block-wise FP8 (DeepSeek-style)

For models like DeepSeek-R1 that ship 128×128 block-wise FP8 weights, the same
`--fp8-recipe blockwise` recipe applies — point `--hf-checkpoint` at the block-wise
FP8 directory and let SGLang autodetect.

## Hardware support

| GPU | FP8 support | Notes |
|---|---|---|
| NVIDIA H100 / H200 | ✅ Native | Full hardware FP8 GEMM via cuBLASLt |
| NVIDIA B100 / B200 | ✅ Native | Higher throughput; same code path |
| NVIDIA A100 | ❌ | Skip FP8; use BF16 |
| AMD MI300X | ✅ Partial | FP8 for forward only — see [AMD notes](../platforms/amd.md) |

## Numerical sanity checks

After enabling FP8, run a few rollouts and verify:

```text
fp8/forward_logprob_mse        ≈ 0
fp8/forward_logprob_max_diff   < 1e-4
router/replay_hit_rate         = 1.000
moe/expert_balance_std         < 0.10
loss/policy_kl                 stable
```

If `forward_logprob_mse` is non-zero, your trainer's FP8 quantisation is not matching
SGLang's. Common causes:

* `--fp8-recipe` doesn't match what SGLang loaded (e.g. trainer on `blockwise` while
  the served checkpoint was per-tensor quantised, or vice versa).
* Trainer running BF16 because `--transformer-impl transformer_engine` was omitted.

## When to NOT use FP8

* Your model is dense and < 30 B. BF16 is fine.
* You're on A100 hardware (no FP8 GEMM).
* You're debugging a new model architecture and need clean BF16 numerics first.

## Reading order

1. [Miles Router](miles-router.md) — get R3 working first.
2. This page — turn on FP8.
3. [INT4 QAT](int4-qat.md) — push further when memory is the constraint.
