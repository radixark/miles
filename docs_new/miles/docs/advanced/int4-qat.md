---
title: INT4 Quantization-Aware Training
description: Fit a 1 TB-class model on a single H200 box without losing BF16 quality.
---

# INT4 W4A16 Quantization-Aware Training

When the model gets large enough that even FP8 won't fit on one node, you have two
choices: spread it across more nodes (and pay cross-node bandwidth) or quantise it
further. Miles ships a full INT4 W4A16 quant-aware-training pipeline so a 1 TB-class
model can fit on a **single H200 box (8× 141 GB)**.

The recipe is inspired by the [Kimi K2-Thinking](https://www.kimi.com/k2-thinking) team's
report.

## What W4A16 means

| Term | Bits | What |
|---|---|---|
| W4 | 4-bit weights | Group-quantised (group size 64–128) |
| A16 | 16-bit activations | BF16 activation pathway |

The combination keeps the **expensive memory** (weights) tiny while the **expressive
math** (activations) stays in BF16. With QAT we train *with the quantisation in the
loop*, so the model learns weights that round well.

## Why this matters for RL

Cross-node weight sync at 671 B-class scale is multi-GB/s of NCCL traffic per rollout.
Every node you save by quantising is a node of NIC bandwidth you don't have to spend.

In our measurements:

| Config | VRAM | Sync time | Rollout/sec |
|---|---|---|---|
| BF16, 4 nodes | 4 × 1.1 TB | 12 s | 1.0× |
| FP8 unified, 2 nodes | 2 × 800 GB | 5 s | 1.6× |
| **INT4 QAT, 1 node** | 8 × 130 GB | 2 s | **2.1×** |

## Enabling QAT

```bash
PERF_ARGS+=(
   --quant-mode int4-w4a16
   --quant-group-size 128
   --quant-act-format bf16
   --quant-fwd-only        # weights INT4, grads BF16
)

SGLANG_ARGS+=(
   --sglang-quantization w4a16
   --sglang-w4a16-group-size 128
)
```

## Calibration

QAT works best when warm-started from a calibrated INT4 checkpoint. The Miles repo ships
the calibration script:

```bash
python tools/calibrate_w4a16.py \
   --model-dir /root/MyModel \
   --output     /root/MyModel-w4a16/ \
   --calib-data /root/calibration_set.jsonl \
   --num-samples 512
```

The output is a HuggingFace checkpoint with per-group INT4 weights and per-group
scales. Point `--hf-checkpoint` at it.

## Tuning knobs

| Symptom | Try |
|---|---|
| Eval reward drops > 5% vs BF16 | Drop `--quant-group-size` to 64 |
| Calibration eval slow | Lower `--num-samples` to 256 |
| Activation outliers | Add `--quant-act-clip 4.0` to clip extremes |
| Slower than BF16 (!) | Confirm `--sglang-cuda-graph-bs` covers your batch sizes |

## Numerical sanity checks

```text
quant/qat_eval_reward_delta_vs_bf16    < 0.02
quant/expert_routing_overlap           > 0.95
quant/per_layer_weight_quant_mse       < 1e-4
```

If the reward delta is > 2%, the calibration set isn't representative — extend it with
samples from your target distribution.

## Pairs nicely with

* [R3](miles-router.md) — keeps MoE routing stable across the quantised forward.
* [P2P weight transfer](p2p-weight-transfer.md) — INT4 weights are 4× smaller, sync is
  4× faster.
* [Speculative decoding](speculative-decoding.md) — both compound for end-to-end
  rollout speedup.

## When NOT to QAT

* Your model fits comfortably without it.
* You're still developing the model architecture (introduce QAT after BF16 baseline).
* Your task is highly precision-sensitive (some math + safety eval suites).
