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

## Calibration

Convert a BF16 HuggingFace checkpoint to INT4 with `tools/convert_hf_to_int4.py`
(GPTQ via `llmcompressor`):

```bash
python tools/convert_hf_to_int4.py \
   --input-dir  /root/MyModel \
   --output-dir /root/MyModel-INT4 \
   --data-dir   /root/calibration_dataset \
   --quant-type W4A16 \
   --num-calibration-samples 256 \
   --quant-group-size 128
```

The output is a HuggingFace directory with per-group INT4 weights and scales. Point
`--hf-checkpoint` at it; SGLang autodetects the quantisation at load time.

## Enabling QAT

QAT is currently driven by environment variables passed through Ray's runtime env
rather than CLI flags. The canonical recipe is
[`examples/low_precision/run-qwen3-30B-A3B-int4.sh`](https://github.com/radixark/miles/blob/main/examples/low_precision/run-qwen3-30B-A3B-int4.sh):

```bash
RUNTIME_ENV_JSON='{
  "env_vars": {
    "OPEN_TRAINING_INT4_FAKE_QAT_FLAG": "1",
    "OPEN_TRAINING_INT4_GROUP_SIZE": "128"
  }
}'

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py ...
```

Pair the INT4 `--hf-checkpoint` with a BF16 `--ref-load` torch_dist directory so the
KL anchor stays full precision.

## Tuning knobs

| Symptom | Try |
|---|---|
| Eval reward drops > 5% vs BF16 | Lower `OPEN_TRAINING_INT4_GROUP_SIZE` (e.g. 64), or recalibrate with more samples |
| Slower than BF16 (!) | Confirm `--sglang-cuda-graph-bs` covers your batch sizes |

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
