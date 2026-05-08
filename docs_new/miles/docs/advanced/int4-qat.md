---
title: INT4 Quantization-Aware Training
description: Fit large models on a single 8-GPU node by training with W4A16 quantization in the loop.
---

# INT4 W4A16 Quantization-Aware Training

When the model is large enough that even FP8 will not fit on one node, the
options are spreading across more nodes (and paying cross-node bandwidth) or
quantizing further. Miles ships an INT4 W4A16 quant-aware-training pipeline.
On an 8 × 141 GB H200 node, this is the path used to fit very large models in
a single box.

The recipe is inspired by the
[Kimi K2-Thinking](https://www.kimi.com/k2-thinking) team's report.

## What W4A16 means

| Term | Bits | Notes |
|---|---|---|
| W4 | 4-bit weights | Group-quantized (typical group size 32–128) |
| A16 | 16-bit activations | BF16 activation pathway |

The combination keeps the weights small (memory-bound) while activations stay
in BF16 (math-bound). With QAT the model trains *with* the quantization in the
loop, so the weights round well during inference.

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

| Flag | Default | Notes |
|---|---|---|
| `--quant-type` | `W4A16` | Also accepts `W8A16`. |
| `--num-calibration-samples` | `256` | Calibration set size. |
| `--quant-group-size` | `32` | GPTQ group size; `128` is also common. |
| `--max-sequence-length` | `2048` | Calibration sequence length. |
| `--dampening-frac` | `0.01` | GPTQ damping. |
| `--trust-remote-code` | off | Pass when the HF config requires custom code. |

The output is a HuggingFace directory with per-group INT4 weights and scales.
Point `--hf-checkpoint` at it; SGLang autodetects the quantization at load time.

## Enabling QAT

QAT is currently driven by environment variables passed through Ray's runtime
env rather than CLI flags. The canonical recipe is
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

Pair the INT4 `--hf-checkpoint` with a BF16 `--ref-load` torch_dist directory
so the KL anchor stays full-precision.

## Tuning

| Symptom | Try |
|---|---|
| Eval reward drops noticeably vs BF16 | Lower `OPEN_TRAINING_INT4_GROUP_SIZE` (e.g. 64), or recalibrate with more samples. |
| Slower than BF16 | Confirm `--sglang-cuda-graph-bs` covers your batch sizes. |

## Pairs with

* [R3](miles-router.md). Keeps MoE routing stable across the quantized forward.
* [P2P weight transfer](p2p-weight-transfer.md). INT4 weights are 4× smaller,
  so weight sync transfers less data.
* [Speculative decoding](speculative-decoding.md). Compounds for end-to-end
  rollout speedup.

## When QAT is not appropriate

* The model fits comfortably without it.
* The model architecture is still in development; introduce QAT after a BF16
  baseline.
* Tasks that are highly precision-sensitive (some math and safety eval suites).
