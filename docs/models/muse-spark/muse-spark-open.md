---
title: Muse Spark Open
description: Launch recipe for Muse Spark Open (816B omni MoE) — full-parameter GRPO on 64 H200 or 32 B300 GPUs, on math or on image-grounded math.
---
## 1. Model Introduction

[Muse Spark Open](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B) is an 816 B omni MoE model (42 B active) released in BF16. miles trains its 62-layer language model natively in Megatron.

**Key highlights:**

- **Hybrid attention**: 46 sliding-window layers (window 2048, RoPE) and 16 full-attention layers with no positional encoding; 128 query heads, 16 KV heads, head dim 64.
- **MoE in every layer**: 256 routed experts, top-8, no shared experts.
- **Gated residuals**: a learned per-channel gate mixes the residual stream and each post-normalized sublayer output.
- **Scaled embeddings and logits**: embeddings RMS-normalized and scaled, logits scaled and soft-capped at 20.
- **Frozen towers**: the vision and audio towers stay in the checkpoint but are not trained.

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| Muse Spark Open | 42 B / 816 B (62 layers) | [meta-models/Muse-Spark-Open-816B-A42B](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B) |

## 3. Environment Setup

### 3.1 Image and storage

```bash
docker pull radixark/miles:muse-spark-open
```

Run it on every node — 8 × 8 H200 or 4 × 8 B300, on InfiniBand. `--model-dir`, `--data-dir` and `--output-dir` need shared storage; `--offload-train-disk-dir` needs node-local NVMe with about 1.2 TB free for the streamed Adam state, 2.5 TB on B300.

### 3.2 Download + `torch_dist` conversion

The launcher's `prepare` step downloads the checkpoint and the dataset, then converts the checkpoint on all nodes into `<model-dir>/Muse-Spark-Open-816B-A42B_torch_dist` (about 1.6 TB), skipping a finished conversion. With `--vlm` it prepares `hiyouga/geometry3k` instead of `zhuzilin/dapo-math-17k`.

### 3.3 SGLang

The image carries the [`sglang-miles-muse-spark-open`](https://github.com/sgl-project/sglang/tree/sglang-miles-muse-spark-open) branch, which serves the omni checkpoint natively. Math reasoning is validated on H200 and B300, image-grounded math on H200 only; B200 is untested.

## 4. Launch

### 4.1 Quick start

Start one Ray cluster across the nodes, then launch from node 0:

```bash
# node 0
ray start --head --port 6379 --num-gpus 8
# other nodes
ray start --address <node-0-address>:6379 --num-gpus 8

cd /root/miles
# math reasoning on DAPO-Math-17k
MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_muse_spark_64xh200.py

# image-grounded math on Geometry3K
MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_muse_spark_64xh200.py \
  --vlm --use-dynamic-sampling-filter --reasoning-effort medium --eval-interval 5
```

`scripts/run_muse_spark_32xb300.py` is the same recipe on 4 × 8 B300 and takes the same options.

| Mode | GPUs | Rollout | Training | Weight sync | `train/train_rollout_logprob_abs_diff` |
|---|---|---|---|---|---|
| Math reasoning | 64 H200 | 2–5.5 min | 3–5.5 min | about 50 s | 0.007 |
| Image-grounded math | 64 H200 | 8–14 min | about 5 min | about 50 s | 0.013 |
| Math reasoning | 32 B300 | 1.7–3.5 min | 8.5–12.5 min (18.5 min at step 0) | about 90 s | 0.004–0.007 |

The log-prob column assumes [R3](/advanced/miles-router) (`--use-rollout-routing-replay`, on in this recipe), which replays the rollout's expert choices so routing noise cannot widen the gap. Reward starts at 0.65–0.75 and reaches 0.8–0.95 within 10 steps on math reasoning; every engine holds bit-identical weights after each sync.

### 4.2 Save and export

`--save-interval N` saves the weights, without optimizer state, to `<output-dir>/<run-id>/checkpoints`, and the launcher converts the last save into a Hugging Face checkpoint at `<output-dir>/<run-id>/hf`. Each save and each conversion takes about 1.6 TB. Save directories count steps from 0, so `iter_0000009` below is the save after step 10:

```bash
python tools/convert_muse_spark_torch_dist_to_hf.py \
  --input-dir <output-dir>/<run-id>/checkpoints/iter_0000009 \
  --hf-checkpoint /root/models/Muse-Spark-Open-816B-A42B \
  --output-dir <output-dir>/<run-id>/hf-step-10
```

## 5. Recipe Configuration

### 5.1 Parallelism

| Launcher | TP | PP | CP | EP | expert-TP | first/last stage layers | `max_tokens_per_gpu` |
|---|---|---|---|---|---|---|---|
| `run_muse_spark_64xh200.py` | 8 | 8 | 1 | 8 | 1 | 7 | 4096 |
| `run_muse_spark_32xb300.py` | 8 | 4 | 1 | 8 | 1 | 15 | 8192 |

`--sequence-parallel`, THD packing (`--use-dynamic-batch-size`) and full recomputation are on.

### 5.2 Algorithm

GRPO with `--eps-clip 0.2 --eps-clip-high 0.28`, no KL or entropy term, per-token loss, and R3 (`--use-rollout-routing-replay`). Each step samples 32 prompts × 16 responses of up to 4096 tokens (8192-token context) at temperature 1. `--rm-type math` rewards a matching final `\boxed{}` answer, and `--reasoning-effort` defaults to `xhigh`.

### 5.3 Rollout & SGLang

Rollout runs on 4 colocated engines, up to 64 concurrent requests each on H200 and 128 on B300:

```bash
--sglang-ep-size 16                # 16 GPUs per engine on H200; 8 (one node) on B300
--sglang-kv-cache-dtype bfloat16   # FP8 KV widens the rollout/training log-prob gap
```

### 5.4 Optimizer

Adam at `--lr 1e-6`, constant, with weight decay 0.1 and no gradient clipping. The FP32 main weights and moments do not fit the GPUs, so they stream through node-local NVMe:

```bash
--stream-optimizer-state-to-disk
--stream-optimizer-state-moment-dtype fp32
--offload-train-target disk
--offload-train-disk-chunk-mb 256
```

`--optimizer muon` switches to Muon (`dist_muon`), which keeps its state and the FP32 main weights in chunked CPU memory (`--chunked-optimizer-state-offload --chunked-muon-master-weights`). Megatron cannot save that state, so a Muon run also needs `--no-load-optim --no-save-optim`.

### 5.5 Image-grounded math (`--vlm`)

`--vlm` trains on Geometry3K. The vision tower stays frozen on the first pipeline stage, outside the optimizer and the checkpoints, and the weight sync re-sends it after every update. `--eval-interval N` scores the Geometry3K test split (601 problems, 4 samples each) and requires `--vlm`.

## 6. Pairs Well With

- [Disk Offload](/advanced/disk-offload)
- [Rollout Routing Replay (R3)](/advanced/miles-router)
