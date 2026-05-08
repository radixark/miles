---
title: SFT on OpenHermes
description: Plain supervised fine-tuning of Qwen3-4B-Base on the OpenHermes-2.5 dataset.
---

# SFT on OpenHermes

**What you'll learn:** how to use Miles for plain supervised fine-tuning. No RL, no
rollout, no reward — just data → loss → optimizer.

Why use Miles for SFT? Two reasons:

1. **Same launch convention as your RL run** — one config, one Ray cluster.
2. **Async data prefetching** — the SFT loop reuses the rollout machinery to overlap
   data loading with training.

## Prerequisites

* You completed the [Qwen3-4B](../models/qwen/qwen3.md) recipe (we reuse the
  conversion).
* ~50 GB free disk for OpenHermes-2.5.

## Quick start

### 1. Convert Qwen3-4B-Base

If you don't already have it:

```bash
hf download Qwen/Qwen3-4B-Base --local-dir /root/Qwen3-4B-Base

cd /root/miles
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen3-4B-Base \
   --save           /root/Qwen3-4B-Base_torch_dist
```

### 2. Prepare the dataset

OpenHermes ships in a custom shape. Convert to OpenAI messages format:

```python
from datasets import load_dataset

ds = load_dataset("teknium/OpenHermes-2.5")["train"]

def convert(sample):
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    return {
        "messages": [
            {"role": role_map[turn["from"]], "content": turn["value"]}
            for turn in sample["conversations"]
        ]
    }

ds = ds.map(convert)
ds.to_parquet("/root/openhermes2_5.parquet")
```

### 3. Run

```bash
bash scripts/run-qwen3-4B-base-sft.sh
```

## What changes vs. the GRPO recipe

Compare to [run-qwen3-4B.sh](../models/qwen/qwen3.md). The deltas:

```diff
- python3 train.py
+ python3 train_async.py        # async for data prefetch

- ROLLOUT_ARGS=( ... GRPO knobs, n-samples-per-prompt, ... )
+ SFT_ARGS=(
+    --rollout-function-path miles.rollout.sft_rollout.generate_rollout
+    --prompt-data /root/openhermes2_5.parquet
+    --input-key messages
+    --rollout-shuffle
+    --num-epoch 3
+    --rollout-batch-size 128
+    --global-batch-size 128
+
+    --loss-type sft_loss
+    --calculate-per-token-loss
+    --disable-compute-advantages-and-returns
+    --debug-train-only
+ )

- GRPO_ARGS=( ... )            # removed entirely
- SGLANG_ARGS=( ... )          # removed — no inference needed
```

## Why each flag

| Flag | Why |
|---|---|
| `--rollout-function-path miles.rollout.sft_rollout.generate_rollout` | Read from disk instead of generating |
| `--rollout-batch-size = --global-batch-size` | One batch read = one optimizer step |
| **No** `--n-samples-per-prompt` | SFT has one target per input |
| `--loss-type sft_loss` | Cross-entropy instead of policy-gradient |
| `--calculate-per-token-loss` | Standard SFT averages over unmasked tokens |
| `--disable-compute-advantages-and-returns` | No advantage / return needed |
| `--debug-train-only` | Skip SGLang init (we don't need rollout) |
| `train_async.py` | Async data prefetch overlaps load with train |

## What to watch

```text
sft/loss                       decreasing
sft/per_token_loss             decreasing (mirrors loss when using per-token)
sft/tokens_seen                steadily increasing
sft/epoch                      0 → num_epoch
data/prefetch_queue_depth      > 0 (else loader is the bottleneck)
```

If `data/prefetch_queue_depth` stays at 0, your data loader is too slow — increase
worker count or use parquet (we already do).

## Tuning knobs

| Knob | Effect |
|---|---|
| `--num-epoch` | Total passes over dataset |
| `--rollout-batch-size` | Bigger = better GPU utilization, more memory |
| `--max-tokens-per-gpu` | As always — push it up until OOM |
| `--lr` | SFT typically `1e-5` to `5e-5` (10× higher than RL) |
| `--lr-decay-style cosine --lr-warmup-iters 100` | Standard SFT schedule |

## Variations

### Mix datasets

Pass multiple `--prompt-data` entries:

```bash
SFT_ARGS+=(
   --prompt-data \
      hermes  /root/openhermes2_5.parquet \
      slimorca /data/slimorca.parquet
)
```

Per-source loss is logged separately.

### Continue with RL

After SFT, point the RL run at the SFT checkpoint:

```bash
CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-4B-Base
   --ref-load      /root/Qwen3-4B-Base_torch_dist     # original (anchor)
   --load          /root/Qwen3-4B-Base_sft/           # SFT output (start point)
   --save          /root/Qwen3-4B-Base_rl/
)
```

### LoRA SFT

Use the LoRA hooks (`--lora-rank 16`) to keep VRAM low when fine-tuning a
larger model. See `examples/lora/` in the repo.
