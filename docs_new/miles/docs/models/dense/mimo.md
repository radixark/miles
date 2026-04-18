---
title: MiMo 7B
description: Launch recipe for MiMo 7B with MTP-based speculative decoding.
---

# MiMo 7B

MiMo 7B is a Xiaomi-authored dense model that ships with a built-in MTP layer, making
it a convenient target for speculative rollout with online MTP-SFT.

## Variant

| Model | Params | HF ID | Model config |
|---|---|---|---|
| MiMo-7B-RL | 7 B | `XiaomiMiMo/MiMo-7B-RL` | `scripts/models/mimo-7B-rl.sh` |

## Launch scripts

| Script | Purpose |
|---|---|
| `scripts/run-mimo-7B-rl-eagle.sh` | Canonical GRPO + EAGLE speculative rollout |

## Convert weights

```bash
cd /root/miles
source scripts/models/mimo-7B-rl.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /data/MiMo-7B-RL \
   --save          /data/MiMo-7B-RL_torch_dist
```

!!! warning "Pass `--mtp-num-layers 1` during conversion"
    The MTP weights must make it into the `torch_dist` checkpoint, or online MTP-SFT
    won't work.

## Parallelism

| TP | PP | CP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|
| 2 | 1 | 1 | 8192 | 8 |

## Launch

```bash
bash scripts/run-mimo-7B-rl-eagle.sh
```

## Speculative decoding flags

The script enables EAGLE-style speculative decoding and online MTP training:

```bash
SGLANG_ARGS+=(
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 4
   --sglang-enable-draft-weights-cpu-backup
)

PERF_ARGS+=(
   --mtp-num-layers 1
   --enable-mtp-training
   --mtp-loss-scaling-factor 0.2
)
```

See [Speculative Decoding](../../advanced/speculative-decoding.md) for what each flag
does.

## Tuning

| Symptom | Fix |
|---|---|
| `spec/draft_acceptance_rate` falling | Increase `--mtp-loss-scaling-factor` |
| MTP loss dominating policy loss | Decrease `--mtp-loss-scaling-factor` |
| Rollout no faster than baseline | Confirm `--sglang-speculative-num-draft-tokens` is non-trivial |
