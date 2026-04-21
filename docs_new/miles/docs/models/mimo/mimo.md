---
title: MiMo 7B
description: Launch recipe for MiMo 7B — a dense model with a built-in MTP layer for speculative rollout.
---

# MiMo 7B

MiMo 7B is Xiaomi's dense RL model with a built-in MTP (Multi-Token Prediction) layer. The MTP layer makes MiMo a convenient target for EAGLE-style speculative rollout with online MTP-SFT — one recipe exercises both the training path and the rollout speedup.

## Variant

| Model | Params | HF ID | Model config |
|---|---|---|---|
| MiMo-7B-RL | 7 B | `XiaomiMiMo/MiMo-7B-RL` | `scripts/models/mimo-7B-rl.sh` |

## Quick start (8× H100)

```bash
cd /root/miles

# 1. Download weights
hf download XiaomiMiMo/MiMo-7B-RL --local-dir /root/MiMo-7B-RL

# 2. Convert HF → Megatron torch_dist — note --mtp-num-layers 1
source scripts/models/mimo-7B-rl.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/MiMo-7B-RL \
   --save          /root/MiMo-7B-RL_torch_dist \
   --mtp-num-layers 1

# 3. Launch with EAGLE speculative rollout + online MTP-SFT
bash scripts/run-mimo-7B-rl-eagle.sh
```

!!! warning "Don't skip `--mtp-num-layers 1`"
    The MTP weights must make it into the `torch_dist` checkpoint, or online MTP-SFT won't work — you'll see `spec/draft_acceptance_rate` stuck at zero and no rollout speedup.

## Expected signal

Two signals beyond the standard `loss=… reward=…`:

- `spec/draft_acceptance_rate` should climb above ~0.5 within the first few hundred steps as the draft (MTP layer) catches up with the policy. If it keeps falling, raise `--mtp-loss-scaling-factor`.
- Rollout throughput should measurably exceed a non-speculative baseline. If it doesn't, confirm `--sglang-speculative-num-draft-tokens` is non-trivial.

---

## Deep dive

### Launch scripts

| Script | Purpose |
|---|---|
| `scripts/run-mimo-7B-rl-eagle.sh` | Canonical GRPO + EAGLE speculative rollout |

### Parallelism

| TP | PP | CP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|
| 2 | 1 | 1 | 8192 | 8 |

### Speculative decoding flags

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

See [Speculative Decoding](../../advanced/speculative-decoding.md) for what each flag does.

### Tuning knobs

| Symptom | Fix |
|---|---|
| `spec/draft_acceptance_rate` falling | Increase `--mtp-loss-scaling-factor` |
| MTP loss dominating policy loss | Decrease `--mtp-loss-scaling-factor` |
| Rollout no faster than baseline | Confirm `--sglang-speculative-num-draft-tokens` is non-trivial |
