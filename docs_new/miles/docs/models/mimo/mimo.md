---
title: MiMo
description: Single-node GRPO + EAGLE speculative recipe with online MTP training.
---

# MiMo 7B

MiMo 7B is Xiaomi's dense RL model with a built-in MTP (Multi-Token Prediction) layer. The MTP layer makes MiMo a convenient target for EAGLE-style speculative rollout with online MTP-SFT — one recipe exercises both the training path and the rollout speedup. The reference launcher is `scripts/run-mimo-7B-rl-eagle.sh`: 1 node × 8 GPU, GRPO + EAGLE speculative rollout, online MTP-SFT.

## Variant

| Local checkpoint path | Model config |
|---|---|
| `/root/MiMo-7B-RL` | `scripts/models/mimo-7B-rl.sh` |

The launcher only references the local path; the HF repo isn't named in `scripts/`, so stage your checkpoint at `/root/MiMo-7B-RL` before launch.

## Paths the launcher expects

From `run-mimo-7B-rl-eagle.sh:30-62`:

```bash
CKPT_ARGS=(
   --hf-checkpoint /root/MiMo-7B-RL
   --ref-load /root/MiMo-7B-RL_torch_dist
   --load /root/MiMo-7B-RL-mtp_miles/
   --save /root/MiMo-7B-RL-mtp_miles/
   --save-interval 2000
)

ROLLOUT_ARGS=( --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
               --rm-type deepscaler ... )
EVAL_ARGS=(   --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl ... )
```

Note: `--save-interval` is **2000** here (much larger than the 20 used in most other launchers), and the actor save dir has a `-mtp` suffix.

## Quick start (1 node × 8 GPU)

```bash
cd /root/miles

hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/aime-2024
# Place the model checkpoint at /root/MiMo-7B-RL

source scripts/models/mimo-7B-rl.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/MiMo-7B-RL \
   --save          /root/MiMo-7B-RL_torch_dist

bash scripts/run-mimo-7B-rl-eagle.sh
```

## Parallelism

| TP | PP | CP | EP | expert-TP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|
| 2 | 1 | 1 | 1 | 1 | 9216 | 8 (1 × 8) |

`--use-dynamic-batch-size` + `--sequence-parallel` on; `--micro-batch-size` is commented out.

## SGLang flags (EAGLE speculative)

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7

   # for speculative decoding
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 4

   # sometimes flashinfer has IMA bugs. Use fa3 instead
   --sglang-attention-backend fa3
)
```

`--rollout-num-gpus-per-engine 1` is intentional — one SGLang engine per GPU. The fa3 attention backend is chosen because flashinfer was hitting IMAs (in-source comment).

## Online MTP training

```bash
SPEC_ARGS=(
   --enable-mtp-training
   --mtp-loss-scaling-factor 0.2
)
```

Plus the model-config side: `--mtp-num-layers 1` (lives in `MODEL_ARGS`, so you get it for free when you source the model config).

## Algorithm

GRPO baseline:

```bash
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)
```

CPU Adam is **not** enabled. R3 is **not** enabled.

## Pairs well with

- [Speculative Decoding](../../advanced/speculative-decoding.md)
