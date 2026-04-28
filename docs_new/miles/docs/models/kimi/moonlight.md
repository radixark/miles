---
title: Moonlight
description: Single-node MoE recipe (8 GPU) — DAPO-style dynamic sampling and CPU Adam on by default.
---

# Moonlight

Moonlight is Moonshot's compact MoE — 16 B total, 3 B active — and a useful single-node test target for MoE RL code changes before scaling to Kimi K2. The whole recipe fits on 8× H100. The reference launcher is `scripts/run-moonlight-16B-A3B.sh`: 1 node × 8 GPU, MLA + MoE, DAPO-style dynamic sampling and CPU Adam are on by default.

## Architecture

| Property | Value |
|---|---|
| Layers | 27 (1 dense + 26 MoE) |
| Hidden / FFN (dense) | 2048 / 11264 |
| Heads | 16 |
| Routed experts | 64, top-6 |
| Shared experts | 2, MoE FFN hidden 1408 |
| Attention | MLA (`--multi-latent-attention`, `--kv-lora-rank 512`, `--kv-channels 128`) |
| Vocab | 163840 |
| RoPE base | 50000 |
| Router | topk-scaling 2.446 |

## Variant

| Model | Local checkpoint path | Model config |
|---|---|---|
| Moonlight-16B-A3B | `/root/Moonlight-16B-A3B` | `scripts/models/moonlight.sh` |

The launcher only references the local path; the HF repo isn't named in `scripts/`, so stage your checkpoint at `/root/Moonlight-16B-A3B` before launch.

## Paths the launcher expects

From `run-moonlight-16B-A3B.sh:30-65`:

```bash
CKPT_ARGS=(
   --hf-checkpoint /root/Moonlight-16B-A3B
   --ref-load /root/Moonlight-16B-A3B_torch_dist
   --load /root/Moonlight-16B-A3B_miles/
   --save /root/Moonlight-16B-A3B_miles/
   --save-interval 20
)

ROLLOUT_ARGS=( --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
               --rm-type math ... )
EVAL_ARGS=(   --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl ... )
```

## Quick start (1 node × 8 GPU)

```bash
cd /root/miles

hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/aime-2024
# Place the model checkpoint at /root/Moonlight-16B-A3B

source scripts/models/moonlight.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Moonlight-16B-A3B \
   --save          /root/Moonlight-16B-A3B_torch_dist

bash scripts/run-moonlight-16B-A3B.sh
```

## Parallelism

| TP | PP | CP | EP | expert-TP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|
| 4 | 1 | 1 | 8 | 1 | 8192 | 8 (1 × 8) |

## Rollout args

```bash
ROLLOUT_ARGS=(
   --rm-type math
   --num-rollout 3000
   --rollout-batch-size 128
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 1

   --over-sampling-batch-size 256
   --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

   --num-steps-per-rollout 4
   --balance-data
)
```

## SGLang flags

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)
```

## Always-on extras

- **CPU Adam** is enabled by default: `--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer`.
- **Megatron-side DeepEP**: `--moe-enable-deepep --moe-token-dispatcher-type flex`.
- `--attention-backend flash` is **commented out** in this script (script comment: "need to comment this when using model with MLA").

GRPO with `--eps-clip 0.2 --eps-clip-high 0.28 --use-kl-loss --kl-loss-coef 0.00`. R3 is **not** enabled.

## Pairs well with

- [Miles Router (R3)](../../advanced/miles-router.md)
- [FP8 & Low Precision](../../advanced/fp8-low-precision.md)
