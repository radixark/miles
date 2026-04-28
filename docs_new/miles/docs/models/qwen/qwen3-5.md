---
title: Qwen3.5
description: Launch recipes for Qwen3.5-4B / 9B / 27B with attention-output-gate.
---

# Qwen3.5

Qwen3.5 dense uses the `miles_plugins.models.qwen3_5` spec — attention-output-gate, `--rotary-base 10000000`, `--rotary-percent 0.25`, vocab 248320 (see `scripts/models/qwen3.5-4B.sh`).

## Variants

The launch scripts read the checkpoint from a local path; HF IDs aren't referenced in `scripts/`, so just stage the weights at the path the script expects.

| Model | Local checkpoint path | Model config |
|---|---|---|
| Qwen3.5-4B | `/root/Qwen3.5-4B` | `scripts/models/qwen3.5-4B.sh` |
| Qwen3.5-9B | `/root/Qwen3.5-9B` | `scripts/models/qwen3.5-9B.sh` |
| Qwen3.5-27B | `/root/Qwen3.5-27B` | `scripts/models/qwen3.5-27B.sh` |

## Paths the launchers expect

`run-qwen3.5-{4B,9B,27B}.sh` all hard-code `/root/...` paths in `CKPT_ARGS` and `ROLLOUT_ARGS`. Stage these before launching (example shows the 4 B values from `run-qwen3.5-4B.sh:29-60`):

```bash
CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3.5-4B
   --ref-load /root/Qwen3.5-4B_torch_dist
   --load /root/Qwen3.5-4B_miles/
   --save /root/Qwen3.5-4B_miles/
   --save-interval 20
)

ROLLOUT_ARGS=( --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl ... )
EVAL_ARGS=(   --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl ... )
```

Swap `4B` → `9B` / `27B` for the other two scripts; everything else is identical (same dataset paths, same `--save-interval 20`).

## Quick start

```bash
cd /root/miles

# 1. Stage datasets at the paths the script expects
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/aime-2024
# Place the model checkpoint at /root/Qwen3.5-4B

# 2. Convert HF → Megatron torch_dist  (script then reads /root/Qwen3.5-4B_torch_dist)
source scripts/models/qwen3.5-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen3.5-4B \
   --save          /root/Qwen3.5-4B_torch_dist

# 3. Launch GRPO
bash scripts/run-qwen3.5-4B.sh
```

## Launch scripts

All three scripts are 1 node × 8 GPU.

| Script | TP | PP | CP | `max_tokens_per_gpu` | SGLang `mem-fraction-static` | CPU Adam |
|---|---|---|---|---|---|---|
| `scripts/run-qwen3.5-4B.sh` | 2 | 1 | 1 | 9216 | 0.7 | – |
| `scripts/run-qwen3.5-9B.sh` | 2 | 1 | 1 | 9216 | 0.6 | – |
| `scripts/run-qwen3.5-27B.sh` | 4 | 1 | 1 | 8192 | 0.5 | ✓ |

Only the 27 B script enables `--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer`.

## SGLang TP=1 workaround

All three scripts pin `--rollout-num-gpus-per-engine 1` with this comment in-source:

```
# Workaround: SGLang TP>1 produces garbage output for Qwen3.5
# (https://github.com/sgl-project/sglang/issues/21039)
# Fixed in sglang main, but miles uses 0.5.9 — keep TP=1 per engine for now.
```

If you bump the SGLang version, you can raise this back up.

## Architecture notes

From `scripts/models/qwen3.5-4B.sh` (and analogous configs for 9 B / 27 B):

- `--spec miles_plugins.models.qwen3_5 get_qwen3_5_spec` — attention-output gate, `A_log` parameter handling
- `--rotary-base 10000000`, `--rotary-percent 0.25`
- `--vocab-size 248320`
- `--apply-layernorm-1p`, `--qk-layernorm`, `--group-query-attention`
- `--attention-output-gate`

See [Backends Beyond Megatron](../../advanced/architecture-support.md) for how miles preserves FP32 parameters like `A_log` through Megatron's mixed-precision pipeline.
