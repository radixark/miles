---
title: Qwen3
description: Launch recipes for dense Qwen3 models (0.6 B – 32 B).
---

# Qwen3

Dense Qwen3 launch recipes, built around `scripts/run-qwen3-4B.sh` (canonical 8-GPU GRPO recipe on `zhuzilin/dapo-math-17k`) plus dedicated 32 B and AMD/FSDP/SFT variants.

## Variants

Model configs (Megatron `MODEL_ARGS`) live in `scripts/models/`. `run_qwen3_4b.py` (the Python launcher) currently knows about `Qwen3-0.6B`, `Qwen3-4B`, `Qwen3-4B-Base`, and `Qwen3-4B-Instruct-2507` — for other sizes you launch the bash scripts directly.

| Model | HF ID | Model config |
|---|---|---|
| Qwen3-0.6B | `Qwen/Qwen3-0.6B` | `scripts/models/qwen3-0.6B.sh` |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | `scripts/models/qwen3-1.7B.sh` |
| Qwen3-4B | `Qwen/Qwen3-4B` | `scripts/models/qwen3-4B.sh` |
| Qwen3-4B-Instruct-2507 | `Qwen/Qwen3-4B-Instruct-2507` | `scripts/models/qwen3-4B-Instruct-2507.sh` |
| Qwen3-8B | `Qwen/Qwen3-8B` | `scripts/models/qwen3-8B.sh` |
| Qwen3-14B | `Qwen/Qwen3-14B` | `scripts/models/qwen3-14B.sh` |
| Qwen3-32B | `Qwen/Qwen3-32B` | `scripts/models/qwen3-32B.sh` |

`scripts/models/qwen3-4B-Instruct-2507.sh` just sets `MODEL_ARGS_ROTARY_BASE=5000000` and re-sources `qwen3-4B.sh` — source this config when converting / launching the Instruct-2507 checkpoint.

## Paths the launcher expects

From `run-qwen3-4B.sh:29-61`:

```bash
CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-4B
   #--hf-checkpoint /root/Qwen3-4B-FP8
   --ref-load /root/Qwen3-4B_torch_dist
   --load /root/Qwen3-4B_miles/
   --save /root/Qwen3-4B_miles/
   --save-interval 20
)

ROLLOUT_ARGS=( --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl ... )
EVAL_ARGS=(   --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl ... )
```

The 32 B / 4xgpu / fsdp / sft / amd variants follow the same naming pattern (`/root/<model>{,_torch_dist,_miles/}`).

## Quick start

```bash
cd /root/miles

# 1. Download model + datasets (paths match what run-qwen3-4B.sh expects)
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/aime-2024

# 2. Convert HF → Megatron torch_dist
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen3-4B \
   --save          /root/Qwen3-4B_torch_dist

# 3. Launch GRPO
bash scripts/run-qwen3-4B.sh
```

The converter auto-derives PP from `WORLD_SIZE`; for larger models drive it with `torchrun --nproc-per-node 8`.

## Launch scripts


| Script | GPUs | Notes |
|---|---|---|
| `scripts/run-qwen3-4B.sh` | 8 | Canonical Megatron + SGLang GRPO recipe |
| `scripts/run-qwen3-4B_4xgpu.sh` | 4 | Half-node variant (`CUDA_VISIBLE_DEVICES=4,5,6,7`) |
| `scripts/run-qwen3-4B_4xgpu-radixtree.sh` | 4 | 4-GPU variant with `--use-miles-router` |
| `scripts/run-qwen3-4B-fsdp.sh` | 8 | FSDP backend, no Megatron conversion needed |
| `scripts/run-qwen3-4B-base-sft.sh` | 8 | SFT on `Qwen3-4B-Base` with `/root/openhermes2_5.parquet` |
| `scripts/amd/run-qwen3-4B-amd.sh` | `${NUM_GPUS}` | AMD ROCm variant |
| `scripts/run-qwen3-32B.sh` | 8 | 32 B dense, TP8 + CPU Adam |

## Parallelism

Only sizes with a real launch script in `scripts/` are listed.

| Script | TP | PP | CP | EP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|
| `run-qwen3-4B.sh` | 2 | 1 | 1 | 1 | 9216 | 8 |
| `run-qwen3-4B_4xgpu.sh` | 2 | 1 | 1 | 1 | 9216 | 4 |
| `run-qwen3-32B.sh` | 8 | 1 | 1 | 1 | 20480 | 8 |

`--sequence-parallel` is on whenever TP > 1. `run-qwen3-32B.sh` additionally enables `--optimizer-cpu-offload`, `--overlap-cpu-optimizer-d2h-h2d`, `--use-precision-aware-optimizer` and SGLang `--sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)`.

## FSDP variant

`scripts/run-qwen3-4B-fsdp.sh` runs the same GRPO recipe with the FSDP backend (`--train-backend fsdp`). It loads the HF checkpoint directly — no Megatron `torch_dist` conversion needed:

```bash
bash scripts/run-qwen3-4B-fsdp.sh
```

Notable extra flags it sets: `--update-weight-buffer-size 536870912`, `--gradient-checkpointing`, `--attn-implementation flash_attention_3`, SGLang attention backend `fa3`.

### BF16 train + FP8 inference

`run-qwen3-4B.sh:30-31` already ships a commented `--hf-checkpoint /root/Qwen3-4B-FP8` alternative — uncomment it (and `hf download Qwen/Qwen3-4B-FP8 --local-dir /root/Qwen3-4B-FP8`) to swap rollout to FP8 while keeping BF16 training. See [FP8 & Low Precision](../../advanced/fp8-low-precision.md).

## Tuning knobs

| Symptom | Fix |
|---|---|
| Train OOM | Drop `--max-tokens-per-gpu` |
| Rollout OOM | Drop `--sglang-mem-fraction-static` (defaults to `0.7` in 4B/32B scripts) |
| Reward stuck at 0 | Verify `--rm-type` matches the dataset's label format (`deepscaler` for dapo-math-17k) |
| Generations don't stop | Set `--rollout-stop-token-ids` explicitly |
