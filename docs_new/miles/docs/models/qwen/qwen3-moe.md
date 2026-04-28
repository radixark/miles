---
title: Qwen3 MoE
description: Launch recipes for Qwen3-30B-A3B (single node) and Qwen3-235B-A22B (multi-node).
---

# Qwen3 MoE

Two MoE sizes ship with miles: `Qwen3-30B-A3B` (single-node, Python launcher) and `Qwen3-235B-A22B` (multi-node, bash launcher with FP8 rollout). All values below come straight from `scripts/`.

## Variants

| Model | Active / Total | HF ID | Model config |
|---|---|---|---|
| Qwen3-30B-A3B | 3 B / 30 B | `Qwen/Qwen3-30B-A3B` | `scripts/models/qwen3-30B-A3B.sh` |
| Qwen3-235B-A22B | 22 B / 235 B | `Qwen/Qwen3-235B-A22B` | `scripts/models/qwen3-235B-A22B.sh` |

## Quick start

The reference launcher is `scripts/run_qwen3_30b_a3b.py`. It downloads the HF checkpoint, `zhuzilin/dapo-math-17k`, and `zhuzilin/aime-2024`, converts to Megatron `torch_dist`, and submits the training job.

```bash
cd /root/miles
python scripts/run_qwen3_30b_a3b.py
```

If you'd rather drive the steps by hand:

```bash
hf download Qwen/Qwen3-30B-A3B --local-dir /root/models/Qwen3-30B-A3B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/datasets/aime-2024

source scripts/models/qwen3-30B-A3B.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/models/Qwen3-30B-A3B \
   --save          /root/models/Qwen3-30B-A3B_torch_dist
```


## Cross-node example (Qwen3-235B-A22B, 8 nodes × 8 GPU)

The bash launcher requires `BASE_FOLDER` (a shared FS reachable from every node) and `MASTER_ADDR`. It defaults to the **FP8** HF checkpoint path:

```bash
export BASE_FOLDER=<shared FS path>
export MASTER_ADDR=<head node IP>

bash scripts/run-qwen3-235B-A22B.sh
```

Expected paths under `$BASE_FOLDER` (from `run-qwen3-235B-A22B.sh:41–69`):

```
Qwen3-235B-A22B-FP8/                      --hf-checkpoint
Qwen3-235B-A22B_torch_dist/               --ref-load
Qwen3-235B-A22B_miles/                    --load / --save
dapo-math-17k/dapo-math-17k.jsonl
aime-2024/aime-2024.jsonl
```

The script fans out to workers via `ssh` over `/root/mpi_rack_hostfile`.

## Launch scripts

| Script | GPUs | Notes |
|---|---|---|
| `scripts/run_qwen3_30b_a3b.py` | 8 (1 node) | Canonical Python launcher; supports FP8 / MXFP8 / INT4 rollout, Blackwell hardware, Megatron-bridge mode, MIS |
| `scripts/run-qwen3-235B-A22B.sh` | 64 (8 nodes × 8) | Multi-node, FP8 HF checkpoint, GSPO advantage, DeepEP `auto`, CPU Adam |
| `scripts/run-qwen3-235B-A22B-sft.sh` | 32 (4 nodes × 8) | SFT on `${BASE_FOLDER}/openhermes2_5.parquet` |

## Parallelism

| Script | Backend | TP | PP | CP | EP | expert-TP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|---|---|
| `run_qwen3_30b_a3b.py` (H100, 1 node) | Megatron | 4 | 1 | 1 | 8 | 1 | 32768 | 8 |
| `run-qwen3-235B-A22B.sh` | Megatron | 4 | 4 | 2 | 16 | 1 | 16384 | 64 |
| `run-qwen3-235B-A22B-sft.sh` | Megatron | 4 | 1 | 1 | 32 | 1 | 9216 | 32 |

`run-qwen3-235B-A22B.sh` additionally sets `--decoder-last-pipeline-num-layers 22` to balance the 235 B layer count across PP=4.

## SGLang flags

`run_qwen3_30b_a3b.py` (H100, 1 node, BF16 rollout):

```bash
--rollout-num-gpus-per-engine 8
--sglang-mem-fraction-static 0.7
--sglang-cuda-graph-max-bs 512
```

`run-qwen3-235B-A22B.sh`:

```bash
--rollout-num-gpus-per-engine 32
--sglang-mem-fraction-static 0.7
--sglang-enable-dp-attention
--sglang-dp-size 4
--sglang-ep-size 32
--sglang-enable-dp-lm-head
--sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
--sglang-moe-a2a-backend deepep
--sglang-deepep-mode auto
```

The 235 B script uses GSPO (`--advantage-estimator gspo`, `--eps-clip 4e-4`, `--use-kl-loss` commented out). The 30 B Python launcher uses GRPO with `--eps-clip 0.2 --eps-clip-high 0.28`.

## CPU Adam

Both `run_qwen3_30b_a3b.py` (H100, 1 node) and `run-qwen3-235B-A22B.sh` enable:

```bash
--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
```

`run_qwen3_30b_a3b.py` removes them when running on Blackwell (`B200/B300/GB200/GB300`) per the hardware match in the launcher.

## Pairs well with

- [FP8 & Low Precision](../../advanced/fp8-low-precision.md)
- [Miles Router (R3)](../../advanced/miles-router.md) — opt-in via `run_qwen3_30b_a3b.py --enable-mis` (TIS / RS) for routing-stability experiments; not on by default in either launcher.
