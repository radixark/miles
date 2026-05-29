---
title: Qwen3 MoE
description: Launch recipes for Qwen3-30B-A3B (single node) and Qwen3-235B-A22B (multi-node).
---

# Qwen3 MoE

## 1. Model Introduction

[Qwen3 MoE](https://github.com/QwenLM/Qwen3) is the Mixture-of-Experts branch of the Qwen3 series, available in two sizes: 30 B-A3B (single-node) and 235 B-A22B (multi-node).

**Key highlights:**

- **Sparse MoE architecture**: 30 B / 3 B-active and 235 B / 22 B-active variants, scaling capacity without proportional compute cost.
- **Strong reasoning and coding**: shares the Qwen3 generation's improvements in instruction following, math, and tool usage.
- **Long-context capability**: 256 K-token context inherited from the Qwen3 series.
- **Flexible scaling**: 30 B fits a single 8-GPU node; 235 B is the canonical multi-node target with FP8 rollout.

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| Qwen3-30B-A3B | 3 B / 30 B | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Qwen3-235B-A22B | 22 B / 235 B | [Qwen/Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) |

## 3. Environment Setup

### 3.1 Required env vars

The 235 B bash launcher requires:

```bash
export BASE_FOLDER=<shared FS path, reachable from every node>
export MASTER_ADDR=<head node IP>
```

The 30 B Python launcher reads no env vars — pass options via the Typer CLI.

### 3.2 Download model + datasets

```bash
# 30 B (single node)
hf download Qwen/Qwen3-30B-A3B --local-dir /root/models/Qwen3-30B-A3B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/datasets/aime-2024

# 235 B (multi-node, FP8 by default)
hf download Qwen/Qwen3-235B-A22B-FP8 --local-dir $BASE_FOLDER/Qwen3-235B-A22B-FP8
```

### 3.3 HF → Megatron `torch_dist` conversion

```bash
source scripts/models/qwen3-30B-A3B.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/models/Qwen3-30B-A3B \
   --save          /root/models/Qwen3-30B-A3B_torch_dist
```

Drive the conversion across more GPUs / nodes for the 235 B variant; the launcher reads `$BASE_FOLDER/Qwen3-235B-A22B_torch_dist/` as `--ref-load`.

## 4. Launch

### 4.1 Quick start

```bash
# 30 B (1 node × 8 GPU) — Python launcher handles download + conversion + submit
cd /root/miles
python scripts/run_qwen3_30b_a3b.py

# 235 B (8 nodes × 8 GPU)
export BASE_FOLDER=...; export MASTER_ADDR=...
bash scripts/run-qwen3-235B-A22B.sh
```

### 4.2 Multi-node fan-out

`run-qwen3-235B-A22B.sh` ssh-fans-out to workers via `/root/mpi_rack_hostfile` itself; you only need the env vars set on the head node. The 30 B launcher is single-node.

## 5. Recipe Configuration

### 5.1 Parallelism

| Script | Backend | TP | PP | CP | EP | expert-TP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|---|---|
| `run_qwen3_30b_a3b.py` (H100, 1 node) | Megatron | 4 | 1 | 1 | 8 | 1 | 32768 | 8 (1 × 8) |
| `run-qwen3-235B-A22B.sh` | Megatron | 4 | 4 | 2 | 16 | 1 | 16384 | 64 (8 × 8) |
| `run-qwen3-235B-A22B-sft.sh` | Megatron | 4 | 1 | 1 | 32 | 1 | 9216 | 32 (4 × 8) |

`run-qwen3-235B-A22B.sh` sets `--decoder-last-pipeline-num-layers 22` to balance the layer count across PP=4.

### 5.2 Algorithm

- **30 B Python launcher**: GRPO with `--eps-clip 0.2 --eps-clip-high 0.28`.
- **235 B bash launcher**: GSPO (`--advantage-estimator gspo`, `--eps-clip 4e-4`); `--use-kl-loss` is commented out.

### 5.3 Rollout & SGLang

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

### 5.4 Optimizer

Both `run_qwen3_30b_a3b.py` (H100, 1 node) and `run-qwen3-235B-A22B.sh` enable CPU Adam:

```bash
--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
```

`run_qwen3_30b_a3b.py` removes them when running on Blackwell (`B200/B300/GB200/GB300`) per the hardware match in the launcher.

### 5.5 Notable quirks

- **30 B Python launcher** supports FP8 / MXFP8 / INT4 rollout, Blackwell hardware, Megatron-bridge mode, and MIS via Typer flags.
- **235 B defaults to FP8 HF checkpoint** — the BF16 directory is available as a commented alternative in `CKPT_ARGS`.
- **R3 not on by default**; opt-in via `run_qwen3_30b_a3b.py --enable-mis` (TIS / RS) for routing-stability experiments.

## 6. Pairs Well With

- [Low Precision RL](../../advanced/fp8-low-precision.md)
- [Rollout Routing Replay (R3)](../../advanced/miles-router.md)
