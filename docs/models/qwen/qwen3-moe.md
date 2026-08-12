---
title: Qwen3 MoE
description: Launch recipes for Qwen3-30B-A3B (single node) and Qwen3-235B-A22B (multi-node).
---
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

The 235 B launcher requires:

```bash
export MASTER_ADDR=<head node IP>
```

Everything else is a Typer flag: `--model-dir` (default `/root/models`) for the checkpoints, `--data-dir` (default `/root/datasets`) for the datasets, `--output-dir` (default `/root/shared_data`) for what the run writes. Point them at a shared FS path reachable from every node. The 30 B launcher reads no env vars at all.

### 3.2 Download model + datasets

```bash
hf download Qwen/Qwen3-30B-A3B --local-dir /root/models/Qwen3-30B-A3B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/datasets/aime-2024

# 235 B (multi-node, FP8 by default)
hf download Qwen/Qwen3-235B-A22B-FP8 --local-dir /root/models/Qwen3-235B-A22B-FP8
```

### 3.3 HF → Megatron `torch_dist` conversion

```bash
MODEL_ARGS_LINE="$(python3 miles/utils/external_utils/model_args_utils.py qwen3-30B-A3B)" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/models/Qwen3-30B-A3B \
   --save          /root/models/Qwen3-30B-A3B_torch_dist
```

Drive the conversion across more GPUs / nodes for the 235 B variant; the launcher reads `<model-dir>/Qwen3-235B-A22B_torch_dist` as `--ref-load`.

## 4. Launch

### 4.1 Quick start

```bash
# 30 B (1 node × 8 GPU) — launcher handles download + conversion + submit
cd /root/miles
python scripts/run_qwen3_30b_a3b.py

# 235 B (8 actor nodes × 8 GPU + a 64-GPU rollout pool)
export MASTER_ADDR=...
python scripts/run_qwen3_235b_a22b.py
```

### 4.2 Multi-node fan-out

`run_qwen3_235b_a22b.py` ssh-fans-out to workers via `--ray-hostfile` (default `/root/mpi_rack_hostfile`) itself; you only need `MASTER_ADDR` set on the head node. Pass `--no-join-ray-workers` when the cluster is already joined. The 30 B launcher is single-node.

## 5. Recipe Configuration

### 5.1 Parallelism

| Launcher | Backend | TP | PP | CP | EP | expert-TP | `max_tokens_per_gpu` | Actor GPUs |
|---|---|---|---|---|---|---|---|---|
| `run_qwen3_30b_a3b.py` (H100, 1 node) | Megatron | 4 | 1 | 1 | 8 | 1 | 32768 | 8 (1 × 8) |
| `run_qwen3_235b_a22b.py` | Megatron | 4 | 4 | 2 | 16 | 1 | 16384 | 64 (8 × 8) |
| `run_qwen3_sft.py --model-name Qwen3-235B-A22B` | Megatron | 4 | 1 | 1 | 32 | 1 | 9216 | 32 (4 × 8) |

`run_qwen3_235b_a22b.py` sets `--decoder-last-pipeline-num-layers 22` to balance the layer count across PP=4. Its rollout is disaggregated, so it needs a further 64 GPUs (`--rollout-num-gpus`) on top of the actor pool.

### 5.2 Algorithm

- **30 B launcher**: GRPO with `--eps-clip 0.2 --eps-clip-high 0.28`.
- **235 B launcher**: GSPO (`--advantage-estimator gspo`, `--eps-clip 4e-4`); `--use-kl-loss` is not passed.

### 5.3 Rollout & SGLang

`run_qwen3_30b_a3b.py` (H100, 1 node, BF16 rollout):

```bash
--rollout-num-gpus-per-engine 8
--sglang-mem-fraction-static 0.7
--sglang-cuda-graph-max-bs 512
```

`run_qwen3_235b_a22b.py`:

```bash
--rollout-num-gpus-per-engine 32
--sglang-mem-fraction-static 0.7
--sglang-enable-dp-attention
--sglang-dp-size 4
--sglang-ep-size 32
--sglang-enable-dp-lm-head
--sglang-cuda-graph-bs 1 2 4 8 16 24 ... 256
--sglang-moe-a2a-backend deepep
--sglang-deepep-mode auto
```

### 5.4 Optimizer

Both `run_qwen3_30b_a3b.py` (H100, 1 node) and `run_qwen3_235b_a22b.py` enable CPU Adam:

```bash
--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
```

`run_qwen3_30b_a3b.py` removes them when running on Blackwell (`B200/B300/GB200/GB300`) per the hardware match in the launcher.

### 5.5 Notable quirks

- **30 B launcher** supports FP8 / MXFP8 / INT4 rollout, Blackwell hardware, Megatron-bridge mode, and MIS via Typer flags.
- **235 B defaults to the FP8 HF checkpoint** — pass `--no-rollout-fp8` to roll out from the BF16 directory instead.
- **R3 not on by default**; opt-in via `run_qwen3_30b_a3b.py --enable-mis` (TIS / RS) for routing-stability experiments.

## 6. Pairs Well With

- [Low Precision RL](/advanced/low-precision)
- [Rollout Routing Replay (R3)](/advanced/miles-router)
