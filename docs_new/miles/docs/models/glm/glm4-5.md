---
title: GLM4.5
description: Launch recipes for GLM-4.5 (355B-A32B) — bash launcher and Python launcher.
---

# GLM4.5

## 1. Model Introduction

[GLM-4.5](https://huggingface.co/zai-org/GLM-4.5) is Zhipu AI's flagship MoE language model with advanced capabilities in reasoning, function calling, and multi-modal understanding.

**Key highlights:**

- **Sparse MoE architecture**: 355 B / 32 B-active for frontier runs and 106 B / 12 B-active for two-node experimentation.
- **Strong reasoning**: built-in step-by-step reasoning, with FP8 rollout supported on Blackwell hardware.
- **Speculative decoding**: EAGLE/MTP rollout supported by the bash launcher; the Python launcher exposes `--enable-mtp`.
- **R3 / MIS opt-in**: routing-stability extensions available behind a flag (`--enable-mis`) on the Python launcher.

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| GLM-4.5-355B-A32B | 32 B / 355 B | [zai-org/GLM-4.5](https://huggingface.co/zai-org/GLM-4.5) |
| GLM-4.5-Air (106B-A12B) | 12 B / 106 B | [zai-org/GLM-4.5-Air](https://huggingface.co/zai-org/GLM-4.5-Air) |

The 106B-A12B variant has no launcher under `scripts/`; the canonical recipe is [`examples/p2p_weight_transfer/GLM-4.5-Air.sh`](https://github.com/radixark/miles/blob/main/examples/p2p_weight_transfer/GLM-4.5-Air.sh) (8-node, P2P weight transfer).

## 3. Environment Setup

### 3.1 Required env vars

The bash launcher (`run-glm4.5-355B-A32B.sh`) requires:

```bash
export BASE_DIR=<shared FS path, reachable from every node>
# MASTER_ADDR is overridden to ${MLP_WORKER_0_HOST} inside the script (L144),
# so it comes from the cluster orchestrator.
```

The Python launcher (`run_glm45_355b_a32b.py`) reads no env vars — pass options via the Typer CLI.

### 3.2 Download model + datasets

```bash
hf download zai-org/GLM-4.5 --local-dir $BASE_DIR/GLM-4.5-355B-A32B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir $BASE_DIR/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir $BASE_DIR/rl_data/aime-2024
```

### 3.3 HF → Megatron `torch_dist` conversion

The bash launcher does **not** convert for you — produce `$BASE_DIR/GLM-4.5-355B-A32B_torch_dist/` ahead of time:

```bash
cd /root/miles
source scripts/models/glm4.5-355B-A32B.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint $BASE_DIR/GLM-4.5-355B-A32B \
   --save          $BASE_DIR/GLM-4.5-355B-A32B_torch_dist
```

The Python launcher automates the full flow (download → optional `tools/convert_hf_to_fp8.py` → `convert_checkpoint` → `rsync` to `model_local_dir` → submit).

## 4. Launch

### 4.1 Quick start

```bash
# Bash launcher (8 nodes × 8 GPU)
cd /root/miles
export BASE_DIR=...
bash scripts/run-glm4.5-355B-A32B.sh

# Python launcher (Blackwell hardware only — _execute_train asserts hardware != "H100")
python scripts/run_glm45_355b_a32b.py train --hardware GB300
```

### 4.2 Multi-node fan-out

`run-glm4.5-355B-A32B.sh` performs Ray fan-out internally via the `ssh` loop over `/root/mpi_rack_hostfile`.

## 5. Recipe Configuration

### 5.1 Parallelism

| Source | TP | PP | CP | EP | expert-TP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|---|
| `run-glm4.5-355B-A32B.sh` | 8 | 4 | 2 | 16 | 1 | 16384 | 64 (8 × 8) |
| `run_glm45_355b_a32b.py` (`num_nodes ≤ 4`, debug) | 4 | 1 | 1 | 4 | 1 | 16384 | ≤ 32 (≤ 4 × 8) |
| `run_glm45_355b_a32b.py` (`num_nodes == 8`) | 4 | 8 | 2 | 8 | 1 | 16384 | 64 (8 × 8) |

### 5.2 Algorithm

| Source | Advantage | Notable flags |
|---|---|---|
| `run-glm4.5-355B-A32B.sh` | GSPO | `--eps-clip 1e-4 --eps-clip-high 2e-4 --use-tis` |
| `run_glm45_355b_a32b.py` | GRPO | `--eps-clip 1e-4 --eps-clip-high 2e-4 --use-tis` |

Neither launcher enables `--use-rollout-routing-replay` by default. The Python launcher exposes `--enable-mis` (TIS/RS config) as an opt-in.

### 5.3 Rollout & SGLang

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 32
   --sglang-mem-fraction-static 0.7
   --sglang-enable-dp-attention
   --sglang-dp-size 4
   --sglang-ep-size 32
   --sglang-enable-dp-lm-head
   --sglang-moe-dense-tp-size 1

   # mtp / EAGLE
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 1
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 2
   --sglang-enable-draft-weights-cpu-backup
)
```

Megatron side: `--moe-token-dispatcher-type flex`, `--moe-enable-deepep`.

### 5.4 Optimizer

CPU Adam on:

```bash
--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
```

### 5.5 Notable quirks

- The bash launcher does not set `--load`/`--save` in `CKPT_ARGS` — `--load` defaults to the value of `--ref-load`.
- `run_glm45_355b_a32b.py` is Blackwell-only: `_execute_train` asserts `args.hardware != "H100"` (L100).

## 6. Pairs Well With

- [Low Precision RL](../../advanced/fp8-low-precision.md)
- [INT4 QAT](../../advanced/int4-qat.md)
- [Rollout Routing Replay (R3)](../../advanced/miles-router.md) — opt-in via `--enable-mis` on the Python launcher.
