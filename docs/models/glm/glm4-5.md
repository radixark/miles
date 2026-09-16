---
title: GLM4.5
description: Launch recipes for GLM-4.5 (355B-A32B) — the 8-node launcher and the Blackwell launcher.
---
## 1. Model Introduction

[GLM-4.5](https://huggingface.co/zai-org/GLM-4.5) is Zhipu AI's flagship MoE language model with advanced capabilities in reasoning, function calling, and multi-modal understanding.

**Key highlights:**

- **Sparse MoE architecture**: 355 B / 32 B-active for frontier runs and 106 B / 12 B-active for two-node experimentation.
- **Strong reasoning**: built-in step-by-step reasoning, with FP8 rollout supported on Blackwell hardware.
- **Speculative decoding**: EAGLE/MTP rollout is always on in the 8-node launcher; `run_glm45_355b_a32b.py` exposes `--enable-mtp`.
- **R3 / MIS opt-in**: routing-stability extensions available behind a flag (`--enable-mis`) on `run_glm45_355b_a32b.py`.

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| GLM-4.5-355B-A32B | 32 B / 355 B | [zai-org/GLM-4.5](https://huggingface.co/zai-org/GLM-4.5) |
| GLM-4.5-Air (106B-A12B) | 12 B / 106 B | [zai-org/GLM-4.5-Air](https://huggingface.co/zai-org/GLM-4.5-Air) |

The 106B-A12B variant has no launcher under `scripts/`; the canonical recipe is [`examples/infra_features/p2p_weight_transfer/GLM-4.5-Air.sh`](https://github.com/radixark/miles/blob/main/examples/infra_features/p2p_weight_transfer/GLM-4.5-Air.sh) (8-node, P2P weight transfer).

## 3. Environment Setup

### 3.1 Required env vars

The 8-node launcher (`run_glm45_355b_a32b_8node.py`) requires:

```bash
export MASTER_ADDR=<ray head IP, reachable from every node>
```

Paths are Typer flags: `--model-dir` (default `/root/models`), `--data-dir` (default `/root/datasets`), `--output-dir` (default `/root/shared_data`) — all three must live on a shared FS reachable from every node. `run_glm45_355b_a32b.py` reads no env vars either.

### 3.2 Download model + datasets

```bash
hf download zai-org/GLM-4.5 --local-dir /root/models/GLM-4.5-355B-A32B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/datasets/aime-2024
```

### 3.3 HF → Megatron `torch_dist` conversion

The 8-node launcher does **not** convert for you — produce `/root/models/GLM-4.5-355B-A32B_torch_dist/` ahead of time:

```bash
cd /root/miles
MODEL_ARGS_LINE="$(python3 miles/utils/external_utils/model_args_utils.py glm4.5-355B-A32B)" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/models/GLM-4.5-355B-A32B \
   --save          /root/models/GLM-4.5-355B-A32B_torch_dist
```

`run_glm45_355b_a32b.py` automates the full flow (download → optional `tools/convert_hf_to_fp8.py` → `convert_checkpoint` → `rsync` to `model_local_dir` → submit).

## 4. Launch

### 4.1 Quick start

```bash
# 8-node launcher (8 nodes × 8 GPU)
cd /root/miles
export MASTER_ADDR=...
python scripts/run_glm45_355b_a32b_8node.py

# Blackwell-only variant (_execute_train asserts hardware != "H100")
python scripts/run_glm45_355b_a32b.py train --hardware GB300
```

### 4.2 Multi-node fan-out

`run_glm45_355b_a32b_8node.py` performs Ray fan-out internally, ssh-ing every host of `--ray-hostfile` (default `/root/mpi_rack_hostfile`) into the cluster. Pass `--no-join-ray-workers` when the cluster is already joined.

## 5. Recipe Configuration

### 5.1 Parallelism

| Source | TP | PP | CP | EP | expert-TP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|---|
| `run_glm45_355b_a32b_8node.py` | 8 | 4 | 2 | 16 | 1 | 16384 | 64 (8 × 8) |
| `run_glm45_355b_a32b.py` (`num_nodes ≤ 4`, debug) | 4 | 1 | 1 | 4 | 1 | 16384 | ≤ 32 (≤ 4 × 8) |
| `run_glm45_355b_a32b.py` (`num_nodes == 8`) | 4 | 8 | 2 | 8 | 1 | 16384 | 64 (8 × 8) |

### 5.2 Algorithm

| Source | Advantage | Notable flags |
|---|---|---|
| `run_glm45_355b_a32b_8node.py` | GSPO | `--eps-clip 1e-4 --eps-clip-high 2e-4 --use-tis` |
| `run_glm45_355b_a32b.py` | GRPO | `--eps-clip 1e-4 --eps-clip-high 2e-4 --use-tis` |

Neither launcher enables `--use-rollout-routing-replay` by default. `run_glm45_355b_a32b.py` exposes `--enable-mis` (TIS/RS config) as an opt-in.

### 5.3 Rollout & SGLang

```bash
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

- The 8-node launcher passes neither `--load` nor `--save`, so `--load` defaults to the value of `--ref-load`. `--save` turns checkpointing to `<output-dir>/checkpoints` back on, every 20 steps.
- `run_glm45_355b_a32b.py` is Blackwell-only: `_execute_train` asserts `args.hardware != "H100"`.

## 6. Pairs Well With

- [Low Precision RL](/advanced/low-precision)
- [INT4 QAT](/advanced/int4-qat)
- [Rollout Routing Replay (R3)](/advanced/miles-router) — opt-in via `--enable-mis` on `run_glm45_355b_a32b.py`.
