---
title: Qwen3-Next 80B-A3B
description: Launch recipes for Qwen3-Next-80B-A3B-Thinking on the Megatron backend.
---
## 1. Model Introduction

[Qwen3-Next](https://huggingface.co/collections/Qwen/qwen3-next) is Alibaba's next-generation Qwen architecture, swapping classical attention for a hybrid Gated DeltaNet + Full Attention design.

**Key highlights:**

- **Hybrid Attention**: combines Gated DeltaNet (linear attention) with Full Attention to handle context lengths up to 262 K tokens efficiently.
- **Highly Sparse MoE**: 80 B total / 3 B active per token — drastically reduces FLOPs per token without sacrificing model capacity.
- **Multi-Token Prediction (MTP)**: built-in MTP layer enables EAGLE-style speculative rollout out of the box.
- **HuggingFace-wrapped Megatron backend**: miles loads the `Qwen/Qwen3-Next-80B-A3B` HF module as a Megatron stage without re-implementing GDN from scratch.

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| Qwen3-Next-80B-A3B-Thinking | 3 B / 80 B | [Qwen/Qwen3-Next-80B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking) |

## 3. Environment Setup

### 3.1 Required env vars

```bash
export MASTER_ADDR=<head node IP>
```

`--topology 4node` needs `MASTER_ADDR` for the ray fan-out; `--topology single-node` needs nothing. Paths are Typer flags: `--model-dir` (default `/root/models`) for the staged checkpoint, `--data-dir` (default `/root/datasets`) for the datasets, `--output-dir` (default `/root/shared_data`) for what the run writes. On a multi-node run all three must be on a shared FS.

### 3.2 Download model + datasets

```bash
hf download Qwen/Qwen3-Next-80B-A3B-Thinking --local-dir /root/models/Qwen3-Next-80B-A3B-Thinking
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/datasets/aime-2024
```

### 3.3 HF → Megatron `torch_dist` conversion

```bash
cd /root/miles
MODEL_ARGS_LINE="$(python3 miles/utils/external_utils/model_args_utils.py qwen3-next-80B-A3B)" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/models/Qwen3-Next-80B-A3B-Thinking \
   --save          /root/models/Qwen3-Next-80B-A3B-Thinking_torch_dist
```

## 4. Launch

### 4.1 Quick start

```bash
cd /root/miles
export MASTER_ADDR=...
python scripts/run_qwen3_next_80b_a3b.py --topology 4node

# single 8-GPU node (6 GPUs train, 2 serve rollout)
python scripts/run_qwen3_next_80b_a3b.py --topology single-node
```

### 4.2 Multi-node fan-out

`--topology 4node` performs ssh fan-out internally over `/root/mpi_rack_hostfile` — set `MASTER_ADDR` on the head node and the launcher reaches out to the workers. Pass `--no-join-ray-workers` when the ray cluster is already complete. `--topology single-node` never fans out.

## 5. Recipe Configuration

### 5.1 Parallelism

Both layouts are one launcher, `scripts/run_qwen3_next_80b_a3b.py`, selected by `--topology`:

| `--topology` | Backend | TP | PP | CP | EP | expert-TP | `max_tokens_per_gpu` | Actor GPUs |
|---|---|---|---|---|---|---|---|---|
| `4node` | Megatron | 2 | 4 | 2 | 8 | 1 | 8192 | 32 (4 × 8) |
| `single-node` | Megatron | 1 | 6 | 1 | 1 | 1 | 2048 | 6 (1 × 6) |

`4node` colocates the rollout engines on the training GPUs; `single-node` dedicates the remaining 2 GPUs of the node to rollout, which also shrinks every batch dimension.

### 5.2 Algorithm

Both topologies use GSPO (`--advantage-estimator gspo --eps-clip 4e-4`); `--use-kl-loss` is not passed.

### 5.3 Rollout & SGLang

`4node` enables EAGLE speculative rollout:

```bash
--rollout-num-gpus-per-engine 8
--sglang-mem-fraction-static 0.8
--sglang-ep-size 8
--sglang-cuda-graph-bs 1 2 4 8 16 24 ... 128

--sglang-speculative-algorithm EAGLE
--sglang-speculative-num-steps 2
--sglang-speculative-eagle-topk 1
--sglang-speculative-num-draft-tokens 3
--sglang-enable-draft-weights-cpu-backup
--sglang-max-running-requests 512
```

`single-node` drops the EAGLE block and uses `--rollout-num-gpus-per-engine 2 --rollout-num-gpus 2 --sglang-mem-fraction-static 0.8 --sglang-ep-size 1`.

### 5.4 Optimizer

Both topologies enable CPU Adam:

```bash
--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
```

### 5.5 Notable quirks

- Gated DeltaNet (GDN) is loaded via the HuggingFace bridge; miles doesn't re-implement GDN in Megatron native code.

## 6. Pairs Well With

- [Backends Beyond Megatron](/advanced/architecture-support)
- [Rollout Routing Replay (R3)](/advanced/miles-router)
- [Speculative Decoding](/advanced/speculative-decoding)
