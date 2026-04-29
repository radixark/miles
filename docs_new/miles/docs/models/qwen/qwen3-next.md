---
title: Qwen3-Next 80B-A3B
description: Launch recipes for Qwen3-Next-80B-A3B-Thinking on Megatron and FSDP backends.
---

# Qwen3-Next 80B-A3B

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
export BASE_FOLDER=<shared FS path, must contain the staged checkpoint + datasets>
export MASTER_ADDR=<head node IP>
```

All three launchers (`run-qwen3-next-80B-A3B.sh`, `run-qwen3-next-80B-A3B-8gpus.sh`, `run-qwen3-next-80B-A3B-fsdp.sh`) hard-fail if these aren't set.

### 3.2 Download model + datasets

```bash
hf download Qwen/Qwen3-Next-80B-A3B-Thinking --local-dir $BASE_FOLDER/Qwen3-Next-80B-A3B-Thinking
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir $BASE_FOLDER/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir $BASE_FOLDER/aime-2024
```

### 3.3 HF → Megatron `torch_dist` conversion

```bash
cd /root/miles
source scripts/models/qwen3-next-80B-A3B.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint $BASE_FOLDER/Qwen3-Next-80B-A3B-Thinking \
   --save          $BASE_FOLDER/Qwen3-Next-80B-A3B-Thinking_torch_dist
```

The FSDP variant skips this step and loads HF directly.

## 4. Launch

### 4.1 Quick start

```bash
cd /root/miles
export BASE_FOLDER=...; export MASTER_ADDR=...
bash scripts/run-qwen3-next-80B-A3B.sh
```

### 4.2 Multi-node fan-out

`run-qwen3-next-80B-A3B.sh` performs ssh fan-out internally — set `BASE_FOLDER` / `MASTER_ADDR` on the head node and the launcher reaches out to the workers. The 8-GPU and FSDP variants are single-node.

## 5. Recipe Configuration

### 5.1 Parallelism

| Script | Backend | TP | PP | CP | EP | expert-TP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|---|---|
| `scripts/run-qwen3-next-80B-A3B.sh` | Megatron | 2 | 4 | 2 | 8 | 1 | 8192 | 32 (4 × 8) |

### 5.2 Algorithm

All three scripts use GSPO (`--advantage-estimator gspo --eps-clip 4e-4`); `--use-kl-loss` is commented out.

### 5.3 Rollout & SGLang

The canonical script enables EAGLE speculative rollout:

```bash
--rollout-num-gpus-per-engine 8
--sglang-mem-fraction-static 0.8
--sglang-ep-size 8
--sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 128)

# mtp / EAGLE
--sglang-speculative-algorithm EAGLE
--sglang-speculative-num-steps 2
--sglang-speculative-eagle-topk 1
--sglang-speculative-num-draft-tokens 3
--sglang-enable-draft-weights-cpu-backup
--sglang-max-running-requests 512
```

The 6-GPU and FSDP variants ship the EAGLE block commented out and use `--rollout-num-gpus-per-engine 2 --rollout-num-gpus 2 --sglang-mem-fraction-static 0.8 --sglang-ep-size 1`.

### 5.4 Optimizer

The Megatron variants enable CPU Adam:

```bash
--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
```

The FSDP variant leaves Adam on GPU.

### 5.5 Notable quirks

- Gated DeltaNet (GDN) is loaded via the HuggingFace bridge; miles doesn't re-implement GDN in Megatron native code.

## 6. Pairs Well With

- [Backends Beyond Megatron](../../advanced/architecture-support.md)
- [Rollout Routing Replay (R3)](../../advanced/miles-router.md)
- [Speculative Decoding](../../advanced/speculative-decoding.md)
