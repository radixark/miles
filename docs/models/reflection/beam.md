---
title: Beam 1.0
sidebarTitle: Beam 1.0
description: Launch recipes for Reflection's Beam 1.0 (501B MoE, 23B active) — full-parameter BF16 GRPO on 96 or 64 H200 GPUs, and LoRA on 32.
---
The complete Beam RL implementation is open on the Miles
[`beam-image`](https://github.com/radixark/miles/tree/beam-image) branch. The scripts and image
below come from that branch.

## 1. Model Introduction

[Beam-1.0-501B-A23B](https://huggingface.co/reflection/Beam-1.0-501B-A23B) is Reflection's 501B-parameter MoE model (23B active) with a 1M-token context window, released in BF16, FP8 and NVFP4. miles trains the BF16 checkpoint, either the full model or a LoRA adapter on the frozen base, and loads it straight from HF through Megatron-Bridge, with no offline `torch_dist` conversion.

**Key highlights:**

- **Hybrid attention**: 43 sliding-window layers (window 512) with RoPE, and 9 full-attention layers without positional encoding, one every six layers plus the last.
- **SSMax**: the full-attention layers scale each query by a factor that grows with its position.
- **Gated GQA**: 48 query heads over 8 KV heads, QK-norm, and a sigmoid output gate with a per-head bias.
- **Sandwich norms**: every attention and MLP block is normalized on both its input and its output, and its residual add is scaled by depth.
- **MoE**: 256 routed experts with grouped top-8 sigmoid routing and no shared expert; the first 3 layers are dense.
- **FP32 softcap and router**: the logits are soft-capped at 30 in FP32, and the router weights stay in FP32 in the checkpoint, the trainer and SGLang.

## 2. Supported Variants

| Model | Active / Total | Layers | HF ID |
|---|---|---|---|
| Beam-1.0-501B-A23B | 23 B / 501 B | 52 (3 dense + 49 MoE) | [reflection/Beam-1.0-501B-A23B](https://huggingface.co/reflection/Beam-1.0-501B-A23B) |

The Beam bridge loads only this BF16 checkpoint; the FP8 and NVFP4 releases are not supported for training.

## 3. Environment Setup

### 3.1 Download

`prepare` downloads the checkpoint into `--model-dir` and DAPO-Math-17K and AIME25 into `--data-dir`; every node must see both at the same path. `run_beam_lora.py prepare` is identical, so one run serves both recipes:

```bash
python scripts/run_beam.py prepare --model-dir /root/models/Beam --data-dir /root/datasets
```

The checkpoint needs no Megatron conversion: the trainer reads it through the Beam bridge at startup. `prepare` also writes `aime-2025/aime-2025-answer-format.jsonl`, the AIME25 problems wrapped in the training prompts' answer-format instruction, which the evals read.

### 3.2 SGLang

The engine needs the Beam support on SGLang's [`beam-miles`](https://github.com/sgl-project/sglang/tree/beam-miles) branch.

The `radixark/miles:beam` image (multi-arch, built on `radixark/miles:v0.1.2` by `docker/Dockerfile.beam`) carries it and the recipe code:

```bash
docker pull radixark/miles:beam
```

## 4. Launch

### 4.1 Full parameter, 12 nodes

`execute` runs on twelve 8×H200 nodes: eight train and four host two rollout engines. Each trainer node needs at least 2 TB free on node-local NVMe for the optimizer state, at `/scratch/beam-optimizer` by default (`--optimizer-disk-dir`):

```bash
MASTER_ADDR=<head ip> python scripts/run_beam.py execute --num-rollout 100 \
    --model-dir /root/models/Beam --data-dir /root/datasets \
    --output-dir /root/shared_data/beam-run
```

The run evaluates on AIME25 every five rollouts and saves a checkpoint every twenty.

### 4.2 Full parameter, colocated, 8 nodes

`execute-colocate` takes the same arguments on eight 8×H200 nodes. Every GPU trains and also hosts one of four rollout engines, which receive the new weights over CUDA IPC. Each node's NVMe holds both the optimizer state and the paused trainer:

```bash
MASTER_ADDR=<head ip> python scripts/run_beam.py execute-colocate --num-rollout 100 \
    --model-dir /root/models/Beam --data-dir /root/datasets \
    --output-dir /root/shared_data/beam-colocate-run
```

### 4.3 LoRA, 4 nodes

`run_beam_lora.py` runs on four 8×H200 nodes: two train and two host one rollout engine. It needs no NVMe, since the adapter's optimizer state stays on the GPUs:

```bash
MASTER_ADDR=<head ip> python scripts/run_beam_lora.py execute --num-rollout 100 \
    --model-dir /root/models/Beam --data-dir /root/datasets \
    --output-dir /root/shared_data/beam-lora-run
```

### 4.4 Resume

Rerun the same command with `--resume`, on the same topology and output directory:

- `run_beam.py` restores the latest checkpoint and the dataset's sampling position. **`--num-rollout` is the total count**, including completed rollouts.
- `run_beam_lora.py` restores the latest adapter with its optimizer and LR scheduler from `<output-dir>/iter_*/adapter`. Rollout ids and the dataset position restart at 0.

## 5. Recipe Configuration

### 5.1 Parallelism

| Launcher | TP | PP | CP | EP | expert-TP | GPUs | rollout engines |
|---|---|---|---|---|---|---|---|
| `run_beam.py execute` | 8 | 1 | 1 | 8 | 8 | 96 (12 × 8): 64 train, 32 rollout | 2 × TP16, `--sglang-mem-fraction-static 0.85` |
| `run_beam.py execute-colocate` | 8 | 1 | 1 | 8 | 8 | 64 (8 × 8), shared | 4 × TP16, `--sglang-mem-fraction-static 0.7` |
| `run_beam_lora.py execute` | 8 | 1 | 1 | 16 | 1 | 32 (4 × 8): 16 train, 16 rollout | 1 × TP16, `--sglang-mem-fraction-static 0.85` |

`--sequence-parallel` and full recompute are on in every recipe. Core attention runs unfused on padded `bshd` micro-batches of one sample, up to 5,120 tokens (1,024 prompt + 4,096 response).

### 5.2 Algorithm

GRPO on DAPO-Math-17K with `--eps-clip 0.2 --eps-clip-high 0.28 --eps-clip-c 10`, per-token loss and `--use-tis`; no KL loss and no entropy bonus. Each rollout samples 8 responses for each of 32 prompts at temperature 1.0 with top-p and top-k off, and its 256 responses feed one optimizer step.

The reward (`examples/beam/reward.py`) checks the `\boxed{}` answer against the label and adds DAPO's soft overlong penalty over the last fifth of the response budget. AIME25 evaluation draws 8 samples per problem and reports plain accuracy.

`execute` and the LoRA recipe run `train_async.py`, which generates the next rollout while the current one trains, so responses are one update behind; `--no-async-training` runs them in turn. `execute-colocate` always trains synchronously. Rollout routing replay is off in `run_beam.py` (`--use-rollout-routing-replay` turns it on with the Miles router) and on in `run_beam_lora.py`.

### 5.3 Optimizer

**Full parameter**: Adam at a constant `--lr 1e-6`, no weight decay, gradient clip 1.0, distributed optimizer. The FP32 master weights and BF16 Adam moments live on node-local NVMe and stream through the GPU at each step:

```bash
--offload-train-disk-dir /scratch/beam-optimizer
--offload-train-disk-chunk-mb 256
--stream-optimizer-state-to-disk
--stream-optimizer-state-moment-dtype bf16
--grad-reduce-in-bf16
```

`execute-colocate` adds `--offload-train-target disk`: while the engines generate, the paused trainer is written to the same NVMe.

**LoRA**: rank 16, alpha 32 on every linear layer — attention q/k/v/o and the output gate, the dense MLPs, and the routed experts with per-expert factors. The router, embeddings and LM head stay frozen. The LR peaks at 1e-4 after 10 warmup steps and decays (cosine) to 1e-5 by step 400: at a constant 1e-5 the zero-initialized LoRA B barely moves, and a constant 2e-4 learns fast but turns unstable after about 100 steps. The adapter has ~4.7B parameters, and its optimizer state (~4.6 GB per GPU) stays on the GPUs: the NVMe-streamed optimizer does not support LoRA.

### 5.4 Rollout

The engines serve BF16 with `--sglang-attention-backend triton --sglang-moe-runner-backend triton` and chunked prefill 8192. `--sglang-max-running-requests` and `--sglang-cuda-graph-max-bs-decode` are each engine's share of the 256 responses. The trainer sends new weights after every rollout; the LoRA recipe sends only the adapter (~8.7 GiB in BF16).

### 5.5 Notable quirks

- **AIME25 prompts carry the answer-format instruction**: the grader requires `\boxed{}`, and on bare AIME problems the policy drops it as its final message gets shorter. `prepare` writes the wrapped file the evals read.
- **Expert-TP 1 for LoRA**: expert LoRA under expert-TP > 1 is wrong in Megatron-Bridge (forward, backward and export), so the LoRA recipe shards experts by EP only.
- **FP32 router**: Megatron casts the router weight to the parameter dtype; the Beam plugin keeps it in FP32, as the checkpoint stores it and SGLang serves it.

## 6. Pairs Well With

- [LoRA Training and Serving](/advanced/lora)
- [Rollout Routing Replay (R3)](/advanced/miles-router)
- [Disk Offload](/advanced/disk-offload)
