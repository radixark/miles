---
title: Beam 1.0
sidebarTitle: Beam 1.0
description: Launch recipes for Reflection's Beam 1.0 (501B MoE, 23B active) — full-parameter BF16 GRPO on 96 or 64 H200 GPUs, and LoRA on 32.
---
The complete Beam RL implementation is open at the Miles pull request:
[`radixark/miles#4149`](https://github.com/radixark/miles/pull/4149), branch
[`beam-support`](https://github.com/radixark/miles/tree/beam-support). The scripts and image
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

### 3.1 Checkpoint and data

There is no separate download step: the `full-train` commands in section 4 fetch the checkpoint into `--model-dir` and DAPO-Math-17K and AIME25 into `--data-dir`, then train. Every node must see both at the same path, and the checkpoint needs no Megatron conversion — the trainer reads it through the Beam bridge at startup.

The fetch also writes `aime-2025/aime-2025-answer-format.jsonl`, the AIME25 problems wrapped in the training prompts' answer-format instruction, which the evals read.

To stage the download on its own — to finish it before claiming the cluster, or to share one copy between the recipes — `prepare` does that part alone, and `run_beam_lora.py prepare` is identical.

### 3.2 SGLang

The engine needs the Beam support on SGLang's [`beam-miles`](https://github.com/sgl-project/sglang/tree/beam-miles) branch.

The `radixark/miles:beam` image (multi-arch, built on `radixark/miles:v0.1.2` by `docker/Dockerfile.beam`) carries it and the recipe code:

```bash
docker pull radixark/miles:beam
```

## 4. Launch

### 4.1 Bring up the Ray cluster

Run `radixark/miles:beam` on every node, with the checkpoint, the datasets and `--output-dir` on shared storage.

Pass `--join-ray-workers` and the launcher builds the cluster itself, ssh-ing every host in `--ray-hostfile` (default `/root/mpi_rack_hostfile`) into it before submitting the job; skip the rest of this section. Without that flag the launchers only submit, so bring the cluster up yourself:

```bash
# on the head node
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats
# on every other node
ray start --address=${MASTER_ADDR}:6379 --node-ip-address ${WORKER_IP} --num-gpus 8 --disable-usage-stats
```

`ray status` on the head must list 96, 64 or 32 GPUs before launching, for 4.2, 4.3 and 4.4 respectively.

### 4.2 Full parameter, 12 nodes

`full-train` downloads the checkpoint and datasets if they are not already in place, then runs on twelve 8×H200 nodes: eight train and four host two rollout engines. Each trainer node needs at least 2 TB free on node-local NVMe for the optimizer state, at `/scratch/beam-optimizer` by default (`--optimizer-disk-dir`):

```bash
MASTER_ADDR=<head ip> python scripts/run_beam.py full-train --num-rollout 100 \
    --model-dir /root/models/Beam --data-dir /root/datasets \
    --output-dir /root/shared_data/beam-run
```

Training is asynchronous (`train_async.py`): the next rollout generates while the current one trains, so its responses are one update stale; `--no-async-training` switches to synchronous `train.py`. Use `train` instead of `full-train` to skip the download and go straight to training. The run evaluates on AIME25 every five rollouts and saves a checkpoint every twenty.

### 4.3 Full parameter, colocated, 8 nodes

`--colocate` takes the same arguments on eight 8×H200 nodes. Every GPU trains and also hosts one of four rollout engines, which receive the new weights over CUDA IPC, and training is synchronous. Each node's NVMe holds both the optimizer state and the paused trainer:

```bash
MASTER_ADDR=<head ip> python scripts/run_beam.py full-train --colocate --num-rollout 100 \
    --model-dir /root/models/Beam --data-dir /root/datasets \
    --output-dir /root/shared_data/beam-colocate-run
```

### 4.4 LoRA, 4 nodes

`run_beam_lora.py` runs on four 8×H200 nodes: two train and two host one rollout engine, asynchronously like `train`. It needs no NVMe, since the adapter's optimizer state stays on the GPUs:

```bash
MASTER_ADDR=<head ip> python scripts/run_beam_lora.py full-train --num-rollout 100 \
    --model-dir /root/models/Beam --data-dir /root/datasets \
    --output-dir /root/shared_data/beam-lora-run
```

## 5. Recipe Configuration

### 5.1 Parallelism

| Launcher | TP | PP | CP | EP | expert-TP | GPUs | rollout engines |
|---|---|---|---|---|---|---|---|
| `run_beam.py full-train` | 8 | 1 | 1 | 8 | 8 | 96 (12 × 8): 64 train, 32 rollout | 2 × TP16, `--sglang-mem-fraction-static 0.85` |
| `run_beam.py full-train --colocate` | 8 | 1 | 1 | 8 | 8 | 64 (8 × 8), shared | 4 × TP16, `--sglang-mem-fraction-static 0.7` |
| `run_beam_lora.py full-train` | 8 | 1 | 1 | 16 | 1 | 32 (4 × 8): 16 train, 16 rollout | 1 × TP16, `--sglang-mem-fraction-static 0.85` |

`--sequence-parallel` and full recompute are on in every recipe. Core attention runs unfused on padded `bshd` micro-batches of one sample, up to 5,120 tokens (1,024 prompt + 4,096 response).

### 5.2 Algorithm

GRPO on DAPO-Math-17K with `--eps-clip 0.2 --eps-clip-high 0.28 --eps-clip-c 10`, per-token loss and `--use-tis`; no KL loss and no entropy bonus. Each rollout samples 8 responses for each of 32 prompts at temperature 1.0 with top-p and top-k off, and its 256 responses feed one optimizer step.

### 5.3 Optimizer

Adam at a constant `--lr 1e-6`, no weight decay, gradient clip 1.0, distributed optimizer. The FP32 master weights and BF16 Adam moments live on node-local NVMe and stream through the GPU at each step:

```bash
--offload-train-disk-dir /scratch/beam-optimizer
--offload-train-disk-chunk-mb 256
--stream-optimizer-state-to-disk
--stream-optimizer-state-moment-dtype bf16
--grad-reduce-in-bf16
```

`--colocate` adds `--offload-train-target disk`: while the engines generate, the paused trainer is written to the same NVMe.

### 5.4 Rollout

The engines serve BF16 with Triton attention and MoE. Each engine takes its share of the 256 responses, and the trainer sends new weights after every rollout:

```bash
--sglang-dtype bfloat16
--sglang-attention-backend triton
--sglang-moe-runner-backend triton
--sglang-chunked-prefill-size 8192
--sglang-max-running-requests 128      # 256 / engines: 128, colocated 64, LoRA 256
--sglang-cuda-graph-max-bs-decode 128  # same as max-running-requests
--update-weight-transfer-mode broadcast
--update-weights-interval 1
```

### 5.5 LoRA

`run_beam_lora.py` keeps the same data, reward, algorithm and rollout, and trains an adapter on every linear layer (attention q/k/v/o and the output gate, the dense MLPs, and each routed expert) while the router, embeddings and LM head stay frozen. What changes:

```bash
--lora-rank 16 --lora-alpha 32 --lora-dropout 0.0
--expert-model-parallel-size 16
--expert-tensor-parallel-size 1        # expert LoRA is wrong under expert-TP > 1 in Megatron-Bridge
--lr 1e-4 --min-lr 1e-5 --lr-decay-style cosine --lr-warmup-iters 10 --lr-decay-iters 400
--use-rollout-routing-replay --use-miles-router
```

The adapter's optimizer state stays on the GPUs, so the NVMe flags of 5.3 are dropped, and each weight update sends only the adapter (~8.7 GiB).

## 6. Pairs Well With

- [LoRA Training and Serving](/advanced/lora)
- [Rollout Routing Replay (R3)](/advanced/miles-router)
- [Disk Offload](/advanced/disk-offload)
