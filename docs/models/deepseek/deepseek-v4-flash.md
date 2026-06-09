---
title: DeepSeek-V4 Flash
description: Launch recipe for DeepSeek-V4-Flash (284 B) — FP8 rollout / BF16 train, 8-node H200 (64 GPUs).
---

# DeepSeek-V4 Flash

DeepSeek V4 training tracking issue: [`radixark/miles#1046`](https://github.com/radixark/miles/issues/1046).

## 1. Model Introduction

[DeepSeek-V4-Flash](https://huggingface.co/sgl-project/DeepSeek-V4-Flash-FP8) is a 13 B-active / 284 B-total MoE model with a substantially different attention stack from V3/R1. The miles + Megatron-Core (`mcore`) integration is shipped together in the [`radixark/miles#1045`](https://github.com/radixark/miles/pull/1045) and [`radixark/Megatron-LM#28`](https://github.com/radixark/Megatron-LM/pull/28) pull requests, and ships in the `radixark/miles:dev` image.

**Key highlights:**

- **Hybrid Attention (CSA + HCA)**: combines **Compressed Sparse Attention** (light compression) and **Heavily Compressed Attention** (heavy compression) layers — DeepSeek's official V4 name (see [HF model card §Introduction](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash#introduction)). Implementation uses low-rank Q (`q_lora_rank=1024`), single-head latent KV (`head_dim=512`), grouped output projection (8 groups, LoRA rank 1024). A learned topk **indexer** (`index_topk=512`, 64 heads × 128 dim) picks 512 KV per query at runtime, inheriting V3.2's DSA-style design.
- **KV compressors**: 44-element compression schedule `compress_ratios = [0, 0, 4, 128, 4, 128, …, 4, 0]` — first / last few layers are uncompressed (ratio 0), middle layers alternate 4× (CSA) and 128× (HCA). The compressor RoPE has its own base (`compress_rope_theta=160000`), separate from the main attention RoPE.
- **Hyper-connection (HC) routing**: each layer expands hidden state into `hc_mult=4` parallel streams and recombines via sinkhorn-normalized mixing. Pipeline-parallel buffers are 4-D `[s, b, hc_mult, d]` instead of 3-D.
- **YaRN RoPE on main attention**: `rope_theta=10000`, YaRN `factor=16`, `original_max_position_embeddings=65536` → effective context length **1,048,576 tokens (1 M)**. Per-head learnable attention sinks (one scalar per head, added to softmax denominator).
- **FP8 weights with simulated FP8 QAT** on indexer and compressor activations.

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| DeepSeek-V4-Flash | 13 B / 284 B | [sgl-project/DeepSeek-V4-Flash-FP8](https://huggingface.co/sgl-project/DeepSeek-V4-Flash-FP8) |

## 3. Quick start

### 3.1 One-line launch

One command runs the full pipeline — dataset download, FP8 → BF16 cast, distributed `torch_dist` conversion, and the training loop:

```bash
# Pull the dev image:
docker pull radixark/miles:dev

# 8-node Flash run, inside the container
cd /root/miles
python scripts/run_deepseek_v4.py full-train \
   --model-name DeepSeek-V4-Flash-FP8 \
   --num-nodes 8 --num-gpus-per-node 8
```

The `full-train` subcommand chains `prepare-download → prepare-single → prepare-spmd → prepare-cp → train`. Each stage has a sentinel-based skip so you can re-run safely after the first invocation.

### 3.2 Launcher path defaults

The Python launcher (`scripts/run_deepseek_v4.py`) takes its path arguments from CLI flags. The defaults are:

| Flag | Default | Use |
|---|---|---|
| `--data-dir` | `/root/datasets` | HF datasets (e.g. dapo-math-17k, …) |
| `--model-dir` | `/root/models` | parent directory holding the HF checkpoint and Megatron `_torch_dist` artifacts as separate sibling sub-directories |
| `--model-local-dir` | `/root/local_data` | local NVMe path on each node; `prepare-cp` rsyncs the HF checkpoint and `_torch_dist` here so the trainer reads from local disk instead of shared storage |
| `--save-dir` | `/root/models` | training checkpoints under `{save-dir}/{run-id}/checkpoints/` |

You can override these on the launcher when your cluster mounts a different layout. There are no `MILES_SCRIPT_*` env vars that preconfigure these paths; the only env vars the launcher reads are `MILES_SCRIPT_EXTERNAL_RAY` and `MILES_SCRIPT_ENABLE_RAY_SUBMIT` (both governing Ray bootstrapping, see [§4.3](#43-multi-node-fan-out)).

## 4. Script breakdown

In this section, we explain what `full-train` does under the hood, and how to drive each stage manually if you need to debug or run outside the one-line launcher.

### 4.1 Download model + datasets

```bash
# inside the radixark/miles:dev container
hf download sgl-project/DeepSeek-V4-Flash-FP8 --local-dir /root/models/DeepSeek-V4-Flash-FP8
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/datasets/aime-2024
```

The Python launcher's `prepare-download` subcommand does the dataset fetch automatically; pass `--hf-checkpoint <path>` to skip the model download when the FP8 weights are already on a shared filesystem.

### 4.2 HF → Megatron `torch_dist` conversion

The conversion happens in two stages — a single-rank FP8 → BF16 cast, followed by a distributed `torch_dist` shard:

```bash
cd /root/miles
python tools/fp8_cast_bf16.py \
   --input-fp8-hf-path /root/models/DeepSeek-V4-Flash-FP8 \
   --output-bf16-hf-path /root/models/DeepSeek-V4-Flash-FP8-bf16/

source scripts/models/deepseek-v4-flash.sh
PYTHONPATH=/root/Megatron-LM torchrun \
   --nproc-per-node 4 --nnodes 8 \
   --master-addr ${MASTER_ADDR} --master-port 12345 \
   --node-rank ${NODE_RANK} \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --tensor-model-parallel-size 1 \
   --pipeline-model-parallel-size 8 \
   --expert-model-parallel-size 4 \
   --decoder-first-pipeline-num-layers 7 \
   --decoder-last-pipeline-num-layers 6 \
   --hf-checkpoint /root/models/DeepSeek-V4-Flash-FP8-bf16/ \
   --save /root/models/DeepSeek-V4-Flash-FP8_torch_dist/
```

The Python launcher's `prepare-spmd` subcommand drives the same conversion.

### 4.3 Multi-node fan-out

The Python launcher manages Ray internally — start each pod with the `radixark/miles:dev` image and a working shared filesystem mounted at the same path on every node, then on the head node:

```bash
ray start --head --num-gpus 8 --disable-usage-stats
# … then on each worker:
ray start --address=${HEAD_IP}:6379 --num-gpus 8 --disable-usage-stats
```

Alternatively, you can set `MILES_SCRIPT_EXTERNAL_RAY=1` and `RAY_ADDRESS=…` to point the launcher at an existing Ray cluster (for example, one that an orchestration layer has already brought up). When `RAY_ADDRESS` is unset, the launcher boots a local Ray head.

### 4.4 Notable quirks

- **Custom `transformers` patch.** miles ships `with_transformers_patch()` (`miles/utils/transformers_patch.py`) so HF's `AutoConfig.from_pretrained` recognizes `model_type=deepseek_v4` / `deepseek_ref` until support lands upstream.

## 5. Example Recipe Configuration

### 5.1 Megatron Parallelism

These are the validated layouts shipped with the launcher; All parallelisms are supported, you can supply any other TP / EP / PP / CP combination that fits your compute.

| Hardware | Nodes × GPUs | TP | PP | CP | EP | expert-TP | Pipeline layout |
|---|---|---|---|---|---|---|---|
| H200 | 8 × 8 = 64 | 8 | 8 | 1 | 8 | 1 | first 4 / last 3 layers |
| GB300 | 8 × 4 = 32 | 8 | 4 | 1 | 8 | 1 | first 11 / last 10 layers |
| GB300 | 8 × 4 = 32 | 2 | 8 | 2 | 4 | 1 | first 4 / last 3 layers |

### 5.2 Algorithm

Using GRPO as an example, you can configure the algorithm with the following flags:

```bash
--advantage-estimator grpo
--eps-clip 0.2
--eps-clip-high 0.28
--kl-loss-coef 0.00
--kl-loss-type low_var_kl
--entropy-coef 0.00
```

The flags `--moe-router-freeze-gate` and `--freeze-e-score-correction-bias` are required and asserted on the `mcore` side — bias-update during RL is forbidden.

### 5.3 Rollout & SGLang

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-tp-size 8
   --sglang-dp-size 8
   --sglang-ep-size 8
   --sglang-enable-dp-attention
   --sglang-attention-backend compressed  # V4 sparse-MLA backend
   --sglang-page-size 256
   --sglang-max-running-requests 64
   --sglang-chunked-prefill-size 8192
   --sglang-mem-fraction-static 0.5  # leave headroom for Megatron during wake_up
   --use-rollout-routing-replay  # MoE routing replay (R3)
   --use-miles-router  # miles router fronts /generate
)
```

The launcher sets the required env vars for you: `SGLANG_SKIP_CHECKPOINT_LOAD_CHECK=1`, `SGLANG_DSV4_FP4_EXPERTS=0`, `MILES_HACK_TRAIN_TORCH_DETERMINISTIC=1`, and `NCCL_ALGO=Ring`.

On the Megatron side, V4 needs `--qkv-format bshd` with CP-aware data slicing. The DSA indexer additionally supports replay via `--use-rollout-indexer-replay` (off by default).

### 5.4 Optimizer

```bash
--optimizer adam
--lr 1e-6 --lr-decay-style constant
--weight-decay 0.1
--adam-beta1 0.9 --adam-beta2 0.98
--accumulate-allreduce-grads-in-fp32
--attention-softmax-in-fp32
--clip-grad 1.0  # Megatron default; not overridden by the launcher
```

The `--low-memory-resume` flag (off by default) puts optimizer states on CPU during ckpt resume to avoid OOM on the very first iteration.

## 6. Pairs Well With

- [FP8 & Low Precision](../../advanced/fp8-low-precision.md)
- [Architecture Support](../../advanced/architecture-support.md) — the V4 plugin lives under `miles_plugins/models/deepseek_v4/`.
