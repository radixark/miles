---
title: DeepSeek-V4 Pro
description: Launch recipe for DeepSeek-V4-Pro (1.6 T) — V4-family architecture at Pro scale.
---

# DeepSeek-V4 Pro

DeepSeek V4 training tracking issue: [`radixark/miles#1046`](https://github.com/radixark/miles/issues/1046).

## 1. Model Introduction

[DeepSeek-V4-Pro](https://huggingface.co/sgl-project/DeepSeek-V4-Pro-FP8) is a 49 B-active / 1.6 T-total MoE that scales up the same sparse-MLA + DSA-indexer + KV-compressor + hyper-connection stack as [V4-Flash](deepseek-v4-flash.md). The architecture family is identical; the deltas are size and a handful of tuned knobs (indexer top-k, output-projection groups, compression schedule). The miles + Megatron-Core integration ships in the same image as Flash and is selected with `--model-name DeepSeek-V4-Pro-FP8`.

**Key highlights** (deltas vs [V4-Flash](deepseek-v4-flash.md#1-model-introduction)):

- **Scaled-up V4 architecture**: 61 layers (vs 43), hidden-size 7168 (vs 4096), 128 attention heads (vs 64), `ffn_hidden_size=3072` and `moe_ffn_hidden_size=3072` (vs 2048). All layers are MoE (same `--moe-layer-freq` pattern). `q_lora_rank=1536` (vs 1024); latent KV (`kv_lora_rank=512`, `qk_head_dim=512`, `v_head_dim=512`) is unchanged across V4.
- **Hybrid Attention with wider indexer and output projection**: `index_topk=1024` (vs Flash's 512) — Pro keeps 64 indexer heads × 128 dim but picks twice as many KV per query. Grouped output projection uses `o_groups=16` (vs 8), keeping `o_lora_rank=1024`.
- **KV compressors start heavily compressed**: 60-element schedule `[128, 128, 4, 128, 4, 128, …, 4, 0]` — Pro skips Flash's two leading uncompressed layers and starts at ratio-128 (HCA) from layer 0. Middle layers still alternate 4× (CSA) and 128× (HCA); only the final layer is uncompressed. Compressor RoPE base (`compress_rope_theta=160000`) is shared with Flash.
- **MoE topology**: 384 routed experts + 1 shared (vs Flash's 256 + 1), top-6. `--moe-router-topk-scaling-factor 2.5` (vs Flash 1.5) compensates for the larger expert pool. The first 3 layers (`num_hash_layers=3`) remain dense-routed via hash buckets.
- **Identical YaRN RoPE and context**: `rope_theta=10000`, YaRN `factor=16`, `original_max_position_embeddings=65536` → effective context length **1,048,576 tokens (1 M)**, same as Flash.
- **Hyper-connection (HC) routing**: `hc_mult=4` parallel streams with sinkhorn-normalised mixing, same as Flash (PP buffers stay 4-D).
- **FP8 weights with simulated FP8 QAT** on indexer and compressor activations; default training is BF16 on the cast checkpoint and default rollout is FP8 in SGLang with `--sglang-attention-backend compressed`.

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| DeepSeek-V4-Pro-FP8 | 49 B / 1.6 T | [sgl-project/DeepSeek-V4-Pro-FP8](https://huggingface.co/sgl-project/DeepSeek-V4-Pro-FP8) |

## 3. Quick start

### 3.1 One-line launch

```bash
# Pull the image matching your cluster:
#   H200 / B200 (cu129 x86) -> radixark/miles:deepseek-v4
#   GB300       (cu130 arm64) -> radixark/miles:gb300-dev-dskv4
docker pull radixark/miles:deepseek-v4

# Production Pro run, inside the container
cd /root/miles
python scripts/run_deepseek_v4.py full-train \
   --model-name DeepSeek-V4-Pro-FP8 \
   --num-nodes 32 --num-gpus-per-node 8
```

The `full-train` subcommand chains `prepare-download → prepare-single → prepare-spmd → prepare-cp → train`. Each stage has a sentinel-based skip so you can re-run safely after the first invocation.

### 3.2 Launcher path defaults

| Flag | Default | Use |
|---|---|---|
| `--data-dir` | `/root/datasets` | HF datasets (e.g. dapo-math-17k, …) |
| `--model-dir` | `/root/models` | parent directory holding the HF checkpoint and Megatron `_torch_dist` artifacts |
| `--model-local-dir` | `/root/local_data` | local NVMe path on each node; `prepare-cp` rsyncs the HF checkpoint and `_torch_dist` here so the trainer reads from local disk |
| `--save-dir` | `/root/models` | training checkpoints under `{save-dir}/{run-id}/checkpoints/` |

TBD — Pro-specific overrides or env-var notes.

## 4. Script breakdown

The under-the-hood stages are essentially identical to V4-Flash — see the [V4-Flash Script breakdown](deepseek-v4-flash.md#4-script-breakdown) and substitute the Pro model name and path defaults shown above.

## 5. Example Recipe Configuration

### 5.1 Megatron Parallelism

These are the validated layouts shipped with the launcher; All parallelisms are supported, you can supply any other TP / EP / PP / CP combination that fits your compute.

| Hardware | Nodes × GPUs | TP | PP | CP | EP | expert-TP | Pipeline layout |
|---|---|---|---|---|---|---|---|
| H200 | 32 × 8 = 256 | 8 | 8 | 1 | 32 | 1 | first 7 / last 6 layers |

### 5.2 Algorithm

Same as Flash — see [V4-Flash §5.2 Algorithm](deepseek-v4-flash.md#52-algorithm).

### 5.3 Rollout & SGLang

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 32
   --sglang-tp-size 32
   --sglang-dp-size 32
   --sglang-ep-size 32
   --sglang-enable-dp-attention
   --sglang-attention-backend compressed       # V4 sparse-MLA backend
   --sglang-page-size 256
   --sglang-max-running-requests 64
   --sglang-chunked-prefill-size 8192
   --sglang-server-concurrency 1024
   --sglang-moe-a2a-backend deepep             # DeepEP normal-mode dispatch
   --sglang-cuda-graph-max-bs 8                # see hang caveat below
   --sglang-mem-fraction-static 0.7            # Pro needs a larger dynamic buffer
   --use-miles-router
)
```

Required env vars (the launcher sets these for you): `SGLANG_SKIP_CHECKPOINT_LOAD_CHECK=1`, `SGLANG_DSV4_FP4_EXPERTS=0`, and the Pro-only pair `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256`, `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0`.

Megatron side: `--qkv-format bshd` (V4 needs `bshd` with CP-aware data slicing). The DSA indexer additionally supports replay via `--use-rollout-indexer-replay` (off by default).

!!! warning "Pro-specific rollout caveats"
    1. **Engine size ≥ 32 GPUs.** Pro needs a single SGLang engine spanning at least 32 GPUs — the launcher hard-codes `--rollout-num-gpus-per-engine 32`. Smaller engines do not leave enough memory after weights, KV cache, indexer state, and DeepEP buffers, and rollout will OOM under load.
    2. **EP is mandatory; pure TP will not shard the model.** 384 routed experts × `moe_ffn_hidden_size=3072` cannot be partitioned by tensor parallelism alone — the model must use expert parallelism (`--sglang-ep-size 32`) to spread the expert MLPs across ranks. `--sglang-tp-size 32` only covers the attention / embedding paths.
    3. **DeepEP normal-mode + CUDA graphs can hang at large batch sizes.** When `--sglang-moe-a2a-backend deepep` is on, an overly large `--sglang-cuda-graph-max-bs` makes SGLang hang during graph capture or replay. The launcher pins it to `8` for Pro — raise it only after verifying the engine doesn't deadlock at your target batch.

### 5.4 Optimizer

```bash
--optimizer adam
--lr 1e-6 --lr-decay-style constant
--weight-decay 0.1
--adam-beta1 0.9 --adam-beta2 0.98
--accumulate-allreduce-grads-in-fp32
--attention-softmax-in-fp32
--clip-grad 1.0                       # Megatron default; not overridden by the launcher

# Pro-only — forced on by the launcher (optimizer_offload=True)
--optimizer-cpu-offload
--use-precision-aware-optimizer
--overlap-cpu-optimizer-d2h-h2d
```

Pro selects `--model-name DeepSeek-V4-Pro-FP8`, which flips `optimizer_offload=True` in the launcher (`scripts/run_deepseek_v4.py`) and appends the three CPU-offload flags above. Adam states live on host RAM and are D2H/H2D-overlapped with the backward pass, freeing GPU memory for the 1.6 T weight + KV footprint. The `--low-memory-resume` flag (off by default) additionally puts optimizer states on CPU during ckpt resume to avoid OOM on the very first iteration.

## 6. Pairs Well With

- [FP8 & Low Precision](../../advanced/fp8-low-precision.md)
- [Architecture Support](../../advanced/architecture-support.md)
- [DeepSeek V4 Flash](deepseek-v4-flash.md) — sibling recipe; shares the V4-family architecture.
