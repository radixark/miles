---
title: DeepSeek-V4 Pro
description: Launch recipe for DeepSeek-V4-Pro (1.6 T) — V4-family architecture at Pro scale.
---

# DeepSeek-V4 Pro

!!! note "Work in progress"
    This page is being filled in. The skeleton below mirrors the
    [V4-Flash recipe page](deepseek-v4-flash.md); sections marked **TBD** will
    receive Pro-specific content as the recipe lands. Tracking issue:
    [`radixark/miles#1046`](https://github.com/radixark/miles/issues/1046).

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
- **Pro-specific launcher defaults**: the launcher flips `optimizer_offload=True` (CPU-offloaded Adam states) and `enable_r3=False` (no Rollout Routing Replay) when `--model-name DeepSeek-V4-Pro-FP8` is selected.

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| DeepSeek-V4-Pro-FP8 | 49 B / 1.6 T | [sgl-project/DeepSeek-V4-Pro-FP8](https://huggingface.co/sgl-project/DeepSeek-V4-Pro-FP8) |

## 3. Quick start

### 3.1 One-line launch

```bash
# Pull the image matching your cluster (TBD)
docker pull <image>

# Production Pro run, inside the container
cd /root/miles
python scripts/run_deepseek_v4.py full-train \
   --model-name DeepSeek-V4-Pro-FP8 \
   --num-nodes 32 --num-gpus-per-node 8
```

TBD — describe what `full-train` chains for Pro and any Pro-specific stage skips.

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

### 5.1 Parallelism

| Hardware | Nodes × GPUs | TP | PP | EP | expert-TP | `max_tokens_per_gpu` | Pipeline layout |
|---|---|---|---|---|---|---|---|
| H200 | 32 × 8 = 256 | 8 | 8 | 32 | 1 | 2048 | first 7 / last 6 layers |

The launcher additionally flips two Pro-specific defaults on selection of
`--model-name DeepSeek-V4-Pro-FP8`: `optimizer_offload=True` (Adam states offloaded to
CPU to fit Pro on H200) and `enable_r3=False` (Rollout Routing Replay disabled).

### 5.2 Algorithm

TBD — GRPO / loss flags / routing-bias freeze constraints for Pro.

### 5.3 Rollout & SGLang

```bash
SGLANG_ARGS=(
   # TBD
)
```

TBD — required env vars and Megatron-side flags.

### 5.4 Optimizer

```bash
# TBD
```

## 6. Pairs Well With

- [FP8 & Low Precision](../../advanced/fp8-low-precision.md)
- [Architecture Support](../../advanced/architecture-support.md)
- [DeepSeek V4 Flash](deepseek-v4-flash.md) — sibling recipe; shares the V4-family architecture.
