---
title: MiMo-V2.6-Flash
sidebarTitle: MiMo-V2.6
description: Launch recipe for Xiaomi's MiMo-V2.6-Flash-RL (309B MoE, hybrid sliding-window attention) — Megatron bridge mode on a BF16 conversion, a 4-layer partial on one 8-GPU node, the full model on two.
---
## 1. Model Introduction

[MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) is Xiaomi's 309B-parameter MoE model (15B active). miles trains its text decoder.

**Key highlights:**

- **Hybrid attention**: 39 sliding-window layers (window 128, learnable attention sink) and 9 global layers, with different KV head counts (8 vs 4) and RoPE bases (1e4 vs 1e7).
- **Asymmetric heads**: 192-dim Q/K and 128-dim V, V scaled by 0.707, partial RoPE (64 of 192 dims).
- **MoE**: 256 routed experts, top-8 sigmoid routing with a fixed score-correction bias; layer 0 is dense.
- **Compressed checkpoint**: FP8 block-quantized dense weights with a fused, kv-head-interleaved `qkv_proj`, and MXFP4 experts. The Megatron bridge reads a BF16 split-q/k/v conversion instead.

## 2. Supported Variants

| `--model-name` | What it is | HF source |
|---|---|---|
| `MiMo-V2.6-Flash-RL-bf16` | full text decoder, 48 layers | [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) |
| `mimo26-p4-bf16` | 4-layer partial (source layers 0, 1, 5, 6: one of each decoder variant), all 256 experts | same, converted with `--layers 0,1,5,6` |

The vision, audio and MTP modules stay in the checkpoint but are not trained (text-only RL).

## 3. Environment Setup

### 3.1 Download + BF16 conversion

`prepare` (run by the launcher before training) downloads the official checkpoint to `<model-dir>/MiMo-V2.6-Flash-RL` and converts it unless `<model-dir>/<model-name>` already holds a finished conversion:

```bash
hf download XiaomiMiMo/MiMo-V2.6-Flash-RL --local-dir /root/models/MiMo-V2.6-Flash-RL
python tools/convert_mimo_v2_to_bf16.py --model-dir /root/models/MiMo-V2.6-Flash-RL \
    --save-dir /root/models/mimo26-p4-bf16 --layers 0,1,5,6 --device cuda
```

The converter dequantizes each kv-head shard of the fused `qkv_proj` with its own FP8 block scales, splits it into q/k/v in head order, and decodes the MXFP4 experts (`--num-experts N` keeps the first N experts for smaller tests). The full BF16 model is 622 GB, the 4-layer partial 46 GB.

### 3.2 SGLang

The BF16 engine needs the MiMo-V2 fixes that are not in the image yet:

- upstream [sgl-project/sglang#40448](https://github.com/sgl-project/sglang/pull/40448) (MXFP4 MoE and BF16 router);
- accepting the `split` attention layout of the BF16 conversion;
- allocating the sliding-window KV pools inside the memory-saver region, otherwise a colocated engine cannot release its KV cache;
- passing `layer_id` to the MoE top-k, otherwise `--use-rollout-routing-replay` crashes the CUDA-graph capture;
- passing `sliding_window_size - 1` to the attention layers: SGLang's window excludes the query token, so the unmodified model attends 129 keys where the HF reference and Megatron attend 128, which shows up as rare large train/rollout log-prob gaps.

## 4. Launch

### 4.1 Quick start (one node, 4-layer partial)

```bash
cd /root/miles
python scripts/run_mimo_v2_6_flash.py --mode rl --model-name mimo26-p4-bf16
python scripts/run_mimo_v2_6_flash.py --mode sft --model-name mimo26-p4-bf16 --prompt-data <chat jsonl>
```

`--mode sft` reads a `messages` column; put a reasoning trace in `reasoning_content`, since the MiMo chat template renders `<think>{reasoning_content}</think>`.

### 4.2 Full model (two nodes)

Join the second node to a ray head on the first, then launch from the head:

```bash
MILES_SCRIPT_EXTERNAL_RAY=1 MASTER_ADDR=<head ip> python scripts/run_mimo_v2_6_flash.py \
    --mode rl --model-name MiMo-V2.6-Flash-RL-bf16 --num-nodes 2 --train-offload-disk-dir /scratch/offload
```

On two 8×H200 nodes with node-local NVMe (4-drive RAID0), a step with 16 samples of up to 2048 response tokens took about 4 minutes of training, of which about 3 minutes is streaming the optimizer state (72 GB read and 144 GB written per GPU), plus 30 s of weight update. Peak GPU memory was 125 GB and peak host memory 1.05 TB per node.

## 5. Recipe Configuration

### 5.1 Parallelism

| `--model-name` | TP | PP | CP | EP | expert-TP | GPUs | rollout engine |
|---|---|---|---|---|---|---|---|
| `mimo26-p4-bf16` | 2 | 2 | 1 | 2 | 1 | 8 (1 × 8) | TP4, `--sglang-mem-fraction-static 0.6` |
| `MiMo-V2.6-Flash-RL-bf16` | 2 | 2 | 1 | 8 | 1 | 16 (2 × 8) | TP8, `--sglang-mem-fraction-static 0.8` |

TP is capped at 4 by the four global-attention KV heads. Context parallelism is not supported. `--sequence-parallel` is on whenever TP > 1. THD packing with `--use-dynamic-batch-size` is the default; `--qkv-format bshd` runs one sample per micro-batch.

### 5.2 Algorithm

GRPO with `--eps-clip 0.2 --eps-clip-high 0.28 --entropy-coef 0.00` and `--rm-type math` on dapo-math-17k. No KL loss: bridge mode has no Megatron-format reference checkpoint for `--ref-load`.

### 5.3 Optimizer

Adam at `--lr 1e-6`. The full model's Adam state (3.7 TB) fits neither the GPUs nor two hosts' memory, so it streams through node-local NVMe:

```bash
--stream-optimizer-state-to-disk
--stream-optimizer-state-moment-dtype bf16
--grad-reduce-in-bf16
--offload-train-target disk
--offload-train-disk-dir <train-offload-disk-dir>
```

### 5.4 Notable quirks

- **Fused attention only**: `--attention-backend fused`. The learnable sink on the sliding-window layers is implemented by TE's FusedAttention (THD needs cuDNN ≥ 9.18) and the unfused path, not by FlashAttention.
- **Dropout**: the bridge sets `hidden_dropout = 0`. Bridge mode does not copy `--hidden-dropout` onto the provider, so a provider left at Megatron's default (0.1) trains with dropout silently.
- **Frozen towers and `--check-weight-update-equal`**: the trainer never sends the vision/audio weights, so exclude them from the post-update check with `--check-weight-update-skip-list visual. audio_tokenizer. input_local_transformer. speech_embeddings. projection.mlp.`.

## 6. Pairs Well With

- [Low Precision RL](/advanced/low-precision)
