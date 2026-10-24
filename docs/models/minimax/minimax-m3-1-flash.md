---
title: MiniMax-M3.1-Flash
sidebarTitle: MiniMax-M3.1
description: Launch recipe for MiniMax-M3.1-Flash (MoE with MSA block-sparse attention) — BF16 train / BF16 rollout, colocated, on GB300.
---
## 1. Model Introduction

[MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) is MiniMax's MoE model with block-sparse attention on every layer. miles trains its text decoder.

**Key highlights:**

- **MSA block-sparse attention on all 60 layers**: a 4-head indexer (128-dim, its own gemma q/k norm and RoPE) scores every causal 128-token block by its maximum logit, and each query attends to the 16 best blocks, its own block always included. Index head `h` selects the blocks for KV head `h`.
- **Attention**: GQA with 64 query and 4 KV heads of dim 128, per-head gemma q/k norm, RoPE on the first 64 dims (base 1e7).
- **MoE**: layers 0–2 are dense (FFN 12288); layers 3–59 route to 4 of 128 experts (FFN 3072) with sigmoid scores, a score-correction bias and a 2.0 scaling factor, plus one shared expert.
- **Activation**: `swigluoai` — `clamp(g, max=7) * sigmoid(1.702 g) * (clamp(u, -7, 7) + 1)` in every MLP.
- **Compressed checkpoint**: MXFP8 linears and NVFP4 routed experts (252 GB). Training and the BF16 rollout read a BF16 conversion (852 GB).

## 2. Supported Variants

| `--model-name` | What it is | GPUs |
|---|---|---|
| `MiniMax-M3.1-Flash-bf16` | text decoder, 60 layers | 32 (8 × 4 GB300) |
| `MiniMax-M3.1-Flash-4layer-bf16` | layers 0, 1, 3, 4: two dense and two MoE layers, every layer sparse | 4 (1 × 4 GB300) |

Both are converted from [MiniMaxAI/MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash). The vision tower and the `dspark/` draft model stay in the checkpoint but are not trained (text-only RL).

## 3. Environment Setup

### 3.1 Download + BF16 conversion

`prepare` (run by the launcher before training) downloads the release to `<model-dir>/MiniMax-M3.1-Flash`, converts it unless `<model-dir>/<model-name>` already holds a finished conversion, converts that to the Megatron reference checkpoint `<ckpt-dir>/<megatron-model-type>_torch_dist`, and fetches dapo-math-17k:

```bash
hf download MiniMaxAI/MiniMax-M3.1-Flash --local-dir /root/models/MiniMax-M3.1-Flash
python tools/convert_minimax_m3_to_bf16.py --model-dir /root/models/MiniMax-M3.1-Flash \
    --save-dir /root/models/MiniMax-M3.1-Flash-4layer-bf16 --device cuda --layers 0,1,3,4
```

The converter dequantizes the MXFP8 linears (one ue8m0 scale per 32 inputs) and the NVFP4 experts (an e4m3 scale per 16 inputs and a per-tensor encode scale) and drops `quantization_config`. Without `--layers` it converts the full model. The full model's torch_dist conversion is sharded PP4 × EP8 over the 8 nodes, since one node runs out of host memory saving it.

### 3.2 SGLang

The colocated BF16 engine needs online weight updates into MiniMax-M3's fused q/k/v + index GEMM: after loading, every sparse layer fuses `qkv_proj` and `index_qkv_proj` into one GEMM and frees the originals, so without the update the first weight sync fails with `IndexError: start out of range`. `sglang-miles` has it since (sglang-miles PR link), and the `radixark/miles:dev` image built from that branch includes it.

## 4. Launch

### 4.1 Four-layer model on one node

```bash
python scripts/run_minimax_m3_1_flash.py --model-name MiniMax-M3.1-Flash-4layer-bf16 --num-nodes 1
```

On 4 × GB300 the BF16 rollout and the trainer agree to a mean absolute log-prob difference of 0.0052, and the bitwise weight check (`--check-weight-update-equal`, on by default for this variant) passes after every update. The model is too shallow to solve the prompts, so every reply runs to the length limit and the reward stays 0; use it to check the setup, not learning.

### 4.2 Full model on eight nodes

Join all eight nodes to a ray head, then launch from the head:

```bash
MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_minimax_m3_1_flash.py \
    --model-name MiniMax-M3.1-Flash-bf16 --num-nodes 8 --train-offload-dir /scratch/offload
```

## 5. Recipe Configuration

### 5.1 Parallelism

| `--model-name` | TP | PP | CP | EP | expert-TP | GPUs | rollout engine |
|---|---|---|---|---|---|---|---|
| `MiniMax-M3.1-Flash-bf16` | 4 | 4 | 1 | 8 | 1 | 32 (8 × 4) | TP8 / EP8 |
| `MiniMax-M3.1-Flash-4layer-bf16` | 2 | 1 | 1 | 2 | 1 | 4 | TP4 / EP4 |

TP is capped at 4 by the four KV heads. Context parallelism is not supported, and the sparse attention needs THD packing (`--qkv-format thd`).

### 5.2 Algorithm

GRPO with `--eps-clip 0.2 --eps-clip-high 0.28`, no KL or entropy term, and `--rm-type math` on dapo-math-17k: 16 prompts × 16 samples per step at temperature 0.8. Rollout routing replay (`--use-rollout-routing-replay`) is on, so the trainer routes every token to the experts the engine picked.

### 5.3 Optimizer

Adam at `--lr 3e-6`. The full model's Adam state fits neither the GPUs nor the hosts' memory, so it streams through node-local NVMe:

```bash
--stream-optimizer-state-to-disk
--stream-optimizer-state-moment-dtype bf16
--offload-train-target disk
--offload-train-disk-dir <train-offload-dir>
```

### 5.4 Notable quirks

- **The indexer is frozen**: its projections and norms are loaded but not trained, and the trainer recomputes the block selection from its own hidden states rather than replaying the engine's. The two selections differ only at near-ties.
- **Only the all-sparse layout**: the plugin builds MSA attention on every layer and rejects a checkpoint with dense-attention layers (for example MiniMax-M3) with a `NotImplementedError`.
- **Frozen towers and `--check-weight-update-equal`**: the trainer never sends the vision weights, so the launcher excludes them from the post-update check with `--check-weight-update-skip-list vision_tower. multi_modal_projector. patch_merge_mlp.`. The check snapshots every weight on the host, which the full model does not fit, so it is off there.

## 6. Pairs Well With

- [Low Precision RL](/advanced/low-precision)
