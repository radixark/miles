---
title: Nemotron-3-Nano
description: Launch recipe for the dense NVIDIA Nemotron-3-Nano-4B (Mamba+Attention hybrid) via Megatron AutoBridge.
---

# Nemotron-3-Nano

## 1. Model Introduction

[NVIDIA Nemotron-3-Nano-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)
is a dense `nemotron_h` hybrid model — interleaved Mamba and attention blocks
with squared-relu FFNs, no RoPE, and a 262 144 max position. miles wires it via
the `megatron.bridge` AutoBridge path, so there is **no `torch_dist` conversion
step**: the AutoBridge constructs the full Megatron provider from the HF
`config.json` at load time, including all Mamba-specific fields
(`mamba_num_heads`, `mamba_state_dim`, `hybrid_override_pattern`, etc.).

**Key highlights:**

- **Hybrid architecture**: Mamba + attention layers (`nemotron_h` family).
- **Bridge-mode load**: `--megatron-to-hf-mode bridge` — no separate Megatron checkpoint.
- **No RoPE**: `--position-embedding-type none`.
- **Vocab**: 131 072 tokens, padded to a multiple of 128.

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| Nemotron-3-Nano-4B | 4 B / 4 B | [nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) |

## 3. Environment Setup

### 3.1 Download model + datasets

```bash
export BASE_DIR=/root/miles_data
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir $BASE_DIR/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir $BASE_DIR/aime-2024
hf download nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 --local-dir $BASE_DIR/NVIDIA-Nemotron-3-Nano-4B-BF16
```

### 3.2 No `torch_dist` conversion

AutoBridge loads the HF checkpoint directly. Both `--hf-checkpoint` and
`--ref-load` point at the HF directory, and `--megatron-to-hf-mode bridge`
turns on the bridge code path:

```bash
CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/NVIDIA-Nemotron-3-Nano-4B-BF16
   --ref-load     $BASE_DIR/NVIDIA-Nemotron-3-Nano-4B-BF16
   --megatron-to-hf-mode bridge
)
```

## 4. Launch

### 4.1 Quick start

```bash
cd /root/miles
export BASE_DIR=/root/miles_data
bash scripts/run-nemotron-3-nano-4b.sh
```

The script targets 1 node × 8 GPU (H100/H200). Default cell is `TP=2 PP=2`.

## 5. Recipe Configuration

### 5.1 Parallelism

The script ships a starting cell of `TP=2 PP=2`. Other verified cells (10-step RL
smoke tests, max train/rollout logprob diff): TP=2, TP=4, PP=2, CP=2, TP=2×PP=2.
Swap the `PERF_ARGS` block to switch.

| Cell | TP | PP | CP | EP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|
| **default (run script)** | 2 | 2 | 1 | 1 | 9216 | 8 (1 × 8) |
| TP=2 | 2 | 1 | 1 | 1 | 9216 | 8 |
| TP=4 | 4 | 1 | 1 | 1 | 9216 | 8 |
| CP=2 | 1 | 1 | 2 | 1 | 9216 | 8 |

`--sequence-parallel` is not enabled in the dense smoke recipe; activation
checkpointing is also off. Dense Nemotron-3-Nano has no expert parallelism.

### 5.2 Algorithm

GRPO with low-variance KL:

```bash
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)
```

### 5.3 Rollout & SGLang

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
)
```

### 5.4 Optimizer

GPU Adam in the smoke recipe (no `--optimizer-cpu-offload`). Switch on CPU Adam if
memory pressure rises.

### 5.5 Notable quirks

From `scripts/models/nemotron-3-nano-4b.sh` and `scripts/run-nemotron-3-nano-4b.sh`:

- **No `--spec`**: the AutoBridge synthesizes the Megatron spec from HF config.
- `--position-embedding-type none` (no RoPE).
- `--vocab-size 131072 --make-vocab-size-divisible-by 128`.
- `--attention-backend auto` (the Mamba layers select their own kernel; flash-only is not safe here).
- Bridge load is required for hybrid `nemotron_h`: the AutoBridge wires `mamba_num_heads`, `mamba_state_dim`, `hybrid_override_pattern`. PP additionally needs miles' PP-unwrap shim (already on the `feat/nemotron-gemma4-rl` branch).

See [Backends Beyond Megatron](../../advanced/architecture-support.md) for the AutoBridge wiring.

## 6. Pairs Well With

- [Backends Beyond Megatron](../../advanced/architecture-support.md)
- [P2P Weight Transfer](../../advanced/p2p-weight-transfer.md)
