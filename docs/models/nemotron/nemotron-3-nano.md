---
title: Nemotron-3-Nano
description: Launch recipe for the dense NVIDIA Nemotron-3-Nano-4B (Mamba+Attention hybrid) via Megatron AutoBridge.
---
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
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/datasets/aime-2024
hf download nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 --local-dir /root/models/NVIDIA-Nemotron-3-Nano-4B-BF16
```

Those are the `--data-dir` and `--model-dir` defaults; point them elsewhere with the flags.

### 3.2 No `torch_dist` conversion

AutoBridge loads the HF checkpoint directly. Both `--hf-checkpoint` and
`--ref-load` point at the HF directory, and `--megatron-to-hf-mode bridge`
turns on the bridge code path:

```bash
--hf-checkpoint <model-dir>/NVIDIA-Nemotron-3-Nano-4B-BF16
--ref-load      <model-dir>/NVIDIA-Nemotron-3-Nano-4B-BF16
--megatron-to-hf-mode bridge
```

## 4. Launch

### 4.1 Quick start

```bash
cd /root/miles
python scripts/run_nemotron_3_nano.py --model-name NVIDIA-Nemotron-3-Nano-4B-BF16
```

`scripts/run_nemotron_3_nano.py` covers both Nano variants; `--model-name` picks the
dense 4 B here and the MoE 30B-A3B on the [sibling page](/models/nemotron/nemotron-3-nano-moe).
The recipe targets 1 node × 8 GPU (H100/H200), default cell `TP=2 PP=2`, and it is a
10-step smoke test (`--num-rollout`). Checkpoints are written under `--output-dir`
(default `/root/shared_data`).

## 5. Recipe Configuration

### 5.1 Parallelism

The recipe ships a starting cell of `TP=2 PP=2`. Other verified cells (10-step RL
smoke tests, max train/rollout logprob diff): TP=2, TP=4, PP=2, CP=2, TP=2×PP=2.
Change the `tp` / `pp` / `cp` fields of the recipe to switch.

| Cell | TP | PP | CP | EP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|
| **default (launcher)** | 2 | 2 | 1 | 1 | 9216 | 8 (1 × 8) |
| TP=2 | 2 | 1 | 1 | 1 | 9216 | 8 |
| TP=4 | 4 | 1 | 1 | 1 | 9216 | 8 |
| CP=2 | 1 | 1 | 2 | 1 | 9216 | 8 |

`--sequence-parallel` is not enabled in the dense smoke recipe; activation
checkpointing is also off. Dense Nemotron-3-Nano has no expert parallelism.

### 5.2 Algorithm

GRPO with low-variance KL:

```bash
--advantage-estimator grpo
--use-kl-loss
--kl-loss-coef 0.00
--kl-loss-type low_var_kl
--entropy-coef 0.00
--eps-clip 0.2
--eps-clip-high 0.28
```

### 5.3 Rollout & SGLang

```bash
--rollout-num-gpus-per-engine 1
--sglang-mem-fraction-static 0.7
```

### 5.4 Optimizer

GPU Adam in the smoke recipe (no `--optimizer-cpu-offload`). Switch on CPU Adam if
memory pressure rises.

### 5.5 Notable quirks

From `scripts/models/nemotron-3-nano-4b.py` and `scripts/run_nemotron_3_nano.py`:

- **No `--spec`**: the AutoBridge synthesizes the Megatron spec from HF config.
- `--position-embedding-type none` (no RoPE).
- `--vocab-size 131072 --make-vocab-size-divisible-by 128`.
- `--attention-backend auto` (the Mamba layers select their own kernel; flash-only is not safe here).
- Bridge load is required for hybrid `nemotron_h`: the AutoBridge wires `mamba_num_heads`, `mamba_state_dim`, `hybrid_override_pattern`. PP additionally needs miles' PP-unwrap shim (already on the `feat/nemotron-gemma4-rl` branch).

See [Backends Beyond Megatron](/advanced/architecture-support) for the AutoBridge wiring.

## 6. Pairs Well With

- [Backends Beyond Megatron](/advanced/architecture-support)
- [P2P Weight Transfer](/advanced/p2p-weight-transfer)
