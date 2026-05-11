---
title: Nemotron-3-Super
description: Launch recipe for NVIDIA Nemotron-3-Super-120B-A12B-FP8 (Mamba+Attention+MoE hybrid, FP8 native) via Megatron AutoBridge.
---

# Nemotron-3-Super

## 1. Model Introduction

[NVIDIA Nemotron-3-Super-120B-A12B-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8)
is the Super-tier sibling of Nemotron-3-Nano: the same `nemotron_h` block
pattern (interleaved Mamba and attention blocks, no RoPE, squared-relu FFNs)
scaled to **120 B total / 12 B active** with a sparse MoE FFN, and shipped as
an **FP8-native** checkpoint.

miles loads it through the `megatron.bridge` AutoBridge with the shared
**NemotronH MoE bridge shim** (`miles_plugins/megatron_bridge/nemotron_h.py`)
that wires `routed_scaling_factor`, `n_group`, and `topk_group` onto the
Megatron provider — without the shim the routed output is silently scaled 1.0×,
the same drift class that affects the Nano-MoE recipe.

**Key highlights:**

- **Hybrid + MoE**: Mamba + attention + sparse MoE in the `nemotron_h` family.
- **FP8 native**: weights ship in FP8; load + train without an offline upcast.
- **Sigmoid routing** with per-token group selection, aux-free expert-bias load balancing.
- **Bridge-mode load**: `--megatron-to-hf-mode bridge` — no `torch_dist` conversion step.
- **No RoPE**: `--position-embedding-type none`.

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| Nemotron-3-Super-120B-A12B-FP8 | 12 B / 120 B | [nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8) |

## 3. Environment Setup

### 3.1 Download model + datasets

```bash
export BASE_DIR=/root/miles_data
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir $BASE_DIR/dapo-math-17k
hf download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \
   --local-dir $BASE_DIR/NVIDIA-Nemotron-3-Super-120B-A12B-FP8
```

### 3.2 No `torch_dist` conversion

AutoBridge + the NemotronH MoE shim load the FP8 HF checkpoint directly. Both
`--hf-checkpoint` and `--ref-load` point at the HF directory:

```bash
CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/NVIDIA-Nemotron-3-Super-120B-A12B-FP8
   --ref-load     $BASE_DIR/NVIDIA-Nemotron-3-Super-120B-A12B-FP8
   --save         $BASE_DIR/nemotron-3-super-120b-a12b_miles
   --save-interval 20
   --megatron-to-hf-mode bridge
)
```

## 4. Launch

### 4.1 Quick start

```bash
cd /root/miles
export BASE_DIR=/root/miles_data
bash scripts/run-nemotron-3-super-120b-a12b.sh
```

The script targets **2 nodes × 8 GPU (H200, FP8)**. Default cell is
`TP=4 PP=2 EP=8`.

## 5. Recipe Configuration

### 5.1 Parallelism

Default cell is `TP=4 PP=2 EP=8` on 16 GPUs. The 120B-A12B footprint requires
either a wider EP fan-out or PP=2 to fit the activation memory of the hybrid
Mamba+Attention stack at the FP8 weight resolution.

| Cell | TP | PP | CP | EP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|
| **default (run script)** | 4 | 2 | 1 | 8 | 1024 | 16 (2 × 8) |
| TP=4×EP=8 (1 node)       | 4 | 1 | 1 | 8 | 1024 | 8 (1 × 8) |
| TP=2×PP=2×EP=8 + SP      | 2 | 2 | 1 | 8 | 1024 | 16 (2 × 8) |

`--sequence-parallel` is enabled in the run script. Activation checkpointing is
on (`--recompute-granularity full --recompute-method uniform
--recompute-num-layers 1`). `--log-probs-chunk-size 128` keeps the smoke
memory budget intact for FP8.

### 5.2 Algorithm

GRPO with low-variance KL — same defaults as Nano-MoE:

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
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
   # Replay the exact rollout routing during training forward so
   # train logprobs match rollout logprobs (needed for sigmoid-routed MoE).
   --use-miles-router
   --use-rollout-routing-replay
)
```

`--use-miles-router --use-rollout-routing-replay` keeps train and rollout
logprobs aligned for the sigmoid-routed MoE — the same routing-replay rule that
Nano-MoE uses.

### 5.4 Optimizer

CPU Adam (`--optimizer-cpu-offload`) is the default for the 120B-A12B smoke
recipe; the FP8 weights save GPU memory but the Adam states still dominate at
this scale.

### 5.5 Notable quirks

- **FP8 native load**: the HF checkpoint is FP8; the bridge passes the
  per-block scales through to Megatron — no offline upcast step.
- **No `--spec`**: AutoBridge + the NemotronH shim synthesize the Megatron MoE
  spec from HF config.
- Routing follows the family default: `--moe-router-score-function sigmoid
  --moe-router-pre-softmax --moe-router-topk-scaling-factor 2.5`.
- Aux-free balancing: `--moe-router-enable-expert-bias
  --moe-router-load-balancing-type seq_aux_loss
  --moe-router-bias-update-rate 0 --moe-aux-loss-coeff 0`.
- `--moe-grouped-gemm`, `--moe-router-dtype fp32`.
- `--position-embedding-type none`, `--vocab-size 131072
  --make-vocab-size-divisible-by 128`.
- `--attention-backend auto` (Mamba layers select their own kernel).

See [Backends Beyond Megatron](../../advanced/architecture-support.md) for how
the bridge shim layers `routed_scaling_factor` / `n_group` / `topk_group` onto
the Megatron provider, and [FP8 & Low Precision](../../advanced/fp8-low-precision.md)
for the FP8 weight format.

## 6. Pairs Well With

- [Backends Beyond Megatron](../../advanced/architecture-support.md)
- [FP8 & Low Precision](../../advanced/fp8-low-precision.md)
- [P2P Weight Transfer](../../advanced/p2p-weight-transfer.md)
