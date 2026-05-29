---
title: Nemotron-3-Nano MoE
description: Launch recipe for NVIDIA Nemotron-3-Nano-30B-A3B (Mamba+Attention+MoE hybrid) via Megatron AutoBridge.
---

# Nemotron-3-Nano MoE

## 1. Model Introduction

[NVIDIA Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
is a hybrid Mamba + attention + MoE model. It pairs the `nemotron_h` block
pattern from the dense 4B with a 128-expert sparse layer (top-6 routing,
1 shared expert, DSv3-style sigmoid routing with `routed_scaling_factor=2.5`).

miles loads it through the `megatron.bridge` AutoBridge with a custom
**NemotronH MoE bridge shim** (`miles_plugins/megatron_bridge/nemotron_h.py`) that
wires `routed_scaling_factor`, `n_group`, and `topk_group` onto the Megatron
provider. Without the shim the routed output is silently scaled 1.0× → ~0.28
logprob drift between train and rollout.

**Key highlights:**

- **Hybrid + MoE**: Mamba + attention + sparse MoE in the `nemotron_h` family.
- **128 experts, top-6 routing**, 1 shared expert (3712-dim), aux-free expert-bias load balancing.
- **Sigmoid routing** with `--moe-router-topk-scaling-factor 2.5`.
- **Bridge-mode load**: `--megatron-to-hf-mode bridge` — no `torch_dist` conversion step.
- **No RoPE**: `--position-embedding-type none`.

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| Nemotron-3-Nano-30B-A3B | 3 B / 30 B | [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) |

## 3. Environment Setup

### 3.1 Download model + datasets

```bash
export BASE_DIR=/root/miles_data
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir $BASE_DIR/dapo-math-17k
hf download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --local-dir $BASE_DIR/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

### 3.2 No `torch_dist` conversion

AutoBridge + the NemotronH MoE shim load the HF checkpoint directly. Both
`--hf-checkpoint` and `--ref-load` point at the HF directory:

```bash
CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
   --ref-load     $BASE_DIR/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
   --save         $BASE_DIR/nemotron-3-nano-30b-a3b_miles
   --save-interval 20
   --megatron-to-hf-mode bridge
)
```

## 4. Launch

### 4.1 Quick start

```bash
cd /root/miles
export BASE_DIR=/root/miles_data
bash scripts/run-nemotron-3-nano-30b-a3b.sh
```

The script targets 1 node × 8 GPU (H200). Default cell is `TP=2 PP=2 EP=2`.

## 5. Recipe Configuration

### 5.1 Parallelism

Default cell is `TP=2 PP=2 EP=2`. Other verified cells from the upstream PR
(10-step RL smoke, max logprob diff ≈ 0.014):

| Cell | TP | PP | CP | EP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|
| **default (run script)** | 2 | 2 | 1 | 2 | 1024 | 8 (1 × 8) |
| EP=4 | 1 | 1 | 1 | 4 | 1024 | 8 |
| TP=2×EP=4+SP | 2 | 1 | 1 | 4 | 1024 | 8 |
| PP=2×EP=4 | 1 | 2 | 1 | 4 | 1024 | 8 |
| CP=2×EP=4 | 1 | 1 | 2 | 4 | 1024 | 8 |
| TP=2×PP=2×EP=2+SP | 2 | 2 | 1 | 2 | 1024 | 8 |

`--sequence-parallel` is enabled in the run script. Activation checkpointing is on
(`--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`).
`--log-probs-chunk-size 128` is required for the smoke memory budget.

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
   # Replay the exact rollout routing during training forward so
   # train logprobs match rollout logprobs (needed for MoE).
   --use-miles-router
   --use-rollout-routing-replay
)
```

`--use-miles-router --use-rollout-routing-replay` is what keeps train and rollout
logprobs aligned for the sigmoid-routed MoE — drop them and you'll see the same
~0.28 drift the bridge shim was added to fix.

### 5.4 Optimizer

GPU Adam in the smoke recipe (no `--optimizer-cpu-offload`). Switch on CPU Adam if
memory pressure rises.

### 5.5 Notable quirks

From `scripts/models/nemotron-3-nano-30b-a3b.sh` and `scripts/run-nemotron-3-nano-30b-a3b.sh`:

- **No `--spec`**: AutoBridge + the NemotronH shim synthesize the Megatron MoE spec from HF config.
- 128 experts, `--moe-router-topk 6`, shared expert (3712-dim).
- Routing: `--moe-router-score-function sigmoid --moe-router-pre-softmax --moe-router-topk-scaling-factor 2.5`.
- Group routing: `--moe-router-num-groups 1 --moe-router-group-topk 1` (no-op for `n_group=1`, kept for parity with HF config).
- Aux-free balancing: `--moe-router-enable-expert-bias --moe-router-load-balancing-type seq_aux_loss --moe-router-bias-update-rate 0 --moe-aux-loss-coeff 0`.
- `--moe-grouped-gemm`, `--moe-router-dtype fp32`.
- `--position-embedding-type none`, `--vocab-size 131072 --make-vocab-size-divisible-by 128`.
- `--attention-backend auto` (Mamba layers select their own kernel).

See [Backends Beyond Megatron](../../advanced/architecture-support.md) for how the bridge
shim layers `routed_scaling_factor` / `n_group` / `topk_group` onto the Megatron provider.

## 6. Pairs Well With

- [Backends Beyond Megatron](../../advanced/architecture-support.md)
- [P2P Weight Transfer](../../advanced/p2p-weight-transfer.md)
- [FP8 & Low Precision](../../advanced/fp8-low-precision.md)
