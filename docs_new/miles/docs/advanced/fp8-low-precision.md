---
title: Low Precision RL
description: Unified low-precision pipelines for RL — block-wise FP8, MXFP8, and NVFP4 across rollout and training.
---

# Low Precision RL

A common failure mode in MoE RL is precision drift between training and
inference. Pipelines that train in BF16 and serve in FP8 accumulate per-layer
numerical disagreement, which compounds into divergent log-probabilities and
gradients pointing in unintended directions.

Miles supports a unified low-precision path where rollout and training share
the same quantization logic on the forward pass. Backward passes and master
weights stay in BF16. The same path is wired up for three formats today —
**block-wise FP8**, **MXFP8**, and (experimental) **NVFP4** — plus the
lower-friction "BF16 train + FP8 inference" mode that's useful when standing
up a new model architecture.

## Choose a precision

| Format | Block layout | Hardware | Models tested | Maturity |
|---|---|---|---|---|
| **BF16** | — | All NVIDIA + AMD MI300X / MI325 / MI350 / MI355X | All | Baseline |
| **FP8 block-wise** (DeepSeek-style) | 128×128, FP32 scales | Hopper (H100 / H200), Blackwell (B200+) | Qwen3-4B, Qwen3-30B-A3B, DeepSeek-V3 / R1 | Generally available |
| **MXFP8** | 1×32, UE8M0 scales | Blackwell only (B200, B300, GB200, GB300) | Qwen3-30B-A3B | Beta |
| **NVFP4** (E2M1) | 1×16, two-level (FP8 + FP32) scales, MoE experts only | Blackwell, following the TransformerEngine NVFP4 reference | — | Experimental |

## Rollout × training compatibility

Each row is a rollout (inference) precision; each column is the trainer's
forward precision. ✅ = supported; ✗ = not supported.

| Rollout \ Train | BF16 | FP8 block-wise | MXFP8 | NVFP4 |
|---|---|---|---|---|
| **BF16**           | ✅ baseline | ✗ | ✗ | ✗ |
| **FP8 block-wise** | ✅ | ✅ Hopper + Blackwell | ✗ | ✗ |
| **MXFP8**          | ✅ | ✗ | ✅ Blackwell | ✗ |
| **NVFP4**          | ✗ | ✗ | ✗ | 🚧 coming soon |

Two rules enforced in the reference script
(`scripts/run_qwen3_30b_a3b.py`):

* `--rollout-mxfp8` and `--rollout-fp8` are mutually exclusive.
* `--train-mxfp8` requires `--rollout-mxfp8` (no MXFP8-train + FP8-rollout
  combo).

## Unified training recipe

| Stage | Typical pipeline | Miles unified low-precision |
|---|---|---|
| Rollout (forward) | FP8 / MXFP8 GEMM | FP8 / MXFP8 GEMM |
| Trainer (forward) | BF16 GEMM | FP8 / MXFP8 GEMM with matching quant config |
| Trainer (backward) | BF16 grads | BF16 backward (master weights in BF16) |
| Optimizer | BF16 master | BF16 master |

The forward pass in training matches rollout. The backward pass and master
weights remain BF16, which keeps the gradient signal stable. Weights are
re-quantized on each weight-update sync to sglang.

## Modes

### 1. BF16 train + FP8 inference

The lowest-friction path. SGLang loads FP8 weights while the trainer keeps a
BF16 `torch_dist` checkpoint. There is precision drift between the two paths;
on MoE workloads, pair this with R3 (and optionally TIS).

```bash
hf download Qwen/Qwen3-30B-A3B-FP8 --local-dir /root/Qwen3-30B-A3B-FP8

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-30B-A3B-FP8        # FP8 weights for SGLang
   --ref-load      /root/Qwen3-30B-A3B_torch_dist  # BF16 torch_dist for trainer
)
```

### 2. Unified block-wise FP8 (DeepSeek-style)

Rollout and training share the same block-wise FP8 quantization. This is the
recipe to use on Hopper, and the recipe DeepSeek-V3 / DeepSeek-R1 ship in.
Block layout is 128×128 with FP32 scales.

```bash
# Megatron / TransformerEngine
--transformer-impl transformer_engine
--bf16
--fp8-format e4m3
--fp8-recipe blockwise

# Optional, for MoE numerical stability
--use-tis
```

| Flag | Effect |
|---|---|
| `--transformer-impl transformer_engine` | Routes Megatron's forward through TransformerEngine so FP8 GEMM is engaged. |
| `--fp8-format e4m3` | Forward FP8 format used by TransformerEngine. |
| `--fp8-recipe blockwise` | 128×128 block-wise quantization; sglang must serve weights in the matching layout. |
| `--use-tis` | Truncated Importance Sampling for residual precision drift. |

Set `NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1` in the Ray runtime env to use FP32
scales (`miles/ray/actor_group.py` already sets this in the actor env).

For models that already ship 128×128 block-wise FP8 weights (DeepSeek-V3,
DeepSeek-R1, `Qwen/Qwen3-30B-A3B-FP8`), point `--hf-checkpoint` at the
block-wise FP8 directory and let SGLang autodetect. Otherwise convert with
`tools/convert_hf_to_fp8.py`.

For MoE workloads, also consider `--use-rollout-routing-replay` (R3). The
canonical recipe leaves it commented out by default but the flag is available.

Reference recipes:

* [`examples/low_precision/run-qwen3-4b-fp8.sh`](https://github.com/radixark/miles/blob/main/examples/low_precision/run-qwen3-4b-fp8.sh) — single-node Qwen3-4B.
* [`examples/low_precision/run-qwen3-30b-a3b-fp8-two-nodes.sh`](https://github.com/radixark/miles/blob/main/examples/low_precision/run-qwen3-30b-a3b-fp8-two-nodes.sh) — two-node Qwen3-30B-A3B.

### 3. Unified MXFP8 (Blackwell)

MXFP8 uses a finer block layout (1×32) with UE8M0 (power-of-two) scales packed
as `uint8`. Weights are stored as `float8_e4m3fn`. This is the format wired
into the Blackwell path of the Qwen3-30B-A3B reference script.

**Hardware**: Blackwell only — B200, B300, GB200, GB300. The reference script
asserts the GPU class on enable (`scripts/run_qwen3_30b_a3b.py`).

**Train flags** — same Megatron knobs as FP8, with `mxfp8` recipe:

```bash
--transformer-impl transformer_engine
--bf16
--fp8-format e4m3
--fp8-recipe mxfp8
```

**Rollout flags** — sglang side:

```bash
--sglang-fp8-gemm-backend triton
--sglang-moe-runner-backend cutlass
# DeepEP / DeepGEMM are not yet supported with MXFP8 in sglang;
# do not set --sglang-moe-a2a-backend deepep here.
```

**Conversion**. There is no HF-shipped MXFP8 checkpoint for Qwen3 today, so
convert from BF16 first:

```bash
python tools/convert_hf_to_mxfp8.py \
   --model-dir /root/models/Qwen3-30B-A3B \
   --save-dir  /root/models/Qwen3-30B-A3B-MXFP8
```

The tool quantizes every 2D `*.weight` tensor whose last dim is divisible by
32, except `layernorm`, `embed`, `router`, `mlp.gate.`, `norm`, `lm_head`,
`eh_proj`, `weights_proj` (`tools/convert_hf_to_mxfp8.py`). The HF
config is rewritten with:

```json
{
  "quant_method": "mxfp8",
  "fmt": "e4m3",
  "weight_block_size": [1, 32],
  "scale_fmt": "ue8m0",
  "activation_scheme": "dynamic"
}
```

**Caveats**:

* No DeepEP / DeepGEMM yet — MoE all-to-all uses the cutlass MoE runner, which
  does not currently support EP. Plan EP/PP accordingly.
* `--train-mxfp8` requires `--rollout-mxfp8` (the script enforces this).

**Reference recipe**: [`scripts/run_qwen3_30b_a3b.py`](https://github.com/radixark/miles/blob/main/scripts/run_qwen3_30b_a3b.py)
with `--rollout-mxfp8 --train-mxfp8 --hardware B200`. There is no dedicated
shell script under `examples/low_precision/` yet.

### 4. NVFP4 (experimental)

NVFP4 is FP4 E2M1 with 1D block scaling (group size 16) and a two-level scale
(per-block FP8 + per-tensor FP32), following the TransformerEngine NVFP4
reference. Today only **MoE expert GEMMs** are quantized; dense layers stay in
their original precision.

The full unified NVFP4 recipe is in development.

## Hardware support

| GPU | BF16 | FP8 block-wise | MXFP8 | NVFP4 |
|---|---|---|---|---|
| NVIDIA H100 / H200 | ✅ | ✅ | ✗ | ✗ |
| NVIDIA B200 / B300 / GB200 / GB300 | ✅ | ✅ | ✅ | 🚧 in development |
| NVIDIA A100 | ✅ | ✗ | ✗ | ✗ |
| AMD MI300X / MI325 / MI350 / MI355X | ✅ | ✗ | ✗ | ✗ |

## When BF16 is enough

* Dense models below ~30 B.
* A100 hardware (no FP8 GEMM).
* AMD hardware today.
* Bring-up of a new model architecture, where clean BF16 numerics simplify
  debugging.

