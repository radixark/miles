---
title: Low Precision RL
description: Unified low-precision pipelines for RL — block-wise FP8, MXFP8, and NVFP4 across rollout and training.
---
A common failure mode in MoE RL is precision drift between training and
inference. Pipelines that train in BF16 and serve in FP8 accumulate per-layer
numerical disagreement, which compounds into divergent log-probabilities and
gradients pointing in unintended directions.

Miles supports a unified low-precision path where rollout and training share
the same quantization logic on the forward pass. The same path is wired up for
three formats today — **block-wise FP8**, **MXFP8**, and **NVFP4** — plus the
lower-friction "BF16 train + FP8 inference" mode that's useful when standing
up a new model architecture.

## Choose a precision

| Format | Block layout | Hardware | Models tested | Maturity |
|---|---|---|---|---|
| **BF16** | — | All NVIDIA + AMD MI300X / MI325 / MI350 / MI355X | All | Baseline |
| **FP8 block-wise** (DeepSeek-style) | 128×128, FP32 scales | Hopper (H100 / H200), Blackwell (B200+) | Qwen3-4B, Qwen3-30B-A3B | Generally available |
| **MXFP8** | 1×32, UE8M0 scales | Blackwell only (B200, B300, GB200, GB300) | Qwen3-30B-A3B, DeepSeek-V3.2 | Beta |
| **NVFP4** (E2M1) | 1×16, two-level (FP8 + FP32) scales | Blackwell only (B200, B300, GB200, GB300) | Qwen3-30B-A3B | Beta |

## Rollout × training compatibility

Each row is a rollout (inference) precision; each column is the trainer's
forward precision. ✅ = supported; ✗ = not supported.

| Rollout \ Train | BF16 | FP8 block-wise | MXFP8 | NVFP4 |
|---|---|---|---|---|
| **BF16**           | ✅ baseline | ✗ | ✗ | ✗ |
| **FP8 block-wise** | ✅ | ✅ Hopper + Blackwell | ✗ | ✗ |
| **MXFP8**          | ✅ | ✗ | ✅ Blackwell | ✗ |
| **NVFP4**          | ✗ | ✗ | ✗ | ✅ Blackwell |

The reference script (`scripts/run_qwen3_30b_a3b.py`) allows only one rollout
precision and one training precision at a time. Use paired rollout and training
flags for the end-to-end MXFP8 and NVFP4 recipes below.

## Unified training recipe

| Stage | Typical pipeline | Miles unified low-precision |
|---|---|---|
| Rollout (forward) | FP8 / MXFP8 / NVFP4 GEMM | Matching low-precision GEMM |
| Trainer (forward) | BF16 GEMM | Matching low-precision GEMM |
| Trainer (backward) | BF16 grads | Recipe-specific precision |
| Optimizer | BF16 master | Higher-precision master weights |

The precision contract covers checkpoint conversion, the trainer forward pass,
SGLang rollout, and live weight export. High-precision tensor exceptions must
match across all four stages.

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

Reference recipe:
[`examples/infra_features/low_precision/run-qwen3-4b-fp8.sh`](https://github.com/radixark/miles/blob/main/examples/infra_features/low_precision/run-qwen3-4b-fp8.sh)
— single-node Qwen3-4B. It serves an FP8 checkpoint to SGLang and trains from a
BF16 `torch_dist` checkpoint; it sets no `--fp8-recipe`, so the trainer forward
stays BF16.

### 2. Unified block-wise FP8 (DeepSeek-style)

Rollout and training share the same block-wise FP8 quantization. This is the
recipe to use on Hopper, and the layout DeepSeek ships its FP8 checkpoints in.
Block layout is 128×128 with FP32 scales.

```bash
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

`NVTE_FP8_BLOCK_SCALING_FP32_SCALES` is set for you in the actor env
(`miles/ray/train/actor_factory.py`), defaulting by hardware: `1` on Hopper, and
`0` on Blackwell, where TransformerEngine emulates the block-wise recipe with
MXFP8 and needs power-of-two scales. Override it only if you know you want the
non-default for your GPU.

For models that already ship 128×128 block-wise FP8 weights (DeepSeek-V3.2,
`Qwen/Qwen3-30B-A3B-FP8`), point `--hf-checkpoint` at the
block-wise FP8 directory and let SGLang autodetect. Otherwise convert with
`tools/convert_hf_to_fp8.py`.

For MoE workloads, also consider `--use-rollout-routing-replay` (R3). The
canonical recipe leaves it commented out by default but the flag is available.

Reference recipe:
[`examples/infra_features/low_precision/run-qwen3-30b-a3b-fp8-two-nodes.sh`](https://github.com/radixark/miles/blob/main/examples/infra_features/low_precision/run-qwen3-30b-a3b-fp8-two-nodes.sh)
— two-node Qwen3-30B-A3B.

### 3. Unified MXFP8 (Blackwell)

MXFP8 uses one-dimensional microscaling blocks: 32 consecutive E4M3 values
share one UE8M0 scale. The end-to-end recipe uses MXFP8 for rollout, forward
propagation, weight-gradient GEMMs, and data-gradient GEMMs while preserving
configured tensors in BF16.

![End-to-end MXFP8 RL recipe](/assets/images/low-precision/mxfp8-e2e.png)

**Hardware:** B200, B300, GB200, or GB300.

The Qwen3-30B-A3B launcher prepares the MXFP8 Hugging Face checkpoint and
configures both sides of the RL loop:

```bash
python scripts/run_qwen3_30b_a3b.py prepare \
  --hardware B200 \
  --rollout-mxfp8 \
  --train-mxfp8

python scripts/run_qwen3_30b_a3b.py execute \
  --hardware B200 \
  --rollout-mxfp8 \
  --train-mxfp8
```

The training path uses:

```bash
--transformer-impl transformer_engine
--bf16
--fp8-format e4m3
--fp8-recipe mxfp8
```

SGLang selects the MoE and dense linear GEMM backends independently. Common
MXFP8 choices are:

| Workload | Miles flag | Backends |
|---|---|---|
| MoE | `--sglang-moe-runner-backend` | `flashinfer_trtllm_routed`, `flashinfer_trtllm`, `cutlass` |
| Dense linear GEMM | `--sglang-fp8-gemm-backend` | `flashinfer_trtllm`, `flashinfer_cutlass`, `triton` |

Choose the pair that matches the model, parallel layout, and installed SGLang
stack. Convert a checkpoint outside the launcher with:

```bash
python tools/convert_hf_to_mxfp8.py \
  --model-dir /root/models/Qwen3-30B-A3B \
  --save-dir /root/models/Qwen3-30B-A3B-MXFP8
```

The converter records a 1x32 weight block and UE8M0 scale layout. It excludes
norms, embeddings, routers, and configured high-precision tensors.

Low-precision parameter gather is not enabled in the reference recipe, so a
higher-precision master weight copy remains present during training.

### 4. NVFP4 (Blackwell)

NVFP4 stores E2M1 values in 16-value blocks with an E4M3 scale for each block
and an outer FP32 scale. The Miles reference recipe combines NVFP4 and BF16
through tensor-level precision configuration.

Activation scaling is computed per token. Gate and up projections are
quantized together so the fused rollout GEMM uses the same outer weight scale.
The trainer, checkpoint converter, and rollout kernels must use the same
scaling contract.

**Hardware:** B200, B300, GB200, or GB300.

Run the Qwen3-30B-A3B reference recipe with paired rollout and training flags:

```bash
python scripts/run_qwen3_30b_a3b.py prepare \
  --hardware B200 \
  --rollout-nvfp4 \
  --train-nvfp4

python scripts/run_qwen3_30b_a3b.py execute \
  --hardware B200 \
  --rollout-nvfp4 \
  --train-nvfp4
```

The launcher selects per-token activation scaling, disables the incompatible
2D quantization, RHT, and stochastic-rounding paths, and loads the matching
tensor-level precision configuration. Unmatched tensors stay in BF16, and
rollout uses a BF16 KV cache.

The training path uses:

```bash
--transformer-impl transformer_engine
--bf16
--fp4-format e2m1
--fp4-recipe nvfp4
```

Note the `--fp4-` prefix: NVFP4 has its own format and recipe flags rather than
reusing the `--fp8-` pair. The launcher also adds `--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer`.

The base NVFP4 recipe uses **high-precision backward**: the forward pass uses
NVFP4 while the BF16 backward GEMMs consume the original BF16 operands.

![NVFP4 with high-precision backward](/assets/images/low-precision/nvfp4-high-precision-backward.png)

The base recipe settings used by the launcher are:

```bash
NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1
NVTE_NVFP4_DISABLE_RHT=1
NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=1
NVTE_NVFP4_ROW_SCALED_ACTIVATION=1
NVTE_BACKWARD_OVERRIDE=high_precision
NVTE_USE_FAST_MATH=0
SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1
TRTLLM_DISABLE_FP4_QUANT_FAST_MATH=1
FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH=1
```

Convert a checkpoint outside the launcher with the same `NVTE_*` recipe
environment:

```bash
python tools/convert_hf_to_nvfp4.py \
  --model-dir /root/models/Qwen3-30B-A3B \
  --save-dir /root/models/Qwen3-30B-A3B-NVFP4
```

#### Advanced: dequantized backward

Dequantized backward keeps the backward GEMMs in BF16 but uses BF16
dequantizations of the NVFP4 operands produced during the forward pass.
See the humans& discussion of
[gradient stability](https://humansand.ai/blog/nvfp4-rl#improving-gradient-stability)
for the motivation and ablations.

![NVFP4 with dequantized backward](/assets/images/low-precision/nvfp4-dequantized-backward.png)

Select this mode in the environment used to launch the job:

```bash
export NVTE_BACKWARD_OVERRIDE=dequantized
```

#### Advanced: four-over-six

[Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling](https://arxiv.org/abs/2512.02010)
optionally chooses, for each NVFP4 block, whether mapping the largest FP4
magnitude to 4 or 6 produces less quantization error.
See the humans&
[four-over-six analysis](https://humansand.ai/blog/nvfp4-rl#four-over-six-for-rl-weights-and-activations)
for the RL recipe discussion.

An example setting is:

```bash
export NVTE_NVFP4_4OVER6=all
export FLASHINFER_NVFP4_4OVER6=1
export NVTE_NVFP4_4OVER6_E4M3_USE_256=all
export FLASHINFER_NVFP4_4OVER6_E4M3_USE_256=1
export NVTE_NVFP4_4OVER6_ERR_MODE=MSE
export FLASHINFER_NVFP4_4OVER6_ERR_MODE=MSE
export NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH=0
export FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH=0
```

## Fine-grained BF16 exceptions

Miles supports per-layer precision configuration across checkpoint
conversion, Megatron training, SGLang rollout, and live weight export.

| Control | Purpose |
|---|---|
| `--num-layers-at-start-in-bf16` | Keep an initial layer range in BF16. |
| `--num-layers-at-end-in-bf16` | Keep a final layer range in BF16. |
| `--first-last-layers-bf16` | Apply the corresponding Megatron training rule. |
| `--extra-high-precision-layers-hf` | Exclude matching Hugging Face tensor names during conversion and rollout export. |
| `--extra-high-precision-layers-megatron` | Exclude matching Megatron tensor names during training and live export. |
| `--te-precision-config-file` | Select Transformer Engine recipes by Megatron tensor name. |

Use equivalent Hugging Face and Megatron name matchers. Common exceptions
include final transformer layers, shared experts, and MLA projections whose
contraction axis does not match a one-dimensional scaling layout.

## Hardware support

| GPU | BF16 | FP8 block-wise | MXFP8 | NVFP4 |
|---|---|---|---|---|
| NVIDIA H100 / H200 | ✅ | ✅ | ✗ | ✗ |
| NVIDIA B200 / B300 / GB200 / GB300 | ✅ | ✅ | ✅ | ✅ |
| NVIDIA A100 | ✅ | ✗ | ✗ | ✗ |
| AMD MI300X / MI325 / MI350 / MI355X | ✅ | ✗ | ✗ | ✗ |

## When BF16 is enough

* Dense models below ~30 B.
* A100 hardware (no FP8 GEMM).
* AMD hardware today.
* Bring-up of a new model architecture, where clean BF16 numerics simplify
  debugging.

## Further reading

* [Blackwell MXFP8 and NVFP4 RL roadmap](https://github.com/radixark/miles/issues/615)
* [Towards Blackwell-Native 8-bit and 4-bit RL: End-to-End MXFP8 and NVFP4 RL in Miles](https://www.lmsys.org/blog/2026-07-29-mxfp8-nvfp4-rl)
* [The 4-bitter Lesson](https://humansand.ai/blog/nvfp4-rl)
