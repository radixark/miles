---
title: MXFP8 and NVFP4 RL
description: Blackwell-native low-precision RL across checkpoint conversion, training, rollout, and live weight updates.
---

Low-precision RL has a stricter consistency requirement than standalone
training or serving. Checkpoint conversion, the trainer forward pass, rollout,
and live weight updates must use the same quantization contract; otherwise the
policy that generates samples differs from the policy being optimized.

Miles provides two Blackwell-native recipes:

* **MXFP8** across rollout, forward propagation, weight-gradient GEMMs, and
  data-gradient GEMMs.
* **NVFP4** for routed MoE expert weights and activations, with BF16 used for
  the rest of the model and for backward GEMMs.

Both recipes support tensor-level BF16 exceptions and matching checkpoint
conversion and live-export paths. See the [public roadmap](https://github.com/radixark/miles/issues/615)
for current integration status.

## Choose a recipe

| Recipe | Quantization | Scope in the reference recipe | Hardware | Starting point |
|---|---|---|---|---|
| **BF16** | BF16 | Entire model | All supported GPUs | Model bring-up and numerical baseline |
| **Block-wise FP8** | E4M3 with 128x128 weight blocks | Trainer and rollout | Hopper and Blackwell | Existing FP8 checkpoints and Hopper systems |
| **MXFP8** | E4M3 values with one E8M0 scale per 32-value block | Major trainer and rollout GEMMs, except configured BF16 tensors | B200, B300, GB200, GB300 | Blackwell-native 8-bit RL |
| **NVFP4** | E2M1 values with E4M3 block scales and an outer FP32 scale | Routed MoE experts; other layers remain BF16 | B200, B300, GB200, GB300 | Blackwell-native 4-bit MoE RL |

Start with BF16 when bringing up a new model. Move to MXFP8 when broad
low-precision GEMM coverage is the goal. Use NVFP4 when MoE expert memory and
rollout bandwidth dominate and the model can tolerate the more aggressive
format.

## The shared precision contract

The low-precision setting is not only a trainer flag. Keep these four stages
aligned:

1. **Checkpoint conversion** creates the quantized Hugging Face checkpoint and
   records which tensors remain in BF16.
2. **Training** applies the matching Transformer Engine recipe to the same
   tensors during the forward pass.
3. **Rollout** loads the converted checkpoint in SGLang and uses a compatible
   kernel and scaling layout.
4. **Live weight export** re-quantizes updated Megatron weights with the same
   layout and BF16 exceptions before synchronization.

The backward pass and optimizer master weights can remain in higher precision.
This does not violate the rollout/trainer policy contract because sampling only
depends on the forward path.

## Block-wise FP8

The existing Hopper-compatible recipe uses E4M3 values with 128x128 weight
blocks and FP32 scales. It remains useful for models that already publish
DeepSeek-style FP8 checkpoints and for H100 or H200 systems. The maintained
examples are:

* [`run-qwen3-4b-fp8.sh`](https://github.com/radixark/miles/blob/main/examples/infra_features/low_precision/run-qwen3-4b-fp8.sh)
  for a single-node dense model.
* [`run-qwen3-30b-a3b-fp8-two-nodes.sh`](https://github.com/radixark/miles/blob/main/examples/infra_features/low_precision/run-qwen3-30b-a3b-fp8-two-nodes.sh)
  for a two-node MoE model.

The rest of this page focuses on the Blackwell-native MXFP8 and NVFP4 paths.

## MXFP8

MXFP8 uses one-dimensional microscaling blocks: 32 consecutive E4M3 values
share one E8M0 scale. Transformer Engine independently quantizes row-wise and
column-wise views when low-precision backward GEMMs need both orientations.

### Run the reference recipe

The Qwen3-30B-A3B launcher is the most direct starting point. Use the same
precision switches for preparation and execution:

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

The launcher configures Transformer Engine with:

```bash
--transformer-impl transformer_engine
--bf16
--fp8-format e4m3
--fp8-recipe mxfp8
```

It also selects the SGLang backend and parallel layout used by the reference
recipe. Treat those settings as a tested combination before changing
parallelism or kernels independently.

To convert a checkpoint outside the launcher:

```bash
python tools/convert_hf_to_mxfp8.py \
  --model-dir /root/models/Qwen3-30B-A3B \
  --save-dir /root/models/Qwen3-30B-A3B-MXFP8
```

The converter writes a Hugging Face quantization config with 1x32 weight
blocks and E8M0 scales. It skips tensors that are not suitable for the format,
including norms, embeddings, routers, and configured high-precision layers.

### MXFP8 backward modes

The default Transformer Engine MXFP8 path uses low-precision backward GEMMs.
Miles also supports two override modes:

* `NVTE_BACKWARD_OVERRIDE=high_precision` uses the original BF16 operands.
* `NVTE_BACKWARD_OVERRIDE=dequantized` uses BF16 dequantizations of the exact
  low-precision forward operands.

![MXFP8 with high-precision backward](/assets/images/low-precision/mxfp8-high-precision-backward.png)

![MXFP8 with dequantized backward](/assets/images/low-precision/mxfp8-dequantized-backward.png)

*High-precision and dequantized MXFP8 backward modes. Source: the
[LMSYS post](https://www.lmsys.org/blog/2026-07-29-mxfp8-nvfp4-rl).*

Use an override only after establishing the default recipe baseline. The modes
trade low-precision backward throughput for different numerical and memory
behavior.

## NVFP4

NVFP4 stores E2M1 values in 16-value blocks. Each block has an E4M3 scale, and
an outer FP32 scale covers a larger scope. In the Miles RL recipe:

* routed MoE experts use NVFP4;
* dense layers, attention, routers, norms, and other unmatched tensors stay in
  BF16;
* activations use one outer FP32 scale per token rather than one shared scale
  for an entire tensor; and
* gate and up projections are quantized together so their fused rollout GEMM
  uses one consistent outer scale.

![NVFP4 two-level scaling](/assets/images/low-precision/nvfp4-two-level-scaling.png)

*NVFP4 two-level scaling. Source: the [LMSYS post](https://www.lmsys.org/blog/2026-07-29-mxfp8-nvfp4-rl).*

### Run the reference recipe

Use the paired rollout and trainer switches for preparation and execution:

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
2D quantization, RHT, and stochastic-rounding paths, and uses
high-precision backward by default. It also keeps the NVFP4 trainer and rollout
on separate GPU sets.

For manual integrations, mirror the launcher's base environment on every
component that participates in conversion, training, or rollout:

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

The training-side precision config applies NVFP4 only to routed expert FC1 and
FC2 tensors and falls back to BF16 for everything else. The current definition
lives in [`scripts/run_qwen3_30b_a3b.py`](https://github.com/radixark/miles/blob/main/scripts/run_qwen3_30b_a3b.py).

To convert a checkpoint outside the launcher:

```bash
python tools/convert_hf_to_nvfp4.py \
  --model-dir /root/models/Qwen3-30B-A3B \
  --save-dir /root/models/Qwen3-30B-A3B-NVFP4
```

Run conversion with the same `NVTE_*` recipe variables that training uses.
The rollout process must receive the matching `FLASHINFER_*` variables.

### NVFP4 backward modes

Miles currently uses BF16 backward GEMMs for the NVFP4 RL path:

* **High-precision backward** (`NVTE_BACKWARD_OVERRIDE=high_precision`) uses
  the original BF16 forward inputs as backward operands. This is the Qwen3
  launcher default.
* **Dequantized backward** (`NVTE_BACKWARD_OVERRIDE=dequantized`) uses BF16
  dequantizations of the NVFP4 forward operands. This more closely follows the
  exact forward path but adds dequantization work.

![NVFP4 with high-precision backward](/assets/images/low-precision/nvfp4-high-precision-backward.png)

![NVFP4 with dequantized backward](/assets/images/low-precision/nvfp4-dequantized-backward.png)

*High-precision and dequantized NVFP4 backward modes. Source: the
[LMSYS post](https://www.lmsys.org/blog/2026-07-29-mxfp8-nvfp4-rl).*

To try dequantized backward with the reference launcher:

```bash
export NVTE_BACKWARD_OVERRIDE=dequantized
```

Set the variable in the environment used to launch the job; the launcher
forwards `NVTE_*` variables to the training actors.

### Advanced: four-over-six

Four-over-six (4/6) is an optional NVFP4 scaling technique. For each block, it
chooses whether mapping the largest FP4 magnitude to 4 or 6 produces lower
quantization error. Miles can apply it to weights and activations, but the
trainer, checkpoint converter, and rollout kernels must make the same choice
bit for bit.

Treat 4/6 as an advanced recipe. First validate the base NVFP4 path, then enable
the paired Transformer Engine and FlashInfer settings together:

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

Keep the base recipe's exact-quantization settings enabled as well. Apply this
environment to checkpoint conversion, training, and rollout, and verify that
the installed Transformer Engine and FlashInfer versions implement the same
4/6 contract. The [humans& NVFP4 RL post](https://humansand.ai/blog/nvfp4-rl)
explains the motivation, numerical trade-offs, and combined recipe.

## Fine-grained BF16 exceptions

Low-precision coverage should be consistent rather than maximal. Miles can
keep selected tensors in BF16 across conversion, training, rollout, and live
weight export.

The main controls are:

| Control | Purpose |
|---|---|
| `--num-layers-at-start-in-bf16` | Keep an initial layer range in BF16. |
| `--num-layers-at-end-in-bf16` | Keep a final layer range in BF16. |
| `--first-last-layers-bf16` | Apply the corresponding Megatron training rule. |
| `--extra-high-precision-layers-hf` | Exclude matching Hugging Face tensor names during conversion and rollout export. |
| `--extra-high-precision-layers-megatron` | Exclude matching Megatron tensor names during training and live export. |
| `--te-precision-config-file` | Select Transformer Engine recipes by Megatron tensor name. |

Use equivalent Hugging Face and Megatron name matchers. A tensor left in BF16
on only one side reintroduces train-rollout mismatch. Common exceptions include
the final transformer layers, shared experts, and MLA projections whose
contraction axis does not match a one-dimensional scaling layout.

## Limitations

* MXFP8 and NVFP4 require NVIDIA Blackwell GPUs in the reference launcher.
* The MXFP8 SGLang reference path uses the CUTLASS MoE runner and does not
  enable expert parallelism, DeepEP, or DeepGEMM.
* The NVFP4 reference path quantizes routed experts rather than every linear
  layer and uses BF16 KV cache.
* Low-precision parameter gather is not part of these reference recipes;
  training retains higher-precision master weights.
* Kernel, parallelism, and environment changes can alter the quantization
  contract. Change one dimension at a time and compare against a BF16 baseline.

## Further reading

* [Blackwell MXFP8 and NVFP4 RL roadmap](https://github.com/radixark/miles/issues/615)
* [Towards Blackwell-Native 8-bit and 4-bit RL](https://www.lmsys.org/blog/2026-07-29-mxfp8-nvfp4-rl)
* [NVFP4 RL recipe and four-over-six analysis](https://humansand.ai/blog/nvfp4-rl)
