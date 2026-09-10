---
title: GLM-5.3 LoRA
description: GLM-5.3 744B-A40B LoRA RL with BF16 training and optional FP8 rollout.
---

GLM-5.3 uses the same 78-layer MoE + MLA + DSA architecture and cross-layer
index-sharing schedule as GLM-5.2. The existing GLM-5.2 LoRA launcher selects it
with `--model-name GLM-5.3`; Megatron-Bridge builds the model from its native
`glm_moe_dsa` config. [GLM-5.3-Flash](glm5-3-flash) is a separate architecture.

## Checkpoints

| Role | Hugging Face repo | Default local path |
|---|---|---|
| BF16 trainer | `zai-org/GLM-5.3-BF16` | `/root/models/GLM-5.3-BF16` |
| Optional FP8 rollout | `zai-org/GLM-5.3` | `/root/models/GLM-5.3` |

The unsuffixed GLM-5.3 checkpoint contains FP8 weights. Do not pass it directly
as the BF16 trainer's `--hf-checkpoint`. Use the official BF16 checkpoint or a
checkpoint that has already been dequantized. Override local paths with
`--hf-checkpoint` and `--fp8-rollout-checkpoint`.

```bash
python scripts/run_glm5_2_744b_a40b_lora.py prepare \
  --model-name GLM-5.3 --fp8-rollout
```

This downloads both checkpoints and GSM8K. Without `--fp8-rollout`, preparation
downloads only the BF16 checkpoint and the dataset. Checkpoints and datasets
must be available at the same paths on every participating node.

## Launch

Use Miles `main`, SGLang `sglang-miles`, and a Megatron-Bridge `bridge` checkout
with the differentiable `LoRALinear.weight` support required by absorbed MLA.
SGLang also needs the LoRA fixes for dense-MLP buffer sizing with
`--moe-dense-tp-size 1` and for DP-attention idle batches. Install each checkout
into the training image before launching.

Example topology for a full model with FP8 rollout: four nodes with eight GPUs
per node. Join all nodes to the same Ray cluster, then run on the head:

```bash
MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_glm5_2_744b_a40b_lora.py train \
  --model-name GLM-5.3 \
  --num-nodes 4 --num-gpus-per-node 8 \
  --fp8-rollout --num-rollout 2 \
  --save-dir /path/to/shared/checkpoints
```

The trainer uses TP within each node, EP across all actor GPUs, PP=1, and
ETP=1. The FP8 rollout defaults to eight GPUs per engine, with DP attention and
MoE EP enabled. The rollout YAML under the shared save directory includes every actor GPU and keeps
`update_weights: true` so adapter updates reach each engine. The example is a
configuration template; the full 78-layer training topology needs its own
hardware validation.

| Knob | Default / behavior |
|---|---|
| `--dsa-attention-backend` | `tilelang`; `megatron` is also selectable |
| `--lora-rank` / `--lora-alpha` | 16 / 32 |
| `--lora-dropout` | 0; required for training with absorbed projection weights |
| `--target-modules` | Attention, MLA, dense MLP and routed/shared expert projections |
| `--experts-shared-outer-loras` | Enabled; disable for per-expert factors |
| `--lora-base-cpu-backup` | Enabled to preserve the frozen BF16 trainer base under colocation |
| `--rollout-num-gpus-per-engine` | All actor GPUs for BF16; up to eight GPUs within a node for FP8 |
| `--sglang-mem-fraction-static` | 0.8 for the full model |

The DSA indexer is excluded from the default LoRA targets. MTP training is
disabled by the existing GLM5 bridge. BF16 and FP8 rollout have different
quantization error; assess train/rollout log-probability differences separately.

## Validation scope

Launcher tests record the GLM-5.3 preparation and four-node FP8 commands,
including the generated SGLang YAML. Model-argument tests verify that GLM-5.3
inherits the GLM-5.2 architecture. Bridge tests compare absorbed and ordinary
LoRA outputs and gradients, including TP-sharded factors.

GPU validation uses a five-layer slice retaining the official GLM-5.3 widths,
256 experts, three dense layers, and DSA index sharing. A slice exercises the
integration path; it does not establish full-model accuracy or throughput.
