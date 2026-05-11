---
title: LoRA Training and Serving
description: Train LoRA adapters with miles RL recipes on Megatron and serve them through SGLang in colocate mode, with full GRPO / PPO support across dense and MoE models.
---

# LoRA Training and Serving

LoRA (Low-Rank Adaptation) freezes the base-model weights and trains a pair of
low-rank update matrices on a chosen set of linear layers. The adapter is
small, cheap to checkpoint, and at inference time it composes with the frozen
base model — which makes it a good fit for RL post-training, where the trainer
and rollout engine sit on the same GPUs and need to exchange weights every
step.

Miles wires LoRA through the Megatron training backend and the SGLang rollout
engine. The same launcher you use for full-parameter RL runs unchanged with
LoRA enabled; the only difference is the set of LoRA flags described below and
the requirement to run in `--colocate` mode.

## When to use LoRA

You can prefer LoRA over full-parameter RL when:

* The base model is too large to keep a full BF16 optimizer state on the
  cluster you have (LoRA only updates ~0.1 – 1% of parameters).
* You want to ship several task-specialized adapters from the same base
  checkpoint without re-training the base.
* You want a fast turn-around on a new reward signal — adapter updates
  converge quickly and the resulting artifact is small enough to swap in and
  out of an inference server.

Stick to full-parameter training when the task requires changes that span
attention, FFN, and norm/embedding layers simultaneously, or when the base
model is small enough that the cost saving doesn't matter.

## Quick start

The canonical entry point is `examples/lora/run-qwen2.5-0.5B-megatron-lora.sh`
in the miles repo. It runs Qwen2.5-0.5B with GRPO on GSM8K, on a single
8-GPU node, with LoRA enabled.

The LoRA-specific arguments in that launcher are:

```bash
CKPT_ARGS=(
   --hf-checkpoint /root/Qwen2.5-0.5B-Instruct/
   --megatron-to-hf-mode bridge
)

LORA_ARGS=(
   --lora-rank 32        # LoRA rank (typical: 8, 16, 32, 64)
   --lora-alpha 32       # LoRA alpha (often 2x rank; equal-to-rank also common)
   --lora-dropout 0.0    # Set to 0.0 for RL
   --target-modules "all-linear"
   --megatron-to-hf-mode bridge
)
```

The launcher then submits the Ray job with `--colocate` and the standard GRPO
flags. The only constraint LoRA adds on top of a regular GRPO run is that the
checkpoint conversion mode has to be `bridge` (the path that goes through
[Megatron-Bridge](https://github.com/NVIDIA/Megatron-Bridge)'s PEFT
integration), and the rollout engine has to be colocated with the trainer.

## Configuration reference

All LoRA-specific flags are defined in `miles/utils/arguments.py`.

| Flag | Default | Description |
|---|---|---|
| `--lora-rank` | `0` | LoRA rank. Setting to `0` disables LoRA. Typical values are 8, 16, 32, 64. |
| `--lora-alpha` | `16` | LoRA scaling factor. A common rule of thumb is `2 * rank`, but `rank == alpha` works well in practice for RL. |
| `--lora-dropout` | `0.0` | Dropout on the LoRA branch. Leave at `0.0` for RL; small non-zero values are sometimes used for SFT. |
| `--lora-type` | `lora` | LoRA variant. `lora` uses Megatron's merged QKV / gated-MLP layout. `canonical_lora` splits Q, K, V (and gate / up) into independent adapters, matching HF PEFT layouts more directly. |
| `--target-modules` | `None` | Which linear layers receive adapters. Required when `--lora-rank > 0`. Accepts `all-linear`, or a comma-separated list using HF names (`q_proj,k_proj,v_proj,o_proj`) or Megatron names (`linear_qkv,linear_proj`). |
| `--exclude-modules` | `None` | Comma-separated names to subtract from `--target-modules`. |
| `--lora-adapter-path` | `None` | Path to a pre-trained adapter to resume from. |
| `--lora-sync-from-tensor` | `False` | Sync adapter weights to SGLang via in-memory tensors instead of via a file round-trip. Recommended when colocate is enabled. |

In addition to the flags above, two existing arguments need specific values
for LoRA:

* `--megatron-to-hf-mode bridge` — the LoRA path is implemented through
  Megatron-Bridge's PEFT integration, so the bridge converter is required.
  The default `raw` converter does not understand LoRA layers.
* `--colocate` — LoRA-aware weight syncing is only implemented for the
  colocated rollout topology. Distributed (PD-disaggregated) rollout with
  LoRA is not supported today.

## Dense and MoE

Miles supports LoRA on both dense and MoE architectures, with a small set of
extra knobs for MoE.

### Dense models

For dense models, `--target-modules all-linear` is a sensible default — it
attaches adapters to `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj`, and `down_proj`. The Qwen2.5-0.5B and Qwen3-4B example launchers
both use this setting.

Reference recipes:

* [`examples/lora/run-qwen2.5-0.5B-megatron-lora.sh`](https://github.com/radixark/miles/blob/main/examples/lora/run-qwen2.5-0.5B-megatron-lora.sh)
  — single-node Qwen2.5-0.5B with GRPO on GSM8K.
* [`examples/lora/run-qwen3-4B-megatron-lora.sh`](https://github.com/radixark/miles/blob/main/examples/lora/run-qwen3-4B-megatron-lora.sh)
  — single-node Qwen3-4B with GRPO on GSM8K.

### MoE models

For MoE models, the FFN experts are usually the part of the network you want
to adapt, so the typical target list is the expert projections only:

```bash
LORA_ARGS=(
   --lora-rank 32
   --lora-alpha 32
   --lora-dropout 0.0
   --target-modules "gate_proj,up_proj,down_proj"
   --sglang-lora-backend triton  # required for MoE LoRA on SGLang
   --megatron-to-hf-mode bridge
)
```

The `--sglang-lora-backend triton` flag is required on SGLang for MoE LoRA.
The default backend does not support LoRA on MoE layers and will log
`Current LoRA backend does not support LoRA on MoE layers; skipping MoE layer`,
which means the expert adapters get silently dropped at inference.

Reference recipe:

* [`examples/lora/run-gpt-oss-20B-megatron-moe-lora.sh`](https://github.com/radixark/miles/blob/main/examples/lora/run-gpt-oss-20B-megatron-moe-lora.sh)
  — single-node GPT-OSS-20B MoE with GRPO and LoRA on the FFN experts.

## Compatibility

| Axis | Supported | Notes |
|---|---|---|
| Training backend | Megatron (`mcore`) | FSDP does not have a LoRA path today. |
| Rollout topology | Colocate | Distributed / PD-disaggregated rollout with LoRA raises `NotImplementedError` at weight-sync time. |
| Algorithms | GRPO, PPO, and any algorithm that drives `train.py` | LoRA is orthogonal to the advantage estimator. |
| Low-precision training | FP8 block-wise, MXFP8, INT4 QAT | The LoRA branch follows the surrounding precision. See [Low Precision RL](fp8-low-precision.md) and [INT4 QAT](int4-qat.md). |
| Model family | Dense (Qwen2.5, Qwen3, Llama-family, …) and MoE (GPT-OSS, Qwen3-MoE, …) | MoE requires `--sglang-lora-backend triton`. |
| Adapter resume | `--lora-adapter-path` | Loads HF PEFT-format adapters. |
| Adapter export | HF PEFT format via Megatron-Bridge | Adapter weights are saved alongside the Megatron checkpoint and can be re-loaded by SGLang or any HF PEFT consumer. |

## Limitations

The current LoRA path has a few rough edges worth knowing about up front:

* **Megatron only.** The FSDP backend does not have LoRA support yet. If you
  need LoRA today, run on Megatron.
* **Colocate only.** LoRA weight syncing assumes the trainer and SGLang
  engine share the same GPUs. Distributed rollout topologies are not
  supported.
* **`--target-modules` is mandatory.** Miles does not auto-detect a sensible
  target set; you have to either pass `all-linear` or list the modules
  explicitly. The launcher will assert at startup if `--lora-rank > 0` and
  `--target-modules` is unset.
* **Single adapter per run.** There is one active adapter per training job;
  multi-LoRA training in a single run is not implemented.
* **MoE on the default SGLang LoRA backend silently skips experts.** This is
  a SGLang-side limitation. Use `--sglang-lora-backend triton` for MoE.

## Internals

The relevant entry points in the miles repo, for readers who want to extend
the LoRA path:

* `miles/backends/megatron_utils/lora_utils.py` — argument parsing helpers,
  detection (`is_lora_enabled`, `is_lora_model`), and HF ↔ Megatron module
  name conversion for both `lora` and `canonical_lora` variants.
* `miles/backends/megatron_utils/bridge_lora_helpers.py` — registers the
  Megatron-Bridge PEFT hook that wraps the model with LoRA layers before
  training.
* `miles/backends/megatron_utils/checkpoint.py` — adapter-aware save and
  load.
* `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py`
  — colocate-mode weight sync from the trainer's LoRA tensors into the
  SGLang rollout engine.

A deeper tutorial covering checkpoint conversion, SGLang adapter hot-swap,
and LoRA-aware evaluation will follow in a future doc pass.
