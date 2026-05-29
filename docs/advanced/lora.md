---
title: LoRA Training and Serving
description: Train LoRA adapters with miles SFT or RL recipes and serve them through SGLang from the same checkpoint.
---

# LoRA Training and Serving

Miles supports LoRA adapters for both SFT and RL recipes. Adapters trained by
miles load directly into SGLang for rollout, so there is no separate merge or
conversion step in the training-serving loop.

This page is a stub; the full LoRA tutorial is being written. In the meantime,
the pieces below are enough to get a recipe running.

## Example launchers

The canonical LoRA recipes live under
[`examples/lora/`](https://github.com/radixark/miles/tree/main/examples/lora) in
the miles repo:

- `examples/lora/run-qwen2.5-0.5B-megatron-lora.sh` — small dense, single GPU.
- `examples/lora/run-qwen3-4B-megatron-lora.sh` — Qwen3-4B, RL with LoRA.
- `examples/lora/run-gpt-oss-20B-megatron-moe-lora.sh` — MoE example.

## Key flags

| Flag | Purpose |
|---|---|
| `--lora-rank` | LoRA rank. Typical values: 8, 16, 32, 64. |
| `--lora-alpha` | LoRA alpha. Usually 2 x rank. |
| `--lora-dropout` | Dropout on the LoRA path. Set to `0.0` for RL training. |
| `--lora-type` | LoRA variant: `lora` (merged QKV / gated-MLP) or `canonical_lora` (split Q / K / V). Default `lora`. |
| `--target-modules` | Which linear layers receive adapters. Required when `--lora-rank > 0`. Accepts `all-linear` or a comma-separated list (HF names like `q_proj,k_proj,v_proj,o_proj` or Megatron names like `linear_qkv,linear_proj`). |
| `--exclude-modules` | Comma-separated names to subtract from `--target-modules`. |
| `--lora-adapter-path` | Path to a pre-trained adapter to resume from. |
| `--lora-sync-from-tensor` | Sync adapter weights to SGLang via in-memory tensors instead of a file round-trip. |

Two existing arguments also have LoRA-specific requirements that are easy to
miss: the launcher has to pass `--megatron-to-hf-mode bridge` (the LoRA path
goes through Megatron-Bridge's PEFT integration; the default `raw` converter
does not understand LoRA layers), and the Ray job has to run with
`--colocate`. Distributed (PD-disaggregated) rollout with LoRA is not
supported today.

## MoE

For MoE models, attach LoRA to the FFN expert projections and switch the
SGLang LoRA backend to triton:

```bash
LORA_ARGS=(
   --lora-rank 32
   --lora-alpha 32
   --lora-dropout 0.0
   --target-modules "gate_proj,up_proj,down_proj"
   --sglang-lora-backend triton  # required for MoE LoRA
   --megatron-to-hf-mode bridge
)
```

The default SGLang LoRA backend skips MoE layers and logs
`Current LoRA backend does not support LoRA on MoE layers; skipping MoE layer`,
which means the expert adapters get silently dropped at inference time. The
GPT-OSS-20B example launcher sets `--sglang-lora-backend triton` for this
reason.

## Compatibility and limitations

* **Training backend**: Megatron only. The FSDP backend does not have a LoRA
  path yet.
* **Rollout topology**: colocate only. Distributed / PD-disaggregated rollout
  raises `NotImplementedError` at weight-sync time when LoRA is enabled.
* **Algorithms**: orthogonal to the advantage estimator; the GRPO recipes in
  `examples/lora/` carry straight over to PPO and any other algorithm that
  drives `train.py`.
* **Low-precision training**: the LoRA branch follows the surrounding
  precision, so block-wise FP8, MXFP8, and INT4 QAT recipes are compatible.
  See [Low Precision RL](fp8-low-precision.md) and [INT4 QAT](int4-qat.md).
* **`--target-modules` is mandatory** when `--lora-rank > 0`. There is no
  auto-detection; the launcher asserts at startup.
* **Single adapter per run**: multi-LoRA training in a single job is not
  implemented today.

## Internals

The bridge between Megatron's LoRA path and SGLang adapter loading is in:

- `miles/backends/megatron_utils/lora_utils.py` — argument parsing helpers,
  LoRA detection (`is_lora_enabled`, `is_lora_model`), and HF ↔ Megatron
  module-name conversion for both the `lora` and `canonical_lora` variants.
- `miles/backends/megatron_utils/bridge_lora_helpers.py` — the Megatron-Bridge
  PEFT hook that wraps the model with LoRA layers before training.
- `miles/backends/megatron_utils/checkpoint.py` — adapter-aware save and load.
- `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py`
  — colocate-mode weight sync from the trainer's LoRA tensors into the SGLang
  rollout engine. We will merge this [PR](https://github.com/radixark/miles/pull/988) soon to support disaggregate mode.

A worked tutorial covering checkpoint conversion, SGLang adapter loading, and
LoRA-specific evaluation will land here in a future doc pass.
