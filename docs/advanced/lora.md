---
title: LoRA Training and Serving
description: Train LoRA adapters with miles SFT or RL recipes and serve them through SGLang from the same checkpoint.
---
Miles supports LoRA adapters for both SFT and RL recipes. Adapters trained by
miles load directly into SGLang for rollout, so there is no separate merge or
conversion step in the training-serving loop.

## Example launchers

The canonical LoRA recipes live under
[`examples/lora/`](https://github.com/radixark/miles/tree/main/examples/lora) in
the miles repo:

- `examples/lora/run-qwen2.5-0.5B-megatron-lora.sh` — small dense model, single GPU.
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
| `--lora-provider-path` | Dotted module supplying a model-specific native-LoRA implementation. Defaults to the generic `miles_plugins.lora` provider. |
| `--debug-lora-train-only` | Train adapters in Megatron while rollout stays on the frozen base policy. Useful when the engine cannot serve the adapter yet. |
| `--check-lora-weight-equal` | Verify the Megatron → SGLang adapter sync with a per-tensor sha256 manifest. |

<Warning>
**`--colocate`** is required. Distributed (PD-disaggregated) rollout with LoRA
is not supported today.
</Warning>

## Choosing a path: bridge or native

There are two implementations, selected by `--megatron-to-hf-mode`:

- **`bridge`** — Megatron-Bridge's PEFT integration. Use it for models the
  generic native provider cannot cover, and for the routed-expert adapter
  layouts (`--experts-shared-outer-loras`).
- **`raw`** (default) — the *native* path in the `miles_plugins/lora/` plugin.
  Adapters attach directly to the mcore model miles' own provider builds,
  before the DDP wrap, so DDP only allocates grad buffers for adapter params.
  Which architectures it serves is an explicit registry keyed on the HF
  `model_type` (`miles_plugins/lora/registry.py`); a checkpoint whose
  `model_type` is not registered fails at startup with the supported list
  rather than training unverified math. Multimodal checkpoints resolve through
  `text_config.model_type`: adapters attach to the registered language decoder
  only and vision weights are never adapted. A VL wrapper whose text model type
  is not registered (e.g. `qwen3_vl`, `qwen2_5_vl`) fails closed — use
  `--megatron-to-hf-mode bridge` for those.

Native LoRA covers standard Megatron-core attention and gated MLPs:

| `--target-modules` | Megatron module | Sharding |
|---|---|---|
| `q_proj` / `k_proj` / `v_proj` | `self_attention.linear_qkv` (fused) | column-parallel |
| `o_proj` | `self_attention.linear_proj` | row-parallel |
| `gate_proj` / `up_proj` | `mlp.linear_fc1` (fused) | column-parallel |
| `down_proj` | `mlp.linear_fc2` | row-parallel |
| `q_a_proj` / `kv_a_proj_with_mqa` | MLA down-projections | replicated |
| `q_b_proj` / `kv_b_proj` | MLA up-projections | column-parallel |

`--attention-output-gate` models (Qwen3.5 / Qwen3-Next) are covered: the gate
slice rides along in `q_proj`, which is also how the HF checkpoint stores it.
Their linear-attention (GDN) layers have no fused qkv and simply carry no
attention adapter — the startup log names them.

Two layouts are deliberately out of scope and need
`--lora-provider-path <dotted.module>` implementing
`wrap_model_provider_with_lora` / `load_lora_adapter_hf` /
`export_lora_hf_named`: routed MoE experts, and attention that is not
mcore's (DeepSeek-V4-Flash's `wq_a` / `wq_b` / `wkv` / `wo_a` / `wo_b`, which
SGLang also has no LoRA support for).

### Parallelism: what each path supports

Native handles TP, SP, PP/VPP and CP. Expert tensor parallelism is moot for it:
the only MoE module it adapts, the shared expert, shards over the *attention*
TP group.

Native refuses at startup (all inert under bridge-mode LoRA too, since the
bridge hook builds its own DDP config):

- `--overlap-grad-reduce`, `--overlap-param-gather`
- `--moe-shared-expert-overlap`

Bridge-only, each with a TODO beside the native code that would change:

- routed/grouped-expert LoRA, three layouts — `spec/moe.py`
- router LoRA — `spec/moe.py`
- expert-TP group, expert-axis export — `distributed.py`, `codec/hf.py`
- GDN `in_proj`, MLA without `q_lora_rank` — `spec/attention.py`

<Note>
On a MoE model, list only the modules the trainer actually adapts. Naming
`gate_proj` makes SGLang allocate adapter buffers for the routed experts too,
and nothing fills them.
</Note>

On a MoE model without shared experts (e.g. Qwen3-MoE), the native path has no
MLP module to adapt: `all-linear` drops its parser-added `gate_proj` /
`up_proj` / `down_proj` with a log line and trains attention-only, while an
explicitly listed MLP target fails at startup naming the bridge and
provider-path escape hatches.

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
  See [Low Precision RL](/advanced/fp8-low-precision) and [INT4 QAT](/advanced/int4-qat).
* **Target modules**: `--target-modules` is required whenever
  `--lora-rank > 0`. There is no auto-detection; the launcher asserts at
  startup.
* **Native partial targets**: raw-mode native LoRA treats fused projections
  exactly: `q_proj` does not allocate K/V adapters, and `gate_proj` does not
  allocate an up-projection adapter. Experimental native checkpoints made
  before this contract used a different optimizer parameter count for partial
  targets; reload their HF adapter weights with a fresh optimizer rather than
  resuming the old optimizer state.
* **Single adapter per run**: only one set of `--lora-*` arguments is
  honored per training job. Training multiple LoRA adapters in parallel
  within a single `train.py` run is not implemented today — run separate
  jobs if you need multiple adapters.
* **Activation recomputation**: supported, but note *why* it needs handling.
  Under `--recompute-granularity full` mcore runs each block's forward inside
  `torch.no_grad()`; with the base frozen, a checkpointed block has no
  grad-requiring input, so autograd never enters it and every adapter gradient
  is zero. Both paths defeat this by forcing the first activation to require
  grad (`_require_grad_on_first_activation` natively, `peft/recompute.py` in
  Megatron-Bridge). The failure is silent — the adapter still syncs and the
  sha256 manifest still matches — so `grad_norm == 0` or a flat
  `max|lora_B|=0.000e+00` in the export log is the signal to look for.

## Internals

The bridge between Megatron's LoRA path and SGLang adapter loading is in:

- `miles/backends/megatron_utils/lora_utils.py` — argument parsing helpers,
  LoRA detection (`is_lora_enabled`, `is_lora_model`), the provider resolver
  (`resolve_lora_provider`), and HF ↔ Megatron module-name conversion for both
  the `lora` and `canonical_lora` variants.
- `miles_plugins/lora/` — the raw-mode path, and the default
  `--lora-provider-path` provider. `lora.py` is its single lifecycle entry
  point; `config.py` and `registry.py` resolve run configuration and
  `model_type` → `LoRAArchSpec`; `spec/` describes attention, MLP, and future
  MoE layouts; `modules/` owns callable adapter modules; `codec/` owns HF
  naming/load/export and native checkpoint helpers; and `distributed.py` owns
  the remaining TP/SP communication and replicated-gradient reduction.
- `miles/backends/megatron_utils/bridge_lora_helpers.py` — the Megatron-Bridge
  PEFT hook that wraps the model with LoRA layers before training.
- `miles/backends/megatron_utils/checkpoint.py` — adapter-aware save and load.
- `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py`
  — colocate-mode weight sync from the trainer's LoRA tensors into the SGLang
  rollout engine. Disaggregate-mode weight sync is not supported yet.

A worked tutorial covering checkpoint conversion, SGLang adapter loading, and
LoRA-specific evaluation will land here in a future doc pass.
