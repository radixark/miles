---
title: Backends Beyond Megatron
description: Embed HuggingFace implementations as black-box modules inside Megatron's parallel pipeline.
---

# Backends Beyond Megatron

Adding a new architecture (such as Qwen3-Next's Gated-Delta-Net) directly to
Megatron-LM's native code path is invasive. Miles takes a different approach:
wrap the model's official HuggingFace implementation as a black-box module and
embed it inside Megatron's parallel scheduling. This trades some throughput
ceiling (no TP inside the wrapped module) for a much shorter time-to-train
when the architecture is new.

This page uses Qwen3-Next 80B-A3B as the running example.

## How it works

Megatron instantiates a model in two steps:

1. Generate a layer specification (`ModuleSpec` / decoder block spec).
2. Instantiate concrete PyTorch modules from that spec.

Miles intercepts step 1 and rewrites specific submodules to point at a
HuggingFace-backed implementation. Three components do the work.

### 1. Custom decoder spec

`miles_plugins/models/qwen3_next.py` defines `get_qwen3_next_spec`. It
starts from `get_gpt_decoder_block_spec`, then for the layers whose HF
`layer_types[i] == "linear_attention"` it overrides the layer's
`submodules.self_attention` with a `ModuleSpec(module=Attention,
params={"args": args})` (referenced from `miles_plugins/models/`):

```python
# miles_plugins/models/qwen3_next.py (simplified excerpt)
transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)
...
for layer_id in range(num_layers_to_build):
    if hf_config.layer_types[layer_id + offset] == "linear_attention":
        layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
        layer_specs.submodules.self_attention = ModuleSpec(
            module=Attention,
            params={"args": args},
        )
        transformer_layer_spec.layer_specs[layer_id] = layer_specs
return transformer_layer_spec
```

The spec function is wired up via `--spec`:

```bash
MODEL_ARGS+=(
   --spec miles_plugins.models.qwen3_next get_qwen3_next_spec
)
```

### 2. Abstract Megatron-side wrapper

`miles_plugins/models/hf_attention.py` defines an abstract
`HuggingfaceAttention(MegatronModule, ABC)` whose `__init__` takes
`(args, config, layer_number, cp_comm_type, pg_collection)`. It loads the
HuggingFace config from `args.hf_checkpoint` and prepares the layout
adapters Megatron's parallelism contract requires (sequence parallel, CP
zigzag/packed-shard conversions). Concrete Attention classes subclass it
and embed the actual HF attention module.

### 3. Align weights with mbridge

The HF parameter layout differs from Megatron's. `miles_plugins/mbridge/`
ships per-architecture bridges that reconcile the two. For Qwen3-Next:

```python
# miles_plugins/mbridge/qwen3_next.py
@register_model("qwen3_next")
class Qwen3NextBridge(Qwen2MoEBridge):
    _ATTENTION_MAPPING = (
        Qwen2MoEBridge._ATTENTION_MAPPING
        | {
            f"self_attention.{w}": ["model.layers.{layer_number}." + w]
            for w in [
                "input_layernorm.weight",
                "linear_attn.A_log",
                "linear_attn.conv1d.weight",
                ...
            ]
        }
        | {
            "self_attention.linear_qkv.weight": [
                "model.layers.{layer_number}.self_attn.q_proj.weight",
                "model.layers.{layer_number}.self_attn.k_proj.weight",
                "model.layers.{layer_number}.self_attn.v_proj.weight",
            ],
        }
    )
```

The class-level `_ATTENTION_MAPPING` (and `_MLP_MAPPING`, etc.) extends the
parent bridge with the layer-name substitutions specific to this
architecture. Bridges that need to reshape weights at conversion time
override `_weight_to_mcore_format`. See
[mbridge](https://github.com/ISEEKYAN/mbridge) for the parent bridges.

## Capabilities and limits

| | Patch Megatron core | Miles wrapper approach |
|---|---|---|
| Pipeline parallel | Supported | Supported |
| Sequence parallel | Supported | Supported |
| MoE acceleration | Supported | Supported |
| TP inside the wrapped module | Supported | Not supported |

For Attention-only swaps, missing TP inside the module is usually acceptable
because Attention parameters are a small fraction of total params in MoE
models. For cases where TP inside the new module is required, the native
Megatron path is the right choice.

## Mixed precision: keeping fp32 parameters fp32

Some architectures need certain parameters to remain fp32 even when the rest
of the model is bf16. Qwen3.5's `A_log` is the canonical example. Rounding it
to bf16 makes Megatron-side activations diverge from SGLang-side rollout,
causing precision drift.

The canonical cast point is Megatron's `Float16Module`, which (per the
docstring on `enforce_marked_param_dtypes` in
`miles/backends/megatron_utils/fp32_param_utils.py`) "unconditionally casts
every floating-point parameter to bf16/fp16 at wrap time". The mbridge
weight-conversion path (`_weight_to_mcore_format` and friends) is the
other place fp32 weights can be silently downcast. Two steps are required
to keep tagged params in fp32.

### Mark the parameter

```python
from miles.backends.megatron_utils.fp32_param_utils import mark_param_dtype

class MyModel(nn.Module):
    def __init__(self, ...):
        super().__init__(...)
        self.A_log = nn.Parameter(torch.log(A).to(torch.float32))
        mark_param_dtype(self.A_log, torch.float32)
```

`enforce_marked_param_dtypes(model)` (already wired into the training and
checkpoint conversion entry points) restores tagged params to fp32 after
`Float16Module` casts the rest of the model to bf16.

### Override the bridge

```python
class Qwen3_5Bridge(Qwen2MoEBridge):
    def _weight_to_mcore_format(self, mcore_weights_name, hf_weights):
        if mcore_weights_name.endswith("self_attention.linear_attn.A_log"):
            assert len(hf_weights) == 1
            return hf_weights[0].to(dtype=torch.float32).contiguous()
        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)
```

If only one of these is done, the final dtype looks correct but the values
have already been rounded.

## When this path fits

* New architectures not yet integrated into Megatron core.
* Research models with non-standard layers (Mamba-style state space,
  Gated-Delta-Net, etc.).
* Cases where the cost of patching Megatron exceeds the value of squeezing
  the last few percent of throughput.

## When native Megatron is preferable

* Stable, frozen architectures (Qwen3 standard, GLM4) where Megatron's native
  path is mature.
* Cases where TP inside the new module is critical.
