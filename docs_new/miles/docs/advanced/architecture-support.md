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

1. Generate a layer specification (`ModuleSpec`).
2. Instantiate concrete PyTorch modules from that spec.

Miles intercepts step 1 and replaces individual modules with HuggingFace
implementations. Three components do the work.

### 1. Replace the Megatron module spec

```python
# miles_plugins/models/qwen3_next.py
def get_qwen3_next_spec(...):
    spec = get_default_decoder_block_spec(...)
    spec.submodules.self_attention = ModuleSpec(module=HuggingfaceAttention)
    spec.submodules.self_attention.params['qk_layernorm'] = True
    return spec
```

### 2. Wrap the HuggingFace module

```python
# miles_plugins/models/hf_attention.py
class HuggingfaceAttention(MegatronModule):
    def __init__(self, config, ...):
        super().__init__(config)
        self.hf_attn = Qwen3NextAttention(config)

    def forward(self, x, *args, **kwargs):
        x = align_for_seq_parallel(x)
        out = self.hf_attn(x, ...)
        return realign(out)
```

The wrapper handles the data-layout conversions Megatron requires (sequence
parallelism, TP) and forwards into the HF module unchanged.

### 3. Align weights with mbridge

```python
# miles_plugins/mbridge/qwen3_next.py
class Qwen3NextBridge(Qwen2MoEBridge):
    @property
    def name_map(self):
        return {
            "model.layers.{i}.self_attn.q_proj": "decoder.layers.{i}.self_attention.q_proj",
            ...
        }
```

[mbridge](https://github.com/ISEEKYAN/mbridge) handles the bidirectional
mapping between HF parameter names and Megatron parameter names so the same
checkpoint can be loaded by either.

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

Megatron has three implicit cast points that downcast fp32 to bf16:

* `Float16Module` construction.
* `Bridge._weight_to_mcore_format`.
* `Bridge.load_weights`.

Two steps are required.

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
