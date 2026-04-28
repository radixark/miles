---
title: Backends Beyond Megatron
description: Use HuggingFace implementations as black-box modules inside Megatron's parallel pipeline.
---

# Backends Beyond Megatron

Megatron-LM is fast but rigid. Adding a brand-new architecture (Qwen3Next's
Gated-Delta-Net, for example) to its native code path is invasive and slow. Miles takes
a different route: **wrap the model's official HuggingFace implementation as a black-box
module** and embed it inside Megatron's parallel scheduling.

This document uses Qwen3Next 80B-A3B as the canonical example.

## How it works

Megatron instantiates a model in two steps:

1. Generate a layer specification (`ModuleSpec`).
2. Instantiate concrete PyTorch modules from that spec.

Miles intercepts step 1 and replaces individual modules with HuggingFace ones. Three
components do the work:

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
        x = align_for_seq_parallel(x)   # bridge to Megatron's parallelism contract
        out = self.hf_attn(x, ...)
        return realign(out)
```

The wrapper handles the data-layout conversions Megatron requires (sequence parallelism,
TP, etc.) and forwards into the HF module unchanged.

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

[mbridge](https://github.com/ISEEKYAN/mbridge) handles the bidirectional mapping
between HF parameter names and Megatron's parameter names so the same checkpoint can be
loaded by either.

## Why this matters

| | Patch Megatron | Miles wrapper approach |
|---|---|---|
| Time-to-first-train | Weeks | Hours |
| Code touched | Megatron core (risky) | A plugin file |
| Pipeline parallel | ✅ | ✅ |
| Sequence parallel | ✅ | ✅ |
| MoE acceleration | ✅ | ✅ |
| TP **inside the module** | ✅ | ❌ (see limits) |

For nearly every Attention-only swap, TP within the module isn't critical — Attention
parameters are a small fraction of total params in MoE. For the rare case where you do
need it, you have to fall back to the native Megatron path.

## Mixed precision: keeping fp32 parameters fp32

Some architectures need certain parameters to *stay* fp32 even when the rest is bf16.
Qwen3.5's `A_log` is the canonical example — round it to bf16 and Megatron-side
activations diverge from SGLang-side rollout, causing precision drift.

Megatron has three implicit cast points that silently downcast fp32 to bf16:

* `Float16Module` construction
* `Bridge._weight_to_mcore_format`
* `Bridge.load_weights`

Two steps fix it (both are required):

### Mark the parameter

```python
from miles.backends.megatron_utils.fp32_param_utils import mark_param_dtype

class MyModel(nn.Module):
    def __init__(self, ...):
        super().__init__(...)
        self.A_log = nn.Parameter(torch.log(A).to(torch.float32))
        mark_param_dtype(self.A_log, torch.float32)
```

`enforce_marked_param_dtypes(model)` — already wired into the training and checkpoint
conversion entry points — restores tagged params to fp32 after `Float16Module` casts the
rest of the model to bf16.

### Override the Bridge

```python
class Qwen3_5Bridge(Qwen2MoEBridge):
    def _weight_to_mcore_format(self, mcore_weights_name, hf_weights):
        if mcore_weights_name.endswith("self_attention.linear_attn.A_log"):
            assert len(hf_weights) == 1
            return hf_weights[0].to(dtype=torch.float32).contiguous()
        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)
```

If you do only one of these, the *final* dtype looks correct but the values were already
rounded — a subtle, painful trap.

## Current limitations

* TP **inside** the wrapped module isn't supported. For Attention this is usually fine
  (small param count); for MoE expert weights it's a deal-breaker — use Megatron native
  for those.
* Performance is bounded by the HF implementation's efficiency. If `flash-attn` isn't
  available, you'll feel it.

## When to use this path

* New model architecture (days/weeks old) not yet in Megatron.
* Research models with non-standard layers (Mamba-style state space, Gated-Delta-Net,
  etc.).
* Anything where the cost of patching Megatron exceeds the value of squeezing the last
  few percent of throughput.

## When NOT

* Production-frozen architectures (Qwen3 standard, GLM4) — Megatron's native is
  faster.
* Anything where TP inside the new module is critical.
