# Supporting Mixed-Precision Parameters (fp32 in bf16 models)

Some model architectures require specific parameters to remain in fp32 even when the rest of the model runs in bf16/fp16. For example, Qwen3.5's `A_log` parameter must stay fp32 — if it gets rounded to bf16, Megatron-side activations no longer match sglang's fp32 `A_log` on the rollout side, causing precision drift.

miles provides a lightweight, model-agnostic utility in `miles/backends/megatron_utils/fp32_param_utils.py` to handle this. The utility requires **no changes** to Megatron or mbridge base code.

## Quick start — 2 steps to support a new fp32 parameter

### Step 1: Mark the parameter in your model definition

At the model definition site (`miles_plugins/models/your_model.py`), call `mark_param_dtype` right after creating the parameter:

```python
from miles.backends.megatron_utils.fp32_param_utils import mark_param_dtype

# In your model's __init__:
self.X = nn.Parameter(some_init_tensor.to(torch.float32))
mark_param_dtype(self.X, torch.float32)
```

This tags the parameter so that `enforce_marked_param_dtypes` (already wired into the training and conversion entry points) will restore it to fp32 after Megatron's `Float16Module` casts the entire model to bf16.

### Step 2: Override the Bridge to preserve fp32 during weight loading

In your Bridge subclass (`miles_plugins/mbridge/your_model.py`), add a name-matched early-return in `_weight_to_mcore_format` so the HF checkpoint value is not pre-cast to bf16:

```python
def _weight_to_mcore_format(
    self, mcore_weights_name: str, hf_weights: list[torch.Tensor]
) -> tuple[list[str], list[torch.Tensor]]:
    if mcore_weights_name.endswith("your_module.X"):
        assert len(hf_weights) == 1
        return hf_weights[0].to(dtype=torch.float32).contiguous()

    return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)
```

That's it. No changes to `enforce_marked_param_dtypes`, `model.py`, `convert_hf_to_torch_dist.py`, Megatron, or mbridge base are needed.

## Complete example: Qwen3.5 `A_log`

### Model definition (`miles_plugins/models/qwen3_5.py`)

```python
from miles.backends.megatron_utils.fp32_param_utils import mark_param_dtype

class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config, layer_idx: int):
        ...
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A).to(torch.float32))
        mark_param_dtype(self.A_log, torch.float32)
```

### Bridge override (`miles_plugins/mbridge/qwen3_5.py`)

```python
class Qwen3_5Bridge(Qwen2MoEBridge):
    def _weight_to_mcore_format(
        self, mcore_weights_name: str, hf_weights: list[torch.Tensor]
    ) -> tuple[list[str], list[torch.Tensor]]:
        if mcore_weights_name.endswith("self_attention.linear_attn.A_log"):
            assert len(hf_weights) == 1
            # Keep A_log in fp32 before TP scatter; this avoids precision loss
            # from Bridge's global pre-cast to self.dtype.
            return hf_weights[0].to(dtype=torch.float32).contiguous()

        # ... other weight conversions ...
        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)
```

### Integration points (already wired — no action needed)

`enforce_marked_param_dtypes(model)` is called right after `get_model` in both:

- `miles/backends/megatron_utils/model.py` — the training entry point
- `tools/convert_hf_to_torch_dist.py` — the HF → Megatron checkpoint conversion tool

## Code path: from parameter definition to loaded weight

The following traces the real execution path of a parameter from model definition through to the final loaded weight. Understanding this path makes it clear where each cast happens and why both steps above are needed.

### Phase 1 — Model construction and Float16Module wrap

```
model_provider_func()                        # builds your nn.Module (fp32 params)
  └─ Qwen3_5GatedDeltaNet.__init__()
       └─ self.A_log = nn.Parameter(...)     # fp32 at creation
       └─ mark_param_dtype(self.A_log, fp32) # tags _miles_forced_param_dtype attr
            │
            ▼
get_model()                                  # megatron/training/training.py:1162
  ├─ build_model()                           # calls model_provider_func
  ├─ model_module.cuda()                     # move to GPU, still fp32
  └─ Float16Module(config, model_module)     # :1264
       └─ module.bfloat16()                  # ← CAST 1: every param.data → bf16
            │
            ▼
enforce_marked_param_dtypes(model)           # miles model.py / convert tool
  └─ for each param with _miles_forced_param_dtype:
       param.data = param.data.to(fp32)      # ← UNDO CAST 1: restore tagged params
```

After this phase, tagged parameters are fp32 in the live model. Untagged parameters stay bf16. The `Parameter` object identity is preserved so optimizer and DDP registered afterwards see stable references.

### Phase 2 — HF weight loading via mbridge

```
bridge.load_weights(model, hf_path)          # mbridge/core/bridge.py:152
  │
  │  for each (local_name, hf_names):
  │    ┌─ load HF tensors from safetensors  (original dtype, e.g. fp32)
  │    │
  │    ├─ _weight_to_mcore_format(name, hf_weights)    # bridge.py:816
  │    │    │
  │    │    ├─ Bridge base: w.to(self.dtype)            # ← CAST 2: hf tensor → bf16
  │    │    │   (self.dtype is typically bf16)
  │    │    │
  │    │    └─ Subclass override (Step 2):
  │    │         if name matches "...A_log":
  │    │           return hf_weights[0].to(fp32)        # ← BYPASS CAST 2
  │    │
  │    ├─ _weight_split_across_tp(name, mcore_weight, param, tp_size)
  │    │
  │    ├─ t.to(param.device, dtype=param.dtype)         # ← CAST 3: align to param dtype
  │    │   param.dtype is fp32 (thanks to enforce above), so this is a no-op for
  │    │   tagged params; for untagged params it casts to bf16 as expected
  │    │
  │    └─ scatter across TP ranks → param.copy_(result)
```

### End-to-end dtype flow for a tagged fp32 parameter

| Stage | What happens | Resulting dtype |
|---|---|---|
| `nn.Parameter(...)` | Created in model definition | fp32 |
| `mark_param_dtype(...)` | Tags the param, no dtype change | fp32 |
| `module.bfloat16()` (cast 1) | `Float16Module` wraps the model | **bf16** |
| `enforce_marked_param_dtypes` | Restores tagged params | fp32 |
| `_weight_to_mcore_format` (cast 2) | Bridge subclass early-returns fp32 | fp32 |
| `t.to(dtype=param.dtype)` (cast 3) | `param.dtype` is fp32, no-op | fp32 |
| `param.copy_(...)` | Final loaded value | fp32 |

Without either step, one of the intermediate stages silently rounds the value to bf16 — see the table below.

## Background: why both steps are required

Megatron's bf16/fp16 training stack introduces **three implicit cast points** that can silently round fp32 parameters:

| # | Location | What it casts |
|---|---|---|
| 1 | `Float16Module` ctor — `module.bfloat16()` | Every `nn.Parameter.data` → bf16 at wrap time |
| 2 | `Bridge._weight_to_mcore_format` — `w.to(self.dtype)` | HF tensor → Bridge's `self.dtype` (bf16) |
| 3 | `Bridge.load_weights` — `t.to(param.device, dtype=param.dtype)` | mcore tensor → Megatron `param.dtype` (bf16, due to cast 1) |

Cast 1 has no declarative opt-out — even Megatron's own `_maintain_float32_expert_bias` uses a post-hoc `.data.to(float32)` workaround. `enforce_marked_param_dtypes` generalizes this pattern.

**Step 1** (`mark_param_dtype` + `enforce_marked_param_dtypes`) closes cast 1 and 3: once `param.dtype == fp32`, the `load_weights` in-place cast is a no-op for tagged params.

**Step 2** (Bridge override) closes cast 2: the HF tensor is kept fp32 before being scattered across TP ranks.

Both steps are necessary. Doing only one leaves a silent precision trap:

| Config | mcore_weight | After load_weights cast | Final dtype | Value-accurate |
|---|---|---|---|---|
| Nothing | bf16 | bf16 | bf16 | no |
| Step 1 only (no bridge override) | **bf16** | fp32 up-cast | fp32 | **no** — bits already rounded at cast 2 |
| Step 2 only (no mark/enforce) | fp32 | **bf16** | **bf16** | no |
| Both steps | fp32 | fp32 | fp32 | yes |

The "Step 1 only" row is the subtle trap: the final dtype *looks* correct (fp32), but the values were already rounded to bf16 precision at cast 2 and then up-cast back into an fp32 container.

## Tests

`tests/fast/backends/megatron_utils/test_fp32_param_utils.py` — 13 CPU-only tests covering the downstream helper, the upstream bridge override, a bit-exact end-to-end round-trip, and a negative regression guard against the Step-1-only failure mode.
