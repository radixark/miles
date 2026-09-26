---
paths:
  - "miles/**/*.py"
  - "scripts/**/*.py"
  - "tools/**/*.py"
  - "train.py"
  - "train_async.py"
---

# Don't use `getattr` / `hasattr` for defensive access

Over-defensive `getattr(obj, "field", default)` / `hasattr(obj, "field")` hide
errors and defeat static checking. If a field is always present, accessing it
defensively is confusing and masks real bugs: a renamed flag turns into a
silently applied default instead of an `AttributeError`. Prefer:

1. **`isinstance` for type narrowing** -- check the type, then access fields
   directly.

2. **Always set the field (to `None` if needed), then do a `None` check** -- the
   field should always exist, so a `None` / non-`None` check is enough:

   ```python
   obj.field = None   # in __init__ / construction
   ...
   if obj.field is not None:
       ...
   ```

Bad -- `--true-on-policy-mode` is registered unconditionally in
`add_miles_arguments`, so the parsed namespace always has the attribute and the
default can never apply; the `getattr` only hides a future rename:

```python
if getattr(args, "true_on_policy_mode", False):   # BAD
if args.true_on_policy_mode:                       # GOOD
```
(see `miles/backends/megatron_utils/arguments.py`)

## Where a default is the honest answer

`args` is an argparse namespace assembled from the Miles, Megatron, and SGLang
parsers, and a config object may come from a third party. Absence is a real
state in exactly these cases, and each site carries a one-line reason above it:

- **A flag the parser may not have registered** -- a plugin flag, a Megatron
  flag that differs across the pinned versions, an `sglang_*` flag the pinned
  SGLang does not define:

  ```python
  # compatible for megatron
  if hasattr(args, "rope_type") and args.rope_type is None:
  ```
  (see `miles/backends/megatron_utils/arguments.py`)

- **An optional key of a HF config**, where the upstream schema itself makes
  the field optional (`getattr(hf_config, "quantization_config", None)`).
- **Iterating a spec's fields against a namespace built for another version
  of that spec** (`hasattr(args, f"sglang_{attr.name}")` over
  `msgspec.structs.fields(ServerArgs)` in
  `miles/backends/sglang_utils/sglang_engine.py`).

Anything else is defensive access. New code only: existing sites are
grandfathered -- fold them into direct access when touching the line, not in
drive-by sweeps.
