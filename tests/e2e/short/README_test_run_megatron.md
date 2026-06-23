# `test_run_megatron.py` — Run Guide & Troubleshooting

End-to-end test that compares Megatron parallel configurations by dumping intermediate
activations from a baseline run (tp=1) and a target run (tp=2, pp=2, cp=2), then
validates them with the SGLang debug comparator.

**Test file:** `tests/e2e/short/test_run_megatron.py`

This is a **Typer CLI**, not a pytest test. Do not run it with `pytest`.

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| GPUs | **8× H100** (CI label: `stage-c-8-gpu-h100`) |
| Environment | Miles Docker container or equivalent (`/root/models`, Megatron-LM, SGLang) |
| Working directory | Miles repo root (`/workspace/miles` or `/home/kgoginen/miles`) |
| Python deps | `polars` (for comparator), `typer`, full miles stack |
| Megatron-LM | Recommended: `radixark/Megatron-LM` branch `miles-main` |

---

## How to Run

### Full pipeline (prepare + baseline + target + compare)

```bash
cd /workspace/miles   # or your miles checkout
python tests/e2e/short/test_run_megatron.py run --mode tp1_vs_tp2pp2cp2
```

On startup the script prints a temp run directory, e.g.:

```
Run directory: /tmp/test_run_megatron_abc123
```

Artifacts live under `{Run directory}/dumps/`.

### Re-run comparator only (no training)

Use this after a successful `run` when you only want to re-check dumps:

```bash
python tests/e2e/short/test_run_megatron.py compare \
  --mode tp1_vs_tp2pp2cp2 \
  --dump-dir /tmp/test_run_megatron_abc123/dumps
```

### Help

```bash
python tests/e2e/short/test_run_megatron.py --help
python tests/e2e/short/test_run_megatron.py run --help
python tests/e2e/short/test_run_megatron.py compare --help
```

### Available modes

| Mode | Baseline | Target | Status |
|------|----------|--------|--------|
| `tp1_vs_tp2pp2cp2` | `--tp 1` | `--tp 2 --pp 2 --cp 2 --ep 1` | **Active** |
| `tp1_vs_tp2pp2cp2_thd` | same | same (THD format) | Commented out — not supported yet |

### What `run` does internally

1. Downloads `fzyzcjy/Qwen3-30B-A3B-5layer` to `/root/models/`
2. Converts checkpoint to `/root/Qwen3-30B-A3B-5layer_torch_dist`
3. Writes a Megatron source-patcher YAML (activation dump hooks)
4. Invokes:

   ```bash
   python -m miles.utils.debug_utils.run_megatron run-and-compare \
     --model-type qwen3-30B-A3B-5layer \
     --baseline '--tp 1' \
     --target '--tp 2 --pp 2 --cp 2 --ep 1' \
     ...
   ```

5. Runs the SGLang comparator on baseline vs target dumps

---

## Errors Encountered & Fixes

The following issues were hit while running this test. Each section lists the symptom,
root cause, solution, and files changed.

---

### Error 1: `Missing command`

#### Symptom

```
Usage: test_run_megatron.py [OPTIONS] COMMAND [ARGS]...
╭─ Error ─────────────────────────────────────────────────────╮
│ Missing command.                                              │
╰───────────────────────────────────────────────────────────────╯
```

#### Cause

The script was invoked without a Typer subcommand:

```bash
python tests/e2e/short/test_run_megatron.py   # wrong
```

#### Solution

Always pass a subcommand:

```bash
python tests/e2e/short/test_run_megatron.py run --mode tp1_vs_tp2pp2cp2
```

#### Files changed

None — usage issue only.

---

### Error 2: `PatchApplicationError` — match text not found in source

#### Symptom

```
sglang.srt.debug_utils.source_patcher.types.PatchApplicationError: match text not found in source:
residual = getattr(self, "_sglang_pre_mlp_residual", hidden_states)
```

Occurs during source patching of `TransformerLayer._forward_mlp`, before forward pass.

#### Cause

The Megatron source-patcher YAML must match exact source lines in
`megatron.core.transformer.transformer_layer.TransformerLayer._forward_mlp`.

Commit `bbea855` (#1175) changed the patch anchor from:

```python
residual = hidden_states
```

to:

```python
residual = getattr(self, "_sglang_pre_mlp_residual", hidden_states)
```

That newer line exists on `radixark/Megatron-LM` `miles-main`, but older Megatron
installs still have `residual = hidden_states`. The patcher does exact text matching,
so it fails when versions disagree.

#### Solution

**Code fix (applied):** Runtime detection of the installed Megatron version picks the
correct anchor line when writing the patcher YAML.

**Alternative:** Update Megatron-LM to `miles-main`:

```bash
cd /root/Megatron-LM
git fetch origin miles-main && git checkout miles-main && git pull
pip install -e .
```

#### Files changed

| File | Change |
|------|--------|
| `tests/e2e/conftest_dumper.py` | Added `_PRE_MLP_RESIDUAL_MATCH_PLACEHOLDER`, `megatron_pre_mlp_residual_match()`, `get_megatron_patcher_yaml()`. YAML templates use placeholder `__PRE_MLP_RESIDUAL_MATCH__` resolved at runtime via `inspect.getsource()`. |
| `tests/e2e/short/test_run_megatron.py` | Import `get_megatron_patcher_yaml` instead of `MEGATRON_PATCHER_YAMLS`; call it in `_prepare()`. |
| `tests/e2e/short/test_dumper.py` | Same import/switch for shared dumper test. |

**Key addition in `conftest_dumper.py`:**

```python
def megatron_pre_mlp_residual_match() -> str:
    import inspect
    from megatron.core.transformer.transformer_layer import TransformerLayer
    source = inspect.getsource(TransformerLayer._forward_mlp)
    if "_sglang_pre_mlp_residual" in source:
        return 'residual = getattr(self, "_sglang_pre_mlp_residual", hidden_states)'
    if "residual = hidden_states" in source:
        return "residual = hidden_states"
    raise RuntimeError(...)

def get_megatron_patcher_yaml(format_key: str) -> str:
    template = MEGATRON_PATCHER_YAMLS[format_key]
    return template.replace("__PRE_MLP_RESIDUAL_MATCH__", megatron_pre_mlp_residual_match())
```

---

### Error 3: `AttributeError: 'types.SimpleNamespace' object has no attribute 'cp'`

#### Symptom

```
File ".../run_megatron/worker/batch.py", line 43, in prepare_batch
  token_tensor = slice_with_cp(token_tensor, **cp_kwargs)
File ".../training_utils/cp_utils.py", line 255, in slice_with_cp
  cp_rank = parallel_state.cp.rank
AttributeError: 'types.SimpleNamespace' object has no attribute 'cp'
```

Fails on ranks where CP > 1 (e.g. ranks 4–7 with `--cp 2`).

#### Cause

PR #1185 (`d2e0b35`) updated `slice_with_cp()` to use the standard `ParallelState` layout:

```python
cp_rank = parallel_state.cp.rank
cp_size = parallel_state.cp.size
```

The standalone Megatron worker in `batch.py` still passed a flat stub:

```python
# Before (broken)
parallel_state=SimpleNamespace(cp_rank=cp_rank, cp_size=cp_size)
```

#### Solution

Match the nested structure expected by `cp_utils.py`:

```python
# After (fixed)
parallel_state=SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=cp_size))
```

#### Files changed

| File | Change |
|------|--------|
| `miles/utils/debug_utils/run_megatron/worker/batch.py` | Fixed `parallel_state` shape in `prepare_batch()` CP kwargs. |
| `tests/fast/utils/debug_utils/run_megatron/worker/test_batch.py` | Updated `_zigzag_slice()` helper to use the same nested `SimpleNamespace`. |

**Diff in `batch.py` (lines 37–42):**

```python
cp_kwargs: dict[str, object] = dict(
    pad_value=0,
    parallel_state=SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=cp_size)),
    qkv_format="bshd",
    max_seq_len=seq_length,
)
```

---

### Error 4: `ModuleNotFoundError: No module named 'polars'`

#### Symptom

```
File ".../sglang/srt/debug_utils/comparator/entrypoint.py", line 9, in <module>
  import polars as pl
ModuleNotFoundError: No module named 'polars'
```

Occurs at the **compare** step after baseline/target runs complete.

#### Cause

The SGLang debug comparator (`python -m sglang.srt.debug_utils.comparator`) requires
`polars`. Miles installs SGLang with `pip install -e "python[all]" --no-deps` in the
Dockerfile, and `polars` is only listed under SGLang's **test** optional dependencies —
not core deps — so it is not installed automatically.

#### Solution

**Immediate fix (existing containers):**

```bash
pip install polars
```

Then re-run compare only if dumps already exist:

```bash
python tests/e2e/short/test_run_megatron.py compare \
  --mode tp1_vs_tp2pp2cp2 \
  --dump-dir /tmp/test_run_megatron_XXXXXX/dumps
```

**Permanent fix (applied):** Added `polars` to miles `requirements.txt`.

#### Files changed

| File | Change |
|------|--------|
| `requirements.txt` | Added `polars` (line 9). |

---

## Summary of All File Changes

| File | Action | Purpose |
|------|--------|---------|
| `tests/e2e/conftest_dumper.py` | Modified | Megatron-version-aware source patcher YAML generation |
| `tests/e2e/short/test_run_megatron.py` | Modified | Use `get_megatron_patcher_yaml()` in `_prepare()` |
| `tests/e2e/short/test_dumper.py` | Modified | Same patcher helper (shared dumper test) |
| `miles/utils/debug_utils/run_megatron/worker/batch.py` | Modified | Fix CP `parallel_state` stub for `slice_with_cp()` |
| `tests/fast/utils/debug_utils/run_megatron/worker/test_batch.py` | Modified | Align unit test helper with batch fix |
| `requirements.txt` | Modified | Add `polars` for SGLang comparator |
| `tests/e2e/short/README_test_run_megatron.md` | **Added** | This document |

No changes were required to `test_run_megatron.py` CLI structure itself — only to
supporting modules and dependencies.

---

## Quick Checklist Before Running

- [ ] 8 GPUs available
- [ ] Running from miles repo root
- [ ] Megatron-LM installed (`pip install -e .` in Megatron-LM checkout)
- [ ] `polars` installed (`pip install polars` or `pip install -r requirements.txt`)
- [ ] Using subcommand: `run --mode tp1_vs_tp2pp2cp2`
- [ ] Local code includes the fixes above (especially `batch.py` and `conftest_dumper.py`)

---

## Related Tests

| File | Purpose |
|------|---------|
| `tests/e2e/short/test_dumper.py` | SGLang vs Megatron dumper comparison (different CLI, same patcher infra) |
| `tests/fast/utils/debug_utils/run_megatron/worker/test_batch.py` | Unit tests for batch CP slicing |

---

## References

- Megatron patch anchor change: commit `bbea855` (#1175)
- CP parallel state fix: commit `d2e0b35` (#1185)
- Megatron-LM repo: `https://github.com/radixark/Megatron-LM` branch `miles-main`
- Underlying CLI: `python -m miles.utils.debug_utils.run_megatron run-and-compare`
