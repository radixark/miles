---
name: debug-precision
description: Automated precision debugging. Runs dumper on baseline and target, compares tensor dumps phase-by-phase, traces divergences back to source code, and identifies root causes. Use when the user wants to find precision bugs between two code versions.
user-invocable: true
argument-hint: "[run|compare|investigate|full]"
allowed-tools: Bash, Read, Grep, Glob, Edit, Task, TodoWrite
---

# Automated Precision Debugging Skill

You are an expert precision debugging agent for the Miles RLHF training framework.
Your job is to find **where and why** tensor values diverge between a baseline and a target code version.

## Architecture Context

Miles has three training phases, executed in order each rollout:

1. **inference** — SGLang engines run model inference to generate rollout responses
2. **fwd_only** — Megatron runs forward-only passes to compute log-probs / values
3. **fwd_bwd** — Megatron runs forward-backward passes to compute gradients and update weights

The dumper (`--dumper-enable`) hooks into all three phases and saves intermediate tensors as `.pt` files.

### Key source files

| File | Role |
|------|------|
| `miles/backends/megatron_utils/model.py` | `forward_only()` and `train_one_step()` — where DumperMegatronUtil wraps forward steps |
| `miles/utils/dumper_utils.py` | DumperMegatronUtil, DumperPhase, SGLang dumper configuration |
| `miles/utils/arguments.py` | `--dumper-*` CLI args and `_maybe_apply_dumper_overrides()` |
| `miles_plugins/mbridge/qwen3_next.py` | Qwen3 Next model bridge (weight conversion, custom layers) |
| `miles_plugins/models/qwen3_next.py` | Model definition (if exists) |
| `sglang.srt.debug_utils.dumper` | Core `_Dumper` class, `dumper.dump()`, file I/O |
| `sglang.srt.debug_utils.dump_comparator` | `check_tensor_pair()`, rel_diff / max_abs_diff |
| `sglang.srt.debug_utils.dump_loader` | `read_meta()` → polars DataFrame of dump metadata |

### Dump file naming convention

```
forward_pass_id=<id>___rank=<rank>___name=<tensor_name>___dump_index=<idx>___[layer_id=<L>___...].pt
```

Each `.pt` file contains a single `torch.Tensor` saved with `torch.save()`.

### Expected dump fields per phase

- **inference (engine_*)**: `input_ids`, `positions`
- **fwd_only**: `input_ids`, `cu_seqlens_q`, `cu_seqlens_kv`, `qkv_format`
- **fwd_bwd**: `input_ids`, `cu_seqlens_q`, `cu_seqlens_kv`, `qkv_format`

When `register_non_intrusive_dumper` hooks the model, additional per-layer tensors (hidden_states, attention outputs, MLP outputs, etc.) are also dumped.

### Dump directory structure

```
<dump_dir>/<tag>/
├── inference/engine_0/sglang_dump_<ts>/*.pt
├── fwd_only/sglang_dump_<ts>/*.pt
└── fwd_bwd/sglang_dump_<ts>/*.pt
```

## Commands

When the user invokes `/debug-precision`, parse `$ARGUMENTS`:

### `full` (default if empty)

Run the complete automated pipeline:

1. **Prepare** — check git branch, confirm settings with user
2. **Dump baseline** — `bash scripts/dump-small-next.sh baseline`
3. **Dump target** — remind user to switch branch (or use current if already on target), run `bash scripts/dump-small-next.sh target`
4. **Compare** — `bash scripts/compare-dumps.sh baseline target`
5. **Investigate** — automatically analyze divergences (see investigation protocol below)
6. **Report** — summarize findings with source code references

### `run <tag>`

Run a single dump with the given tag. Execute:
```bash
bash scripts/dump-small-next.sh <tag>
```
Report tensor counts per phase when done.

### `compare [baseline_tag] [target_tag]`

Compare two existing dumps (defaults: `baseline` and `target`). Execute:
```bash
bash scripts/compare-dumps.sh <baseline_tag> <target_tag> 2>&1
```
Then proceed to investigation.

### `investigate [dump_dir]`

Skip dumping, go straight to analyzing an existing comparison or dump directory.
If a dump_dir is given, load metadata and inspect.
Otherwise, re-run comparison and analyze.

## Investigation Protocol

When you find divergences in the comparison output, follow this systematic procedure:

### Step 1: Triage — identify the first divergent phase

Check phases in causal order: `inference` → `fwd_only` → `fwd_bwd`.
The **first** phase with divergence is where the bug originates.

- If **inference** diverges: bug is in SGLang model loading, weight conversion (`mbridge/`), or inference forward pass
- If **fwd_only** diverges but inference is clean: bug is in Megatron weight sync (`update_weight/`), packed sequence handling, or Megatron forward pass
- If only **fwd_bwd** diverges: bug is in loss computation, gradient calculation, or optimizer step

### Step 2: Identify the divergent tensor

From the comparator output, extract:
- `name` — which tensor (e.g. `input_ids`, `hidden_states`, `layer_start__hidden_states`)
- `layer_id` — which layer (if present in filename)
- `forward_pass_id` — which micro-batch step
- `rank` — which GPU rank
- `rel_diff` and `max_abs_diff` — severity

Sort divergent tensors by `dump_index` to find the **earliest** point of divergence.

### Step 3: Trace to source code

Based on the divergent tensor name and phase:

**For inference phase tensors:**
1. Read `miles_plugins/mbridge/qwen3_next.py` — weight conversion logic
2. Search sglang model code for where the tensor is produced
3. Check `miles/backends/megatron_utils/update_weight/` — weight sync from Megatron to SGLang

**For fwd_only / fwd_bwd phase tensors:**
1. Read `miles/backends/megatron_utils/model.py` — `forward_only()` or `train_one_step()`
2. If layer-specific: find the layer implementation in Megatron-LM
   - Dense layers: `/root/Megatron-LM/megatron/core/transformer/`
   - MoE layers: `/root/Megatron-LM/megatron/core/transformer/moe/`
   - SSM/hybrid: `/root/Megatron-LM/megatron/core/ssm/`
   - Attention: `/root/Megatron-LM/megatron/core/transformer/attention.py`
3. If input tensors diverge: check data loading and packed sequence params
4. If activations diverge at a specific layer: read that layer's forward() method

**For gradient / weight update tensors:**
1. Check optimizer config in the launch script
2. Read `miles/backends/megatron_utils/model.py` train_one_step gradient handling
3. Check if `--accumulate-allreduce-grads-in-fp32` or precision settings differ

### Step 4: Differential analysis

Compare the source code between baseline and target branches for the files identified in Step 3.
Use `git diff` to see what changed:
```bash
git diff main..HEAD -- <file_path>
```

Look for common precision bug patterns:
- **dtype changes**: fp32 vs bf16 in computation
- **transpose / reshape errors**: wrong dimension ordering
- **missing scaling factors**: attention scale, MoE router normalization
- **stride / layout issues**: contiguous() calls, packed vs padded tensors
- **gate / routing changes**: MoE expert selection, shared expert gating
- **normalization order**: pre-norm vs post-norm, RMSNorm epsilon
- **position encoding**: rotary base, rotary percent, position ID computation
- **weight tying / untying**: embedding and output weight sharing

### Step 5: Deep tensor inspection (if needed)

If the comparison output is insufficient, load specific tensors for manual inspection:

```python
import torch
from sglang.srt.debug_utils.dump_loader import read_meta

# Load metadata
df = read_meta("/tmp/dumper/baseline/fwd_only/sglang_dump_xxx/")
print(df)

# Load specific tensors
t_baseline = torch.load("<baseline_path>", weights_only=False)
t_target = torch.load("<target_path>", weights_only=False)

# Inspect
print(f"shape: {t_baseline.shape} vs {t_target.shape}")
print(f"dtype: {t_baseline.dtype} vs {t_target.dtype}")
print(f"diff:  {(t_baseline.float() - t_target.float()).abs().max()}")
```

### Step 6: Report

Produce a clear report with:

1. **Summary**: one-line description of the bug
2. **Divergence point**: phase, tensor name, layer, first micro-batch
3. **Severity**: rel_diff magnitude and what it means
4. **Root cause**: the specific code change that causes the divergence
5. **Source references**: file paths and line numbers
6. **Fix suggestion**: what code change would resolve it (with diff if possible)

## Severity Guide

| rel_diff | Meaning |
|----------|---------|
| < 1e-7 | Identical (fp rounding only) |
| 1e-7 ~ 1e-5 | Numerically equivalent (recompute / op order) |
| 1e-5 ~ 1e-3 | Minor difference — may be acceptable (e.g. different attention backend) |
| 1e-3 ~ 1e-1 | Significant divergence — likely a bug |
| > 1e-1 | Severe — completely wrong computation |

## Tips

- Always check `input_ids` first — if inputs already differ, everything downstream will too
- MoE models: check if expert routing is deterministic (topk selection ties)
- Packed sequences: check `cu_seqlens` alignment between baseline and target
- Multi-GPU: compare per-rank to isolate if it's a specific TP/EP shard
- If only some micro-batches diverge, it might be data-dependent (e.g. sequence length edge cases)
