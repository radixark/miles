---
name: dump-compare
description: Run precision debugging with sglang dumper. Use when the user wants to compare tensor precision between two code versions (e.g. baseline vs a feature branch) using the dumper integration.
user-invocable: true
argument-hint: "[baseline|target|compare|help]"
allowed-tools: Bash, Read, Grep, Glob, Edit
---

# Dumper Precision Comparison Skill

You are helping the user run precision debugging using the sglang dumper integration (`--dumper-enable`).
The dumper captures intermediate tensors at three training phases (inference, fwd_only, fwd_bwd) and saves them as `.pt` files for comparison.

## Workflow Overview

The typical workflow is:
1. **dump baseline** — run training on the reference branch, dump tensors
2. **dump target** — run training on the feature branch, dump tensors
3. **compare** — diff the two dumps phase-by-phase using `sglang.srt.debug_utils.dump_comparator`

## Key Files

- `scripts/dump-small-next.sh` — launch script with dumper enabled (small 8L model)
- `scripts/compare-dumps.sh` — compare two tagged dump directories
- `miles/utils/dumper_utils.py` — miles-side dumper integration (DumperPhase, DumperMegatronUtil)
- `sglang.srt.debug_utils.dump_comparator` — tensor comparison tool (rel_diff, max_abs_diff, mean_abs_diff)
- `sglang.srt.debug_utils.dump_loader` — reads dump metadata into polars DataFrame

## Default Config

- Dump base dir: `DUMPER_BASE_DIR` env var, defaults to `/tmp/dumper`
- Each run is tagged: `/tmp/dumper/<tag>/` with subdirs `inference/`, `fwd_only/`, `fwd_bwd/`
- The dumper auto-disables fault tolerance, health checks, eval, and save to minimize interference

## When user says `$ARGUMENTS`

### If "help" or empty:
Explain the workflow and available commands.

### If "baseline" or "target":
1. Confirm which script to use (small-next or full model) and which branch they're on
2. Run `bash scripts/dump-small-next.sh <tag>`
3. After completion, report how many tensors were dumped per phase

### If "compare" or "compare <baseline_tag> <target_tag>":
1. Default tags are "baseline" and "target" if not specified
2. Run `bash scripts/compare-dumps.sh <baseline_tag> <target_tag>`
3. Analyze the output:
   - Tensors with rel_diff > 1e-3 are flagged with ❌
   - Report which phase first diverges and which layer/tensor
   - Suggest where to look for the root cause

### If user provides a dump directory path:
Inspect it using `sglang.srt.debug_utils.dump_loader.read_meta()` and report contents.

### If user asks to compare specific tensors or phases:
Use `--phase` and `--filter` options:
```bash
bash scripts/compare-dumps.sh baseline target --phase fwd_bwd --filter "hidden_states"
```

## Analysis Guidelines

When analyzing comparison results:
1. **Check phase order**: inference → fwd_only → fwd_bwd. If inference already diverges, the issue is in SGLang weight loading or inference code.
2. **Check layer progression**: if divergence starts at a specific layer, focus on that layer's implementation.
3. **rel_diff magnitude**:
   - < 1e-6: numerically identical (fp rounding)
   - 1e-6 ~ 1e-3: minor numerical differences (likely harmless, e.g. recompute order)
   - > 1e-3: significant divergence, likely a bug
4. **If only fwd_bwd diverges**: the issue is in gradient computation or optimizer step
5. Check if divergence correlates with MoE layers vs dense layers, attention vs MLP, etc.
