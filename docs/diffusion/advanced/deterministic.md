---
title: Deterministic Training
description: What --deterministic-mode covers, which attention backends it accepts, and what it deliberately does not fix.
---

`--deterministic-mode` configures the **training actor's** forward and backward for repeatable execution with the same
hardware, topology, software stack, and inputs. However, some argument gates are still incomplete, so configurations
that do not support deterministic execution may still pass validation.

## What it turns on

### At actor spawn (`miles/ray/actor_group.py`)

```bash
NCCL_DETERMINISTIC=1
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

NCCL reads its variable at `init_process_group` and cuBLAS at the first matmul — both before the
actor body executes, so setting them from Python would be too late. Both are `setdefault`, so an
explicit value in `--train-env-vars` wins.

`:4096:8` rather than `:16:8`: both are deterministic, but the larger workspace keeps cuBLASLt
from being workspace-limited. Costs roughly 32 MiB per handle.

### In the actor (`miles/backends/fsdp_utils/actor.py`)

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)
```

`warn_only=False` is load-bearing: SDPA's deterministic backward is gated on `!warnOnly` inside
`aten`'s `attention_backward.cu`, so `warn_only=True` would silently be a no-op on the native
backend — the most common one.

## Attention is the hard part

`torch.use_deterministic_algorithms` only governs torch-native ops. A custom attention kernel is
opaque to it and would run nondeterministically **without any warning**, so backends are
classified explicitly:


| `--fsdp-attention-backend`             | How determinism is obtained                                                                     |
| -------------------------------------- | ----------------------------------------------------------------------------------------------- |
| unset, `*native*`, `*math*` (SDPA)     | torch's global flag covers it. `math` backends are deterministic by construction.               |
| `*flash*` (flash-attn, FA3)            | torch's flag cannot reach it — miles patches `deterministic=True` onto the kernel entry points. |
| `sage`, `xformers`, `flex`, `aiter`, … | No hook exists. **Rejected.**                                                                   |


The check runs **driver-side, before any actor launches** (`validate_attention_args`), so a
misconfiguration fails in seconds instead of after a multi-node startup.

A flash backend that is installed but exposes no `deterministic` parameter is also rejected, with a message naming which kernels were found.

### How the flash patch works

For diffusers-backed families, miles wraps the dispatch functions diffusers routes flash through:

```
flash_attn_func         flash_attn_varlen_func
flash_attn_3_func       flash_attn_3_varlen_func
```

Each entry point with a `deterministic` parameter is replaced by
`functools.partial(fn, deterministic=True)`; the rest are skipped.

## What it does not cover

- **Rollout determinism.** Guaranteed by sglang-d and its post-training support.
- **Train/rollout agreement.** Determinism removes run-to-run variance; it does not make the two
forwards equal. That is a precision problem — see [Dtype Control](/diffusion/advanced/dtype-control).

## Cost

Real but usually acceptable: `cudnn.benchmark=False` gives up kernel autotuning, deterministic
kernels are chosen over faster nondeterministic ones, and cuBLAS reserves ~32 MiB per handle.
Recipes that need reproducibility more than throughput keep it on permanently.

## In CI

The e2e suite is built on this flag. Each test under `tests/e2e/short/` runs a real recipe with
`--deterministic-mode` for its registered short run (currently two or four rollouts) and compares every registered
metric series — reward statistics, old/new log-probs, grad norm — **bit for bit** against a standard committed under
`tests/ci/fixtures/e2e_standards/` (strict unless the test registers a per-metric tolerance).
Determinism is what makes strict comparison viable: a tolerance wide enough to absorb run-to-run
noise would also absorb small regressions. Standards are re-recorded by the PR author
(`python tests/ci/e2e_metrics_registry.py record --test <test-file>`), never by CI.

Because reward metrics are compared too, the CI recipes also pin the engine side — e.g.
`--sglang-dit-precision fp16`, plus the batch-invariant-op environment above — matching the
"what it does not cover" list: bit-stable curves need both halves.

## Related

- [Dtype Control](/diffusion/advanced/dtype-control) — the other half of train/rollout numeric agreement.

