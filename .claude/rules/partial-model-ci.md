---
paths:
  - "tests/e2e/**/*.py"
---

# Partial Model CI

A CI test that runs an N-layer slice of a released model instead of the full
model is a partial model test, whatever directory holds it and whichever
backend trains it. The checkpoint name usually carries the layer count
(`GLM-5.2_5layer`, `Qwen3-30B-A3B-5layer`). Most live in
`tests/e2e/megatron/model_scripts/`; others sit in `tests/e2e/ft/`,
`tests/e2e/precision/`, and `tests/e2e/megatron/`. A slice is a sanity check,
not a convergence or throughput run, so a larger batch only adds runtime on
the wide GPU stages.

- Default to 16 samples per rollout and a global batch of 16, e.g.
  `--rollout-batch-size 4 --n-samples-per-prompt 4 --global-batch-size 16`.
- A rollout may grow to 32 samples
  (`rollout-batch-size × n-samples-per-prompt`), never more.
- When the test drives a launcher, set these values through its `ScriptArgs`
  fields in the test file. A launcher that hardcodes them gains fields for
  them, as `launch-and-model-scripts.md` requires, and its defaults stay the
  full-model recipe.
