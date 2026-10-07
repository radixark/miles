---
paths:
  - "tests/e2e/megatron/model_scripts/**/*.py"
---

# Partial Model CI

Tests under `tests/e2e/megatron/model_scripts/` run an N-layer slice of a large
model. They are sanity checks that the slice converts, loads, rolls out, and
trains. They measure neither convergence nor throughput, so a larger batch only
adds runtime on the wide GPU stages.

- Default to 16 samples per rollout and a global batch of 16, e.g.
  `--rollout-batch-size 4 --n-samples-per-prompt 4 --global-batch-size 16`.
- A rollout may grow to 32 samples
  (`rollout-batch-size × n-samples-per-prompt`), never more.
- Set these values in the test file through the launcher's `ScriptArgs`
  fields. A launcher that hardcodes them gains fields for them, as
  `launch-and-model-scripts.md` requires, and its defaults stay the full-model
  recipe.
