---
paths:
  - "tests/e2e/megatron/model_scripts/**/*.py"
---

# Partial Model CI

Tests under `tests/e2e/megatron/model_scripts/` run an N-layer slice of a large
model. They are sanity checks that the slice converts, loads, rolls out, and
trains. They measure neither convergence nor throughput, so a larger batch only
adds runtime on the wide GPU stages.

- Use `--rollout-batch-size 4 --n-samples-per-prompt 4 --global-batch-size 16`:
  16 samples per rollout, trained as one global batch.
- A test that needs more samples per rollout keeps
  `rollout-batch-size × n-samples-per-prompt` at or below 32, and
  `--global-batch-size` at 16.
- Set the three in the test file through the launcher's `ScriptArgs` fields. A
  launcher that hardcodes them gains fields for them, as
  `launch-and-model-scripts.md` requires, and its defaults stay the full-model
  recipe.
