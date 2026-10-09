---
paths:
  - "tests/e2e/**/*.py"
---

# Partial Model CI

A partial model test runs an N-layer slice of a model above 200B total
parameters, because the full model does not fit a CI runner; the checkpoint
name usually carries the layer count (`GLM-5.2_5layer`,
`DeepSeek-V4-Flash-FP8-4layer`). What decides it is the model, not the
directory or the backend: most live in `tests/e2e/megatron/model_scripts/`,
others in `tests/e2e/megatron/` and `tests/e2e/precision/`. A slice of a
smaller model used as a cheap stand-in, such as the `Qwen3-30B-A3B-5layer`
fault-tolerance runs, is not one. A slice is a sanity check, not a
convergence or throughput run, so a larger batch only adds runtime on the
wide GPU stages.

- Default to 16 samples per rollout and a global batch of 16, e.g.
  `--rollout-batch-size 4 --n-samples-per-prompt 4 --global-batch-size 16`.
- A rollout may grow to 32 samples
  (`rollout-batch-size × n-samples-per-prompt`), never more.
- When the test drives a launcher, set these values through its `ScriptArgs`
  fields in the test file. A launcher that hardcodes them gains fields for
  them, as `launch-and-model-scripts.md` requires, and its defaults stay the
  full-model recipe.
