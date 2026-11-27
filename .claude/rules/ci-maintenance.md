---
paths:
  - ".github/workflows/**/*.yml"
  - "tests/ci/**/*.py"
  - "docs/developer/ci/**/*.md"
---

# CI Maintenance

## GPU shard runtime budget

When maintaining GPU CI scope or sharding, check regular-cadence `run-ci-image` and nightly selection in each affected stage. Exclude disabled registrations and preserve each cadence's eligibility filter. Apply `auto_partition`, sum the registered `est_time` values in each shard, and target a largest sum of at most two hours for regular-cadence `run-ci-image` and less than two hours for nightly. Do not add environment setup, cleanup, or queue time.

Use `run-ci-image` as the broad PR sizing baseline. It does not admit
`nightly=True` registrations; nightly has a larger scope. Explicit extra
labels and `run-ci-all` can exceed this baseline, and a `long` / `ft-long`
file can exceed the target alone.

Try shard counts in ascending multiples of the matching runner pool capacity and choose the first whose largest shard sum meets the budget. Do not set `max-parallel` on Hopper or ROCm shard matrices; let available runners control execution. Preserve hosted empty-shard planning and selected-test coverage. Adding shards cannot shorten an indivisible test file and adds setup cost for each nonempty shard; do not change test coverage merely to meet the runtime target.

## CPU stage runtime budget

CPU stages gate every PR and every CUDA stage, so they carry tighter budgets than GPU stages:

- **`stage-a-cpu` (unlabeled PR):** target at most 5 minutes of `Run tests` per shard (at most 7–8 minutes total wall time including setup), measured from CI run logs rather than summing implicit 1-second estimates.
- **`stage-b-cpu` (regular `run-ci-image` and nightly):** apply `auto_partition` and sum registered `est_time` values per shard, targeting at most 30 minutes per shard.

When a CPU stage exceeds its budget, resolve causes in order: first fix test bugs (real sleeps, unstubbed network timeouts, retry loops), next move heavy domain suites to `stage-b-cpu` with domain labels and calibrate `est_time`, and only expand shard count last because each extra nonempty shard adds runner setup and pytest collection overhead.
