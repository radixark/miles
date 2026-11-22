---
paths:
  - ".github/workflows/**/*.yml"
  - "tests/ci/**/*.py"
  - "docs/developer/ci/**/*.md"
---

# CI Maintenance

## Shard runtime budget

When maintaining CI scope or sharding, check regular-cadence `run-ci-image` and nightly selection in each affected stage. Exclude disabled registrations and preserve each cadence's eligibility filter. Apply `auto_partition`, sum the registered `est_time` values in each shard, and target a largest sum of at most two hours for regular-cadence `run-ci-image` and less than two hours for nightly. Do not add environment setup, cleanup, or queue time.

Use `run-ci-image` as the broad PR sizing baseline. It does not admit
`nightly=True` registrations; nightly has a larger scope. Explicit extra
labels and `run-ci-all` can exceed this baseline, and a `long` / `ft-long`
file can exceed the target alone.

Try shard counts in ascending multiples of the matching runner pool capacity and choose the first whose largest shard sum meets the budget. Do not set `max-parallel` on Hopper or ROCm shard matrices; let available runners control execution. Preserve hosted empty-shard planning and selected-test coverage. Adding shards cannot shorten an indivisible test file and adds setup cost for each nonempty shard; do not change test coverage merely to meet the runtime target.
