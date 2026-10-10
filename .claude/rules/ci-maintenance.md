---
paths:
  - ".github/workflows/**/*.yml"
  - "tests/ci/**/*.py"
  - "docs/developer/ci/**/*.md"
---

# CI Maintenance

## Shard runtime budget

When maintaining CI scope or sharding, check regular-cadence `run-ci-image`
and nightly selection in each affected stage. Exclude disabled registrations
and preserve each cadence's eligibility filter. Apply `auto_partition`, sum
the registered `est_time` values in each shard, and compare the largest sum
with the targets in `docs/developer/ci/00-stage.md`. Do not add environment setup, cleanup,
or queue time.

Use `run-ci-image` as the broad PR sizing baseline. It does not admit
`nightly=True` registrations; nightly has a larger scope. Explicit extra
labels and `run-ci-all` can exceed this baseline, and a `long` / `ft-long`
file can exceed the target alone.

Try shard counts in ascending multiples of the matching runner pool capacity
and choose the first whose largest shard sum meets the budget. Cap concurrency
at that capacity (one per matrix for weekly).
Preserve hosted empty-shard planning and selected-test coverage. Adding shards
cannot shorten an indivisible test file and adds setup cost for each nonempty
shard; do not change test coverage merely to meet the runtime target.
