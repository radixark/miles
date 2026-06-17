---
title: Stage
description: How CI stages are defined, how a test's suite maps to a stage, and what each stage does.
---

# Stage

A *stage* is one CI job in `.github/workflows/pr-test.yml`. A *suite* is the `suite=` value a test declares in `register_*_ci(...)`. Stage names and suite names are the same set, mapped **1:1**: a test runs in exactly the stage whose name equals its `suite`.

## Suite → stage mapping

The canonical suite list is `PER_COMMIT_SUITES` in `tests/ci/run_suite.py`, grouped by hardware backend (CPU / CUDA). Each entry has exactly one matching job in `pr-test.yml`. A test picks its stage purely by `suite=`; the stage job runs `run_suite.py --suite <name>`, which collects exactly the tests carrying that suite.

The mapping is kept in sync by hand on both sides:
- A `suite=` with no matching job never runs.
- A stage job whose suite no test uses runs zero tests and exits 0 (intended during incremental migration).

Stage names follow `stage-<tier>-<gpus>-<hw>` (or `stage-<tier>-<hw>` for CPU, e.g. `stage-a-cpu`): `tier ∈ {a, b, c}` classifies cost/role, `gpus` is the GPU count the test needs, `hw ∈ {cpu, h100, h200}` is the hardware class.

## Stage roster

| Stage / suite | Hardware | Runner labels (`runs_on`) | Shards | Depends on |
|---|---|---|---|---|
| `stage-a-cpu` | GitHub-hosted CPU | — (`ubuntu-latest`) | 4 | `resolve-ci-image` |
| `stage-b-cpu` | GitHub-hosted CPU | — (`ubuntu-latest`) | 1 | — |
| `stage-b-2-gpu-h200` | 2× H200 | `["h200","2gpu"]` | 1 | `resolve-ci-image`, `stage-a-cpu` |
| `stage-c-2-gpu-h200` | 2× H200 | `["h200","2gpu"]` | 2 | `resolve-ci-image`, `stage-a-cpu` |
| `stage-c-4-gpu-h200` | 4× H200 | `["h200","4gpu"]` | 3 | `resolve-ci-image`, `stage-a-cpu` |
| `stage-c-8-gpu-h100` | 8× H100 | `["h100","8gpu"]` | 2 | `resolve-ci-image`, `stage-a-cpu` |

`tier a` (CPU fast) gates the GPU fleet after `resolve-ci-image`; the GPU stages (`b` / `c`) all depend on `resolve-ci-image` and `stage-a-cpu`, and run concurrently with each other — the `b` / `c` letters classify role, they are not a sequential pipeline.

## What each stage does

**Image resolution (`resolve-ci-image`).** Before the GPU stages, a small `ubuntu-latest` job resolves the container image: it reads `ci-image-tag:` from the PR description (or the `ci_image_tag` dispatch input), defaults to `dev`, validates it is a bare tag, and outputs `radixark/miles:<tag>`. Every GPU stage uses this as its `container_image`. Distinct from this, the **`run-ci-image` label** (alongside `run-ci-all`, and any `workflow_dispatch`) makes each stage add `--match-all-labels`, running the full suite regardless of per-test labels — this is how you validate a PR that bumps the image.

**Dependencies / gating.** The job graph is `resolve-ci-image` → `stage-a-cpu` → all GPU stages (in parallel). GPU stages require `resolve-ci-image` to succeed; by default they also require `stage-a-cpu` to succeed, so a CPU-test failure short-circuits the expensive GPU fleet. The `bypass-fastfail` PR label relaxes only the `stage-a-cpu` failure gate and passes `--continue-on-error` to each stage; it does not bypass `resolve-ci-image`. `stage-b-cpu` has no dependency and runs alongside `stage-a-cpu`, outside the GPU gating path.

**Runner selection.** GPU stages request runners by label via `runs_on`, a JSON list passed through to `runs-on` — a runner must carry **all** listed labels (GPU class + count). CPU stages set `cpu_runner: true` and run on GitHub-hosted `ubuntu-latest` instead, so they don't occupy GPU-fleet slots.

**Launch.** Every stage is a thin caller of the reusable workflow `_run-ci.yml` (`uses: ./.github/workflows/_run-ci.yml`). The stage passes only `execute_command`, `runs_on`, `container_image`, and `cpu_runner`; `_run-ci.yml` owns the rest — starting the container, waiting for the GPU to be ready, installing dependencies, then running `execute_command` twice (once `--list-only` to print the plan, then for real). The stage itself holds no test logic; it is purely "which runner, which image, which command".

**Secrets.** Stages call the reusable workflow with `secrets: inherit`, so `_run-ci.yml` receives the caller's secrets (e.g. `WANDB_API_KEY`) without re-declaring each one.

**Sharding.** A stage with a `partition_id` matrix splits its tests across N shards; `run_suite.py` balances the shards by each test's `est_time`. Each shard is an independent job instance running the same `execute_command` with a different `--auto-partition-id`.

## Assumptions

- Suite ↔ stage stays 1:1 and is kept in sync manually across `run_suite.py` and `pr-test.yml`.
- Runner placement assumes the live fleet actually carries the requested `runs_on` labels for each GPU class and count.
- `est_time` only affects shard balancing and per-file timeout, never pass/fail.
