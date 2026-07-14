---
title: Stage
description: How CI stages are defined, how a test's suite maps to a stage, and what each stage does.
---

# Stage

A *stage* is one CI job in `.github/workflows/pr-test.yml`. A *suite* is the `suite=` value a test declares in `register_*_ci(...)`. Stage names and suite names are the same set, mapped **1:1**: a test runs in exactly the stage whose name equals its `suite`.

## Suite → stage mapping

The canonical suite list is `CI_SUITES` in `tests/ci/run_suite.py`, grouped by hardware backend (CPU / CUDA / ROCm). Cadence does not change this inventory: regular and nightly runs use the same stages. Every CPU and CUDA entry has one matching job in `pr-test.yml`; the ROCm entries are outside this workflow's roster. A test picks its stage purely by `suite=`; the stage job runs `run_suite.py --suite <name>`, which collects exactly the tests carrying that suite.

The mapping is kept in sync by hand on both sides:
- A `suite=` with no matching job never runs.
- A stage job whose suite no test uses runs zero tests and exits 0 (intended during incremental migration).

Stage names follow `stage-<tier>-<gpus>-<hw>` (or `stage-<tier>-<hw>` for CPU, e.g. `stage-a-cpu`): `tier ∈ {a, b, c}` classifies cost/role, `gpus` is the GPU count the test needs, `hw ∈ {cpu, h100, h200}` is the hardware class.

## Stage roster

| Stage / suite | Hardware | Runner labels (`runs_on`) | Shards | Depends on |
|---|---|---|---|---|
| `stage-a-cpu` | GitHub-hosted CPU | — (`ubuntu-latest`) | 4 | `resolve-ci-policy`, `resolve-ci-image` |
| `stage-b-cpu` | GitHub-hosted CPU | — (`ubuntu-latest`) | 1 | `resolve-ci-policy`, `resolve-ci-image` |
| `stage-b-2-gpu-h200` | 2× H200 | `["h200","2gpu"]` | 1 | both resolvers, `stage-a-cpu` |
| `stage-c-2-gpu-h200` | 2× H200 | `["h200","2gpu"]` | 2 | both resolvers, `stage-a-cpu` |
| `stage-c-4-gpu-h200` | 4× H200 | `["h200","4gpu"]` | 3 | both resolvers, `stage-a-cpu` |
| `stage-c-8-gpu-h100` | 8× H100 | `["h100","8gpu"]` | 2 | both resolvers, `stage-a-cpu` |
| `stage-c-8-gpu-h200` | 8× H200 | `["h200","8gpu"]` | 2 | both resolvers, `stage-a-cpu` |

`tier a` (CPU fast) gates the GPU fleet after both resolvers; the GPU stages (`b` / `c`) all depend on both resolvers and `stage-a-cpu`, and run concurrently with each other — the `b` / `c` letters classify role, they are not a sequential pipeline.

## What each stage does

**Image resolution (`resolve-ci-image`).** Before the GPU stages, a small `ubuntu-latest` job resolves the container image: it reads `ci-image-tag:` from the PR description (or the `ci_image_tag` dispatch input), defaults to `dev`, validates it is a bare tag, and outputs `radixark/miles:<tag>`. Every GPU stage uses this as its `container_image`. Distinct from this, the **`run-ci-image` label** selects the image scope — every enabled tag except `ft-short`/`ft-long` — which is how you validate a PR that bumps the image without scheduling FT tests.

**Policy resolution (`resolve-ci-policy`).**

- `pull_request`, `schedule`, and `workflow_dispatch` only say how the workflow started; none itself implies a cadence or domain scope.
- The workflow job passes trigger facts to `tests/ci/ci_policy.py` and publishes its `cadence`, `raw_labels`, and `bypass_fastfail` outputs. That module owns trigger adaptation and the shared `resolve_policy` consumed by `run_suite.py`.
- A PR `nightly` label maps to nightly cadence.
- A scheduled run maps its exact `github.event.schedule` cron: the current `0 15 * * *` entry maps to nightly, an unknown cron fails, and a future weekly entry must add its own mapping.
- A manual dispatch keeps regular cadence and has no PR labels, so it runs the ordinary always-on selection; its operation inputs do not broaden CI scope.

A **nightly** policy selects every enabled tag except `ft-long`, admits both regular and `nightly=True` registrations, and disables fast-fail. Regular cadence admits only regular registrations. Both cadences use the same stage inventory.

`run-ci-all` selects the full domain-tag set without changing cadence. `run-ci-image` selects every enabled non-FT tag. If scope signals overlap, the precedence is `run-ci-all` > nightly > `run-ci-image`. The resolved cadence and raw/synthetic labels are passed to `run_suite.py`, which computes one run policy (see docs/ci/01-label.md for the subtraction semantics).

**Dependencies / gating.** Both CPU stages require both resolvers. GPU stages also require both resolvers and, by default, a successful `stage-a-cpu`, so a CPU-test failure short-circuits the expensive GPU fleet. Resolved nightly cadence and the `bypass-fastfail` PR label relax only the `stage-a-cpu` failure gate and make each suite continue after a test failure; neither bypasses resolver failure.

**Runner selection.** GPU stages request runners by label via `runs_on`, a JSON list passed through to `runs-on` — a runner must carry **all** listed labels (GPU class + count). CPU stages set `cpu_runner: true` and run on GitHub-hosted `ubuntu-latest` instead, so they don't occupy GPU-fleet slots.

**Launch.** Every stage is a thin caller of the reusable workflow `_run-ci.yml` (`uses: ./.github/workflows/_run-ci.yml`). The stage passes only `execute_command`, `runs_on`, `container_image`, and `cpu_runner`; `_run-ci.yml` owns the rest — starting the container, waiting for the GPU to be ready, installing dependencies, then running `execute_command` twice (once `--list-only` to print the plan, then for real). Each `execute_command` receives the same resolved cadence and labels; the stage itself holds no selection policy.

**Secrets.** Stages call the reusable workflow with `secrets: inherit`, so `_run-ci.yml` receives the caller's secrets (e.g. `WANDB_API_KEY`) without re-declaring each one.

**Sharding.** A stage with a `partition_id` matrix splits its tests across N shards; `run_suite.py` balances the shards by each test's `est_time`. Each shard is an independent job instance running the same `execute_command` with a different `--auto-partition-id`.

## Assumptions

- Suite ↔ stage stays 1:1 and is kept in sync manually across `run_suite.py` and `pr-test.yml`.
- Runner placement assumes the live fleet actually carries the requested `runs_on` labels for each GPU class and count.
- `est_time` only affects shard balancing and per-file timeout, never pass/fail.
