---
title: Stage
description: How CI stages are defined, how a test's suite maps to a stage, and what each stage does.
---
A *stage* is one CI job in a Miles CI workflow. A *suite* is the `suite=` value a test declares in `register_*_ci(...)`. Stage names and suite names are the same set, mapped **1:1**: a test runs in exactly the stage whose name equals its `suite`.

## Suite → stage mapping

The canonical suite list is `CI_SUITES` in `tests/ci/run_suite.py`, grouped by hardware backend (CPU / CUDA / ROCm). Cadence does not change this inventory: regular, nightly, weekly, and release runs use the same stages. Every CPU and CUDA entry has one matching job in `pr-test.yml`; `stage-c-4-gpu-mi350` has its matching job in `pr-test-rocm.yml`; the `nightly-` prefixed MI350 suites are owned by the external nightly described below. A test picks its stage purely by `suite=`; the stage job runs `run_suite.py --suite <name>`, which collects exactly the tests carrying that suite.

The mapping is kept in sync by hand on both sides:
- A `suite=` with no matching job never runs.
- A stage job whose suite no test uses runs zero tests and exits 0 (intended during incremental migration).

Stage names follow `stage-<tier>-<gpus>-<hw>` (or `stage-<tier>-<hw>` for CPU, e.g. `stage-a-cpu`): `tier ∈ {a, b, c}` classifies cost/role, `gpus` is the GPU count the test needs, `hw ∈ {cpu, h100, h200, mi350}` is the hardware class. A `nightly-` prefix marks a suite that only the external MI350 nightly schedules, keeping its registrations in a separate namespace from the ones this repository's own workflows consume.

## Stage roster

| Stage / suite | Hardware | Runner labels (`runs_on`) | Shards | Depends on |
|---|---|---|---|---|
| `stage-a-cpu` | GitHub-hosted CPU | — (`ubuntu-latest`) | 4 | `resolve-ci-policy` |
| `stage-b-cpu` | GitHub-hosted CPU | — (`ubuntu-latest`) | 1 | `resolve-ci-policy` |
| `stage-b-2-gpu-h200` | 2× H200 | `["h200","2gpu"]` | 1 | both resolvers, `stage-a-cpu` |
| `stage-c-2-gpu-h200` | 2× H200 | `["h200","2gpu"]` | 2 | both resolvers, `stage-a-cpu` |
| `stage-c-4-gpu-h200` | 4× H200 | `["h200","4gpu"]` | 3 | both resolvers, `stage-a-cpu` |
| `stage-c-8-gpu-h100` | 8× H100 | `["h100","8gpu"]` | 2 | both resolvers, `stage-a-cpu` |
| `stage-c-8-gpu-h200` | 8× H200 | `["h200","8gpu"]` | 2 | both resolvers, `stage-a-cpu` |
| `stage-c-4-gpu-mi350` | 4× MI350 | `["self-hosted","amd","mi350","4gpu"]` | 2 | both resolvers |
| `nightly-stage-c-2-gpu-mi350` | 2× MI350 | external nightly | — | — |
| `nightly-stage-c-4-gpu-mi350` | 4× MI350 | external nightly | — | — |
| `nightly-stage-c-8-gpu-mi350` | 8× MI350 | external nightly | — | — |

In `pr-test.yml`, `tier a` (CPU fast) gates PR-image preparation and the NVIDIA GPU fleet; its GPU stages (`b` / `c`) all depend on both resolvers and `stage-a-cpu`, and run concurrently with each other — the `b` / `c` letters classify role, they are not a sequential pipeline. The MI350 stage has no CPU-test gate.

`pr-test.yml` treats `pull_request.closed` as cancellation-only: the close event shares the PR's concurrency group, cancels any queued or running `PR Test` run, and starts no resolver or test jobs.

Both PR workflows are also reusable `workflow_call` entry points for release CI. Called runs group concurrency by `inputs.ref`, so redispatching the same release branch cancels the older run; literal `pr-test-` and `pr-test-rocm-` prefixes keep the CUDA and ROCm groups from cancelling each other under the same branch-cut caller.

## What each stage does

**Image resolution (`resolve-ci-image`).** In `pr-test.yml`, a small `ubuntu-latest` job takes the called workflow's `image_tag` first, then reads `ci-image-tag:` from the PR description or the `ci_image_tag` dispatch input, defaults to `dev`, validates the result is a bare tag, and outputs `radixark/miles:<tag>`. `release-branch-cut.yml` passes the prune-exempt `release-vX.Y.Z-ci` tag recorded in `release-lock.json`.

The ROCm resolver uses only its dispatch input, defaults to its undated `rocm/sgl-dev` tag, and uses that default for called release runs too.

Distinct from image selection, the **`run-ci-image` label** selects the test scope — every enabled tag except `long`, `ft-short`, and `ft-long` — which validates an image bump without selecting those domains implicitly.

**Policy resolution (`resolve-ci-policy`).**

- `pull_request`, `schedule`, and `workflow_dispatch` only say how the workflow started; none itself implies a cadence or domain scope.
- Each Miles PR workflow passes trigger facts and, for PRs, the diff to `tests/ci/ci_policy.py`, which publishes the resolved policy and `skipped_stages` for `run_suite.py` and GPU job gates.
- A PR `nightly` label maps to nightly cadence.
- A scheduled run maps its exact UTC `github.event.schedule` cron: `0 15 * * 0-5` maps to nightly and `0 15 * * 6` maps to weekly; an unknown cron fails.
- A manual dispatch keeps regular cadence and has no PR labels. Both GPU workflows add `--match-all-labels` so an explicit manual operation runs the full regular GPU suites; CPU selection remains unchanged.
- A reusable `workflow_call` supplies an explicit cadence override because the called workflow inherits the caller's event name. Policy resolution deliberately runs the caller commit's `ci_policy.py`; only suite jobs check out the requested release ref.

A **nightly** policy selects every enabled tag except `long` and `ft-long`, admits both regular and `nightly=True` registrations, and disables fast-fail. **Weekly** and **release** select every enabled tag, admit both registration types, and disable fast-fail; release differs by never writing the rolling performance baseline. Regular cadence admits only regular registrations. All four cadences use the same stage inventory.

`run-ci-all` selects the full domain-tag set without changing cadence. `run-ci-image` selects every enabled tag except `long`, `ft-short`, and `ft-long`. If scope signals overlap, the precedence is `run-ci-all` > weekly/release full scope > nightly > `run-ci-image`. The resolved cadence and raw/synthetic labels are passed to `run_suite.py`, which computes one run policy (see [Labels](/ci/01-label) for the subtraction semantics).

**PR GPU stage selection.** Before runner allocation, PR GPU stages are filtered by `runnable ∩ affected`: `runnable` reuses cadence/label selection, while `affected` maps changed registered tests to their suites. Known non-GPU paths affect none; unknown, missing, or malformed diffs affect all. `nightly`, `run-ci-all`, and `run-ci-image` add their runnable stages; `bypass-fastfail` does not. CPU stages and scheduled/manual runs are not filtered.

**Dependencies / gating.** In `pr-test.yml`, both CPU stages require only `resolve-ci-policy`. PR-image preparation is gated by `stage-a-cpu`, and the NVIDIA GPU stages follow image resolution; `stage-b-cpu` stays parallel and does not gate that chain. Resolved nightly, weekly, or release cadence and the `bypass-fastfail` PR label admit the chain after an actual `stage-a-cpu` failure and make each suite continue after a test failure; none bypasses policy or Docker/image failure.

**Runner selection.** CUDA stages request runners by label via `runs_on`, a JSON list passed through to `runs-on` — a runner must carry **all** listed labels (GPU class + count). CPU stages call `_run-cpu-ci.yml`, whose only job runs on GitHub-hosted `ubuntu-latest`, so they don't occupy GPU-fleet slots.

**Dependency boundary.** CUDA stages start from dependencies baked into `radixark/miles`, reconcile Miles runtime dependencies from `requirements.txt`, update the SGLang and Megatron-LM checkouts to the selected refs, and expose all three source trees through `PYTHONPATH`; they do not rebuild or install the Miles, SGLang, or Megatron-LM source trees after the container starts. The hosted CPU stages install dependencies from `requirements.txt` and the fully pinned `tests/ci/requirements-ci-cpu.txt`, then expose the Miles, SGLang, and Megatron-LM source trees through `PYTHONPATH` without editable installs or inline package lists. The ROCm stage instead uses the SGLang and Megatron-LM versions baked into `rocm/sgl-dev`.

CUDA and CPU dependency refs resolve in this order: explicit dispatch input or PR-body directive, committed `release-lock.json`, then the moving `sglang-miles` / `miles-main` branch heads. A called release run therefore checks out its requested Miles `ref` and consumes the lockfile on that ref unless an explicit override exists. ROCm checks out the requested Miles ref but keeps the dependencies baked into its image.

**Launch.** Each CPU/CUDA stage is a thin caller of one hardware-specific reusable workflow: CPU stages use `_run-cpu-ci.yml`, while CUDA stages use `_run-ci.yml`. Each reusable workflow declares only its matching job, so GitHub does not add a skipped CPU sibling to CUDA stages or a skipped CUDA sibling to CPU stages.

Both workflows receive `execute_command` and an optional `ref`; CUDA callers additionally pass `runs_on` and `container_image`. The reusable workflows own checkout, runner setup, dependency and source resolution, and the two command invocations (first `--list-only`, then the real run); each stage owns only which runner class, image, ref, and command to select.

**Secrets.** CPU/CUDA stages call their reusable workflow with `secrets: inherit`. The ROCm caller passes `WANDB_API_KEY` explicitly; GitHub withholds it from fork `pull_request` runs.

**Sharding.** A stage with a `partition_id` matrix splits its tests across N shards; `run_suite.py` balances the shards by each test's `est_time`. Each shard is an independent job instance running the same `execute_command` with a different `--auto-partition-id`.

Weekly runs keep the same shards but set each GPU matrix's `max-parallel` to one, so each stage occupies at most one matching runner. Stages remain independent: `stage-b-2-gpu-h200` and `stage-c-2-gpu-h200` may each occupy one 2-GPU runner at the same time. PR, nightly, and release runs retain their existing matrix parallelism.

## ROCm PR/nightly/weekly/release mirror

`pr-test-rocm.yml` exposes `pull_request`, exact nightly and weekly crons, `workflow_dispatch`, and `workflow_call`. PR runs use the same low-trust merge-commit model as `pr-test.yml`. It runs `stage-c-4-gpu-mi350` through `_run-ci-rocm.yml` on two 4-GPU MI350 runners and splits tests into two `est_time`-balanced shards; weekly alone limits the matrix to one runner at a time. It runs no CPU tests.

A called release run checks out the supplied Miles `ref` with `cadence=release` but resolves the same undated `rocm/sgl-dev:miles-rocm720-mi35x` image as the other automatic paths. SGLang and Megatron-LM remain baked into that image, so release ROCm is a smoke signal and manual dispatch exposes no dependency-ref inputs.

Only tests registered with `register_rocm_ci(suite="stage-c-4-gpu-mi350", ...)` run; the `nightly-` prefixed suites and CUDA registrations are not inherited. PR, nightly, weekly, and release runs consume the shared cadence and label policy: `run-ci-amd` selects the `amd`-labelled subset, other `run-ci-*` labels select matching subsets, nightly admits regular plus `nightly=True` registrations, and weekly or release selects every enabled registration.

Manual dispatch adds `--match-all-labels` and runs the full regular 4-GPU suite. The external nightly coverage of all three prefixed MI350 suites is described below.

Fork PRs use GitHub's standard `pull_request` protections: checkout tests the merge commit, repository secrets are withheld, and held runs follow the shared maintainer approval flow described in [`01-label.md`](01-label.md).

An 8-GPU CUDA case needs a separate 4-GPU `test_amd_<name>.py` variant rather than an `IS_HIP` branch in the original test.

Both runner containers can see all eight host GPUs through `/dev/dri`. Each runner restricts itself to four GPUs with `HIP_VISIBLE_DEVICES`, which `_run-ci-rocm.yml` forwards into the container.

**External MI350 nightly.** Miles declares `nightly-stage-c-2-gpu-mi350`, `nightly-stage-c-4-gpu-mi350`, and `nightly-stage-c-8-gpu-mi350` in `CI_SUITES` for the external nightly, and keeps the unprefixed `stage-c-4-gpu-mi350` for its own `pr-test-rocm.yml`. The three prefixed suites are run by the external [`sgl-project/sglang` ROCm 7.2 nightly workflow](https://github.com/sgl-project/sglang/blob/main/.github/workflows/nightly-test-amd-miles-rocm720.yml).

## Assumptions

- Suite-to-stage mappings are maintained manually across `run_suite.py`, the Miles workflows, and the external MI350 nightly workflow.
- Runner placement assumes the live fleet actually carries the requested `runs_on` labels for each GPU class and count.
- `est_time` only affects shard balancing and per-file timeout, never pass/fail.
