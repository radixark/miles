---
title: Stage
description: How CI stages are defined, how a test's suite maps to a stage, and what each stage does.
---

# Stage

A *stage* is one CI job in a Miles CI workflow. A *suite* is the `suite=` value a test declares in `register_*_ci(...)`. Stage names and suite names are the same set, mapped **1:1**: a test runs in exactly the stage whose name equals its `suite`.

## Suite → stage mapping

The canonical suite list is `CI_SUITES` in `tests/ci/run_suite.py`, grouped by hardware backend (CPU / CUDA / ROCm). Cadence does not change this inventory: regular and nightly runs use the same stages. Every CPU and CUDA entry has one matching job in `pr-test.yml`; `stage-c-4-gpu-mi300x` has its matching job in `pr-test-rocm.yml`, while the MI350 suites are owned by the external nightly described below. A test picks its stage purely by `suite=`; the stage job runs `run_suite.py --suite <name>`, which collects exactly the tests carrying that suite.

The mapping is kept in sync by hand on both sides:
- A `suite=` with no matching job never runs.
- A stage job whose suite no test uses runs zero tests and exits 0 (intended during incremental migration).

Stage names follow `stage-<tier>-<gpus>-<hw>` (or `stage-<tier>-<hw>` for CPU, e.g. `stage-a-cpu`): `tier ∈ {a, b, c}` classifies cost/role, `gpus` is the GPU count the test needs, `hw ∈ {cpu, h100, h200, mi300x, mi350}` is the hardware class.

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
| `stage-c-4-gpu-mi300x` | 4× MI300X | `["self-hosted","amd","mi300x","4gpu"]` | 2 | both resolvers |

In `pr-test.yml`, `tier a` (CPU fast) gates the NVIDIA GPU fleet after both resolvers; its GPU stages (`b` / `c`) all depend on both resolvers and `stage-a-cpu`, and run concurrently with each other — the `b` / `c` letters classify role, they are not a sequential pipeline. The MI300X stage has no CPU-test gate.

`pr-test.yml` treats `pull_request.closed` as cancellation-only: the close event shares the PR's concurrency group, cancels any queued or running `PR Test` run, and starts no resolver or test jobs.

## What each stage does

**Image resolution (`resolve-ci-image`).** In `pr-test.yml`, a small `ubuntu-latest` job reads `ci-image-tag:` from the PR description (or the `ci_image_tag` dispatch input), defaults to `dev`, validates it is a bare tag, and outputs `radixark/miles:<tag>`. The ROCm resolver uses only its dispatch input, defaults to its dated `rocm/sgl-dev` tag, and uses that same default for PR and nightly runs. Distinct from this, the **`run-ci-image` label** selects the image scope — every enabled tag except `long`, `ft-short`, and `ft-long` — which validates an image bump without selecting those domains implicitly.

**Policy resolution (`resolve-ci-policy`).**

- `pull_request` / `pull_request_target`, `schedule`, and `workflow_dispatch` only say how the workflow started; none itself implies a cadence or domain scope.
- Each Miles PR workflow passes trigger facts to `tests/ci/ci_policy.py` and publishes its `cadence`, `raw_labels`, and `bypass_fastfail` outputs. That module owns trigger adaptation and the shared `resolve_policy` consumed by `run_suite.py`.
- A PR `nightly` label maps to nightly cadence.
- A scheduled run maps its exact `github.event.schedule` cron: the current `0 15 * * *` entry maps to nightly, an unknown cron fails, and a future weekly entry must add its own mapping.
- A manual dispatch keeps regular cadence and has no PR labels. `pr-test.yml` therefore runs the ordinary always-on selection, while the dedicated ROCm dispatch adds `--match-all-labels` to preserve its full regular MI300X run.

A **nightly** policy selects every enabled tag except `long` and `ft-long`, admits both regular and `nightly=True` registrations, and disables fast-fail. Regular cadence admits only regular registrations. Both cadences use the same stage inventory.

`run-ci-all` selects the full domain-tag set without changing cadence. `run-ci-image` selects every enabled tag except `long`, `ft-short`, and `ft-long`. If scope signals overlap, the precedence is `run-ci-all` > nightly > `run-ci-image`. The resolved cadence and raw/synthetic labels are passed to `run_suite.py`, which computes one run policy (see docs/ci/01-label.md for the subtraction semantics).

**Dependencies / gating.** In `pr-test.yml`, both CPU stages require both resolvers. Its GPU stages also require both resolvers and, by default, a successful `stage-a-cpu`, so a CPU-test failure short-circuits the expensive NVIDIA fleet. Resolved nightly cadence and the `bypass-fastfail` PR label relax only the `stage-a-cpu` failure gate and make each suite continue after a test failure; neither bypasses resolver failure.

**Runner selection.** CUDA stages request runners by label via `runs_on`, a JSON list passed through to `runs-on` — a runner must carry **all** listed labels (GPU class + count). CPU stages call `_run-cpu-ci.yml`, whose only job runs on GitHub-hosted `ubuntu-latest`, so they don't occupy GPU-fleet slots.

**Dependency boundary.** CUDA stages start from dependencies baked into `radixark/miles`, reconcile Miles runtime dependencies from `requirements.txt`, update the SGLang and Megatron-LM checkouts to the selected refs, and expose all three source trees through `PYTHONPATH`; they do not rebuild or install the Miles, SGLang, or Megatron-LM source trees after the container starts. The hosted CPU stages install dependencies from `requirements.txt` and the fully pinned `tests/ci/requirements-ci-cpu.txt`, then expose the Miles, SGLang, and Megatron-LM source trees through `PYTHONPATH` without editable installs or inline package lists. The ROCm stage instead uses the SGLang and Megatron-LM versions baked into `rocm/sgl-dev`.

**Launch.** Each CPU/CUDA stage is a thin caller of one hardware-specific reusable workflow: CPU stages use `_run-cpu-ci.yml`, while CUDA stages use `_run-ci.yml`. Each reusable workflow declares only its matching job, so GitHub does not add a skipped CPU sibling to CUDA stages or a skipped CUDA sibling to CPU stages.

Both workflows receive `execute_command`; CUDA callers additionally pass `runs_on` and `container_image`. The reusable workflows own runner setup, dependency and source resolution, and the two command invocations (first `--list-only`, then the real run); each stage owns only which runner class, image, and command to select.

**Secrets.** CPU/CUDA stages call their reusable workflow with `secrets: inherit`. The ROCm caller passes `WANDB_API_KEY` explicitly for schedules, manual runs, and same-repository PRs, but passes an empty value for fork PRs.

**Sharding.** A stage with a `partition_id` matrix splits its tests across N shards; `run_suite.py` balances the shards by each test's `est_time`. Each shard is an independent job instance running the same `execute_command` with a different `--auto-partition-id`.

## ROCm PR/nightly mirror

`pr-test-rocm.yml` has PR-level, exact nightly cron, and `workflow_dispatch` entry points. Its `pull_request_target` entry keeps the workflow and authorization code on the trusted base branch, while adapting the event to the same `resolve-ci-policy` path as `pr-test.yml`. It runs `stage-c-4-gpu-mi300x` through `_run-ci-rocm.yml` on two 4-GPU MI300X runners, resolves `rocm/sgl-dev:<tag>`, and splits tests into two `est_time`-balanced shards. It runs no CPU tests. SGLang and Megatron-LM come from the image, so manual dispatch exposes no dependency-ref inputs.

Only tests registered with `register_rocm_ci(suite="stage-c-4-gpu-mi300x", ...)` run; CUDA registrations are not inherited. PR and nightly runs consume the shared cadence and label policy: `run-ci-amd` selects the full MI300X set, other `run-ci-*` labels can select matching subsets, and nightly cadence admits regular plus nightly-only registrations. Manual dispatch adds `--match-all-labels` and runs the full regular suite.

For fork PRs, a base-branch authorization step admits the privileged MI300X stage only when the event carries a canonical `run-ci-*` label. The reusable workflow then checks out `refs/pull/<number>/merge` explicitly, so it tests the PR merge result without trusting workflow or gate code from the fork. Fork jobs receive no `WANDB_API_KEY`; same-repository PRs, schedules, and manual dispatches do not need the label gate.

An 8-GPU CUDA case needs a separate 4-GPU `test_amd_<name>.py` variant rather than an `IS_HIP` branch in the original test.

Both runner containers can see all eight host GPUs through `/dev/dri`. Each runner restricts itself to four GPUs with `HIP_VISIBLE_DEVICES`, which `_run-ci-rocm.yml` forwards into the container.

**External MI350 nightly.** Miles declares `stage-c-2-gpu-mi350`, `stage-c-4-gpu-mi350`, and `stage-c-8-gpu-mi350` in `CI_SUITES`, but does not schedule them in its own workflows. All three are run by the external [`sgl-project/sglang` ROCm 7.2 nightly workflow](https://github.com/sgl-project/sglang/blob/main/.github/workflows/nightly-test-amd-miles-rocm720.yml).

## Assumptions

- Suite-to-stage mappings are maintained manually across `run_suite.py`, the Miles workflows, and the external MI350 nightly workflow.
- Runner placement assumes the live fleet actually carries the requested `runs_on` labels for each GPU class and count.
- `est_time` only affects shard balancing and per-file timeout, never pass/fail.
