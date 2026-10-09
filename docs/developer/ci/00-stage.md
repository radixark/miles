---
title: Stage
description: How CI stages are defined, how a test's suite maps to a stage, and what each stage does.
---
A *stage* is one CI job in a Miles CI workflow. A *suite* is the `suite=` value a test declares in `register_*_ci(...)`. Stage names and suite names are the same set, mapped **1:1**: a test runs in exactly the stage whose name equals its `suite`.

## Suite → stage mapping

The canonical suite list is `CI_SUITES` in `tests/ci/run_suite.py`, grouped by hardware backend (CPU / CUDA / ROCm). Cadence does not change this inventory: regular, nightly, weekly, and release runs use the same stages. Every CPU and CUDA entry has one matching job in `pr-test.yml` or its B200 child `_run-ci-b200.yml`; `stage-c-4-gpu-mi350` has its matching job in `pr-test-rocm.yml`; the `nightly-` prefixed MI350 suites are owned by the external nightly described below. A test picks its stage purely by `suite=`; the stage job runs `run_suite.py --suite <name>`, which collects exactly the tests carrying that suite.

The mapping is kept in sync by hand on both sides:
- A `suite=` with no matching job never runs.
- A stage job whose suite no test uses runs zero tests and exits 0 (intended during incremental migration).

Stage names follow `stage-<tier>-<gpus>-<hw>` (or `stage-<tier>-<hw>` for CPU, e.g. `stage-a-cpu`): `tier ∈ {a, b, c}` classifies cost/role, `gpus` is the stage capacity (B200 allocates each test's `num_gpus` instead), `hw ∈ {cpu, h100, h200, b200, mi350}` is the hardware class. A `nightly-` prefix marks a suite that only the external MI350 nightly schedules, keeping its registrations in a separate namespace from the ones this repository's own workflows consume.

## Stage roster

| Stage / suite | Hardware | Runner labels (`runs_on`) | Shards | Depends on |
|---|---|---|---|---|
| `stage-a-cpu` | GitHub-hosted CPU | — (`ubuntu-latest`) | 4 | `resolve-ci-policy` |
| `stage-b-cpu` | GitHub-hosted CPU | — (`ubuntu-latest`) | 1 | `resolve-ci-policy` |
| `stage-b-2-gpu-h200` | 2× H200 | `["h200","2gpu"]` | 1 | both resolvers, `stage-a-cpu` |
| `stage-c-2-gpu-h200` | 2× H200 | `["h200","2gpu"]` | 2 | both resolvers, `stage-a-cpu` |
| `stage-c-4-gpu-h200` | 4× H200 | `["h200","4gpu"]` | 3 | both resolvers, `stage-a-cpu` |
| `stage-c-8-gpu-h100` | 8× H100 | `["h100","8gpu"]` | 1 | both resolvers, `stage-a-cpu` |
| `stage-c-8-gpu-h200` | 8× H200 | `["h200","8gpu"]` | 2 | both resolvers, `stage-a-cpu` |
| `stage-c-8-gpu-b200` | B200 pool | `["b200","<num_gpus>gpu"]` | one job per file | both resolvers, `stage-a-cpu` |
| `stage-c-4-gpu-b200` | same B200 pool | `["b200","<num_gpus>gpu"]` | one job per file | same plan, no 8-GPU stage barrier |
| `stage-c-4-gpu-mi350` | 4× MI350 | `["self-hosted","amd","mi350","4gpu"]` | 2 | both resolvers |
| `nightly-stage-c-2-gpu-mi350` | 2× MI350 | external nightly | — | — |
| `nightly-stage-c-4-gpu-mi350` | 4× MI350 | external nightly | — | — |
| `nightly-stage-c-8-gpu-mi350` | 8× MI350 | external nightly | — | — |

B200 jobs declare their minimum allocation with `register_cuda_ci(..., num_gpus=N)` for an integer N from 1 to 8. The existing B200 suite names select tests; they do not reserve a fixed partition. `_run-ci-b200.yml` plans all selected files at an exact commit and makes their jobs independently runnable. The host allocates any free GPU set, allowing `4+2+1+1`, `2+2+2+2`, or other fitting combinations. An 8-GPU job requires the whole host. Weekly uses the same scheduling rules. Jobs wait for GPUs after the runner accepts them, so their timeout includes admission waiting. The planner splits large selections into sequential batches whose summed file-timeout plus setup/cleanup allowances stay at or below 23 hours, below the 24-hour `GITHUB_TOKEN` lifetime and 5-day self-hosted job limit. Every job gets its batch's serial budget, covering even the last job to acquire GPUs; per-file execution timeouts remain unchanged. Files within each batch have no matrix parallelism cap.

The repository-wide `b200-oma` workflow queue keeps PRs and B200 file reruns in order. Host device locks remain necessary for old workflow revisions and hold through container cleanup. See `tests/ci/README.md` for deployment, cancellation, and orphan-container handling.

In `pr-test.yml`, `tier a` (CPU fast) gates PR-image preparation and the NVIDIA GPU fleet; its GPU stages (`b` / `c`) all depend on both resolvers and `stage-a-cpu`, and can run concurrently subject to available B200 GPU capacity — the `b` / `c` letters classify role, they are not a sequential pipeline. The MI350 stage has no CPU-test gate.

`pr-test.yml` and `pr-test-rocm.yml` listen to both `pull_request.labeled` and `pull_request.unlabeled`. Adding or removing any PR label starts a new run in the same PR concurrency group, cancels the previous queued or running run, and resolves CI policy from the remaining labels. Independently dispatched `/rerun-test` runs are unaffected.

Both workflows treat `pull_request.closed` (including merges) as cancellation-only: each close event shares its workflow's PR concurrency group, cancels queued or running runs of that workflow, and starts no resolver or test jobs.

A PR event starts jobs only when the PR's base is the default branch or the PR carries a `run-ci*` label. Otherwise `resolve-ci-policy` is skipped, and because every other job needs it, the run starts nothing. A stacked PR therefore runs `PR Test` only while labeled; retargeting a PR to `main` is an `edited` event, which does not start a run.

Both PR workflows expose reusable `workflow_call` entry points. Release CI calls only `pr-test.yml` for CPU and CUDA coverage. Called runs group concurrency by `inputs.ref`, so redispatching the same release branch cancels the older run; literal `pr-test-` and `pr-test-rocm-` prefixes keep the two workflows in separate concurrency groups.

## What each stage does

**Image resolution (`resolve-ci-image`).** In `pr-test.yml`, a small `ubuntu-latest` job selects, in order: the called workflow's `image_tag`, the `ci_image_tag` dispatch input, a current `pr-<num>` image, the PR description's `ci-image-tag:`, then `dev`. It validates the result is a bare tag and outputs `radixark/miles:<tag>`. `release-branch-cut.yml` passes the prune-exempt `release-vX.Y.Z-ci` tag recorded in `release-lock.json`.

The ROCm resolver uses only its dispatch input, defaults to its undated `rocm/sgl-dev` tag, and uses that default for reusable calls too. For PRs, it runs only when `stage-c-4-gpu-mi350` is selected. CPU-only PRs skip both CUDA image preparation and ROCm image resolution.

Distinct from image selection, the **`run-ci-image` label** selects the test scope — every enabled tag except `long`, `ft-short`, and `ft-long` — which validates an image bump without selecting those domains implicitly.

**Policy resolution (`resolve-ci-policy`).**

- `pull_request`, `schedule`, and `workflow_dispatch` only say how the workflow started; none itself implies a cadence or domain scope.
- Each Miles PR workflow passes trigger facts and, for PRs, the diff to `tests/ci/ci_policy.py`, which publishes the resolved policy and `skipped_stages` for `run_suite.py` and GPU job gates. `needs_cuda_image` derives from that selection and gates CUDA image preparation.
- A PR `nightly` label maps to nightly cadence.
- A scheduled run maps its exact UTC `github.event.schedule` cron: `0 15 * * 0-5` maps to nightly and `0 15 * * 6` maps to weekly; an unknown cron fails.
- A manual dispatch keeps regular cadence and has no PR labels. Both GPU workflows add `--match-all-labels` so an explicit manual operation runs the full regular GPU suites; CPU selection remains unchanged.
- A reusable `workflow_call` supplies an explicit cadence override because the called workflow inherits the caller's event name. Policy resolution deliberately runs the caller commit's `ci_policy.py`; only suite jobs check out the requested release ref.

A **nightly** policy selects every enabled tag except `long` and `ft-long`, admits both regular and `nightly=True` registrations, and disables fast-fail. **Weekly** and **release** select every enabled tag, admit both registration types, and disable fast-fail; release differs by never writing the rolling performance baseline. Regular cadence admits only regular registrations. All four cadences use the same stage inventory.

`run-ci-all` selects the full domain-tag set without changing cadence. `run-ci-image` selects every enabled tag except `long`, `ft-short`, and `ft-long`. If scope signals overlap, the precedence is `run-ci-all` > weekly/release full scope > nightly > `run-ci-image`. The resolved cadence and raw/synthetic labels are passed to `run_suite.py`, which computes one run policy (see [Labels](/developer/ci/01-label) for the subtraction semantics).

**PR GPU stage selection.** Before runner allocation, PR GPU stages are filtered by `runnable ∩ affected`: `runnable` reuses cadence/label selection, while `affected` maps changed registered tests to their suites. Known non-GPU paths affect none; unknown, missing, or malformed diffs affect all. `nightly`, `run-ci-all`, and `run-ci-image` add their runnable stages; `bypass-fastfail` does not. This path-based filter does not apply to CPU stages or scheduled/manual runs.

**Empty GPU jobs.** B200 uses `b200_plan.py` to select individual files and their minimum GPU budgets. Other CUDA and ROCm suite jobs first run their exact `--list-only` command on `ubuntu-latest`, including cadence, labels, arch dispatch, disabled tests, and partition arguments. `--github-output` publishes `has_tests` after partitioning. Only a successful plan with `has_tests=true` can allocate a GPU runner, which checks out the planned commit. Empty suites and shards skip GPU execution; planning errors fail the workflow. `/rerun-test` instead uses its existing trusted resolver, which requires one enabled registration at the exact requested SHA.

**Dependencies / gating.** In `pr-test.yml`, both CPU stages require only `resolve-ci-policy`. PR-image preparation requires selected CUDA tests and the `stage-a-cpu` success/bypass gate; NVIDIA GPU stages follow image resolution. `stage-b-cpu` stays parallel and does not gate that chain. Resolved nightly, weekly, or release cadence and the `bypass-fastfail` PR label admit the chain after an actual `stage-a-cpu` failure and make each suite continue after a test failure. This fast-fail exception does not itself select GPU tests or bypass policy or Docker/image failure; cadence and labels determine selection as described above. Scheduled, manual, and called release runs retain their existing image preparation.

**Runner selection.** CUDA stages request runners by label via `runs_on`, a JSON list passed through to `runs-on` — a runner must carry **all** listed labels (GPU class + count). CPU stages call `_run-cpu-ci.yml`, whose only job runs on GitHub-hosted `ubuntu-latest`, so they don't occupy GPU-fleet slots.

**NVSHMEM on 4-GPU H200.** When `runs_on` carries both `h200` and `4gpu`, `_run-ci.yml` sets `NVSHMEM_HCA_LIST` to a NIC name that matches nothing before any test command runs. NVSHMEM then skips its IB transports at init, and DeepEP keeps its intra-node traffic on NVLink. The override relies on the stage being single-node; every other stage keeps NVSHMEM's RDMA transports, and a multi-node stage must not inherit it.

That runner pool can include RoCE hosts, and GitHub attaches every job container to a runner-created bridge network (`container.options` cannot set `--network`). The host's RoCE GIDs are unusable inside that container, so NVSHMEM's IBRC/IBGDA connection setup fails and DeepEP low-latency waits until the job timeout.

**Arch dispatch.** `tests/ci/hardware.py::CUDA_STAGES` is the single source of truth for the CUDA taxonomy: each stage's GPU generation, GPU count, and runner labels. `CI_SUITES` and the default `/rerun-test` runner map derive from it. B200 overrides the runner width with the registration's `num_gpus`; a stage's `--suite` still names its generation.

A test normally runs at its home stage. A [dispatch label](/developer/ci/01-label) can send it to another generation instead, in which case the destination is the smallest stage on that generation with enough GPUs — a test declares its own GPU budget through `ray start --num-gpus` / `torchrun --nproc-per-node` rather than reading the devices it can see, so a wider stage simply leaves the surplus idle. Blackwell dispatch sends 2/4-GPU registrations to `stage-c-4-gpu-b200` and 8-GPU registrations to `stage-c-8-gpu-b200`.

Without a dispatch label nothing leaves its home stage, so scheduled and called runs are unaffected. Whatever stage actually executes a run is also the stage its performance baseline is keyed on, keeping the generations' numbers apart.

**Dependency boundary.** CUDA stages start from dependencies baked into `radixark/miles` and install `examples/multi_lora/requirements.txt` before `requirements.txt` through `tests/ci/reconcile_dependencies.py`. Installed CUDA 12/13 cuDNN versions are preserved with uv overrides; images without cuDNN use pip. Before each install, a dry-run gate rejects replacement of installed GPU runtimes or packages with at least 100 MiB of recorded installed files, including same-version reinstalls. These packages must be updated in the image.

CUDA stages then update the SGLang and Megatron-LM checkouts to the selected refs and expose all three source trees through `PYTHONPATH`; they do not rebuild or install those source trees after the container starts. Hosted CPU stages install `requirements.txt` and the fully pinned `tests/ci/requirements-ci-cpu.txt`, then expose the source trees through `PYTHONPATH` without editable installs or inline package lists. The ROCm stage uses the SGLang and Megatron-LM versions baked into `rocm/sgl-dev`, unless the run names a ref for either.

CUDA and CPU dependency refs resolve in this order: explicit dispatch input or PR-body directive, committed `release-lock.json`, then the moving `sglang-miles` / `miles-main` branch heads. A called release run therefore checks out its requested Miles `ref` and consumes the lockfile on that ref unless an explicit override exists. ROCm checks out the requested Miles ref but keeps the dependencies baked into its image unless the run names a ref for one.

**Launch.** CPU stages call `_run-cpu-ci.yml`; Hopper stages call `_run-ci.yml`. The B200 caller uses `_run-ci-b200.yml` to queue each run and plan one `_run-ci.yml` job per selected file; runner hooks allocate devices for each job. The CPU workflow has one execution job; each GPU workflow has a hosted planning job followed by conditional GPU execution.

Both workflows receive `execute_command` and an optional `ref`; CUDA callers additionally pass `runs_on` and `container_image`. The reusable workflows own checkout, runner setup, dependency and source resolution, and the two command invocations (first `--list-only`, then the real run). GPU planning needs only Python's standard library and runs before GPU container startup or dependency installation; each stage owns only which runner class, image, ref, and command to select.

**Secrets.** CPU/CUDA stages call their reusable workflow with `secrets: inherit`. The ROCm caller passes `WANDB_API_KEY` explicitly; GitHub withholds it from fork `pull_request` runs.

**Sharding.** A stage with a `partition_id` matrix splits its tests across N shards; `run_suite.py` balances the shards by each test's `est_time`. Each shard is an independent job instance running the same `execute_command` with a different `--auto-partition-id`.

Weekly runs keep the same shards but limit each Hopper and ROCm GPU matrix to one runner. B200 uses the same per-file GPU pool for every cadence, with no matrix parallelism cap within each bounded batch; each file starts when its declared GPU budget is available. Other stages remain independent: `stage-b-2-gpu-h200` and `stage-c-2-gpu-h200` may each occupy one 2-GPU runner at the same time. PR, nightly, and release runs retain their existing matrix parallelism.

## ROCm PR/nightly/weekly mirror

`pr-test-rocm.yml` exposes `pull_request`, exact nightly and weekly crons, `workflow_dispatch`, and `workflow_call`. PR runs use the same low-trust merge-commit model as `pr-test.yml`. It runs `stage-c-4-gpu-mi350` through `_run-ci-rocm.yml` on two 4-GPU MI350 runners and splits tests into two `est_time`-balanced shards; weekly alone limits the matrix to one runner at a time. It runs no CPU tests.

Reusable calls can select a Miles `ref` and cadence, but resolve the same undated `rocm/sgl-dev:miles-rocm724-mi35x` image as the other automatic paths. SGLang and Megatron-LM remain baked into that image. The release branch-cut workflow does not call ROCm CI or include it in the `release-ci` gate.

Only tests registered with `register_rocm_ci(suite="stage-c-4-gpu-mi350", ...)` run; the `nightly-` prefixed suites and CUDA registrations are not inherited. PR, nightly, weekly, and release runs consume the shared cadence and label policy: `run-ci-amd` selects the `amd`-labelled subset, other `run-ci-*` labels select matching subsets, nightly admits regular plus `nightly=True` registrations, and weekly or release selects every enabled registration.

Manual dispatch adds `--match-all-labels` and runs the full regular 4-GPU suite. The external nightly coverage of all three prefixed MI350 suites is described below.

Fork PRs use GitHub's standard `pull_request` protections: checkout tests the merge commit, repository secrets are withheld, and held runs follow the shared maintainer approval flow described in [`01-label.md`](01-label.md).

An 8-GPU CUDA case needs a separate 4-GPU `test_amd_<name>.py` variant rather than an `IS_HIP` branch in the original test.

Both runner containers can see all eight host GPUs through `/dev/dri`. Each runner restricts itself to four GPUs with `HIP_VISIBLE_DEVICES`, which `_run-ci-rocm.yml` forwards into the container.

**External MI350 nightly.** Miles declares `nightly-stage-c-2-gpu-mi350`, `nightly-stage-c-4-gpu-mi350`, and `nightly-stage-c-8-gpu-mi350` in `CI_SUITES` for the external nightly, and keeps the unprefixed `stage-c-4-gpu-mi350` for its own `pr-test-rocm.yml`. The three prefixed suites are run by the external [`sgl-project/sglang` ROCm 10 nightly workflow](https://github.com/sgl-project/sglang/blob/main/.github/workflows/nightly-test-amd-miles-rocm10.yml).

## Assumptions

- Suite-to-stage mappings are maintained manually across `run_suite.py`, the Miles workflows, and the external MI350 nightly workflow.
- Runner placement assumes the live fleet actually carries the requested `runs_on` labels for each GPU class and count.
- `est_time` only affects shard balancing and per-file timeout, never pass/fail.
