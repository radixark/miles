---
title: Stage
description: How CI stages are defined, how a test's suite maps to a stage, and what each stage does.
---
A *stage* is one CI job in a Miles CI workflow. A *suite* is the `suite=` value a test declares in `register_*_ci(...)`. Stage names and suite names are the same set, mapped **1:1**: a test runs in exactly the stage whose name equals its `suite`.

## Suite → stage mapping

The canonical suite list is `CI_SUITES` in `tests/ci/run_suite.py`, grouped by hardware backend (CPU / CUDA / ROCm). Cadence does not change this inventory: regular, nightly, weekly, and release runs use the same stages. Every CPU and CUDA entry has one matching job in `pr-test.yml` or its B200 child `_run-ci-b200.yml`; every ROCm entry has its matching job in `pr-test-rocm.yml`, and the external nightly described below runs the same ROCm suites. A test picks its stage purely by `suite=`; the stage job runs `run_suite.py --suite <name>`, which collects exactly the tests carrying that suite.

The mapping is kept in sync by hand on both sides:
- A `suite=` with no matching job never runs.
- A stage job whose suite no test uses runs zero tests and exits 0 (intended during incremental migration).

Stage names follow `stage-<tier>-<gpus>-<hw>` (or `stage-<tier>-<hw>` for CPU, e.g. `stage-a-cpu`): `tier ∈ {a, b, c}` classifies cost/role, `gpus` is the stage capacity (B200 allocates each test's `num_gpus` instead), `hw ∈ {cpu, h100, h200, b200, mi350}` is the hardware class.

## Stage roster

| Stage / suite | Hardware | Runner labels (`runs_on`) | Shards | Depends on |
|---|---|---|---|---|
| `stage-a-cpu` | GitHub-hosted CPU | — (`ubuntu-latest`) | 4 | `resolve-ci-policy` |
| `stage-b-cpu` | GitHub-hosted CPU | — (`ubuntu-latest`) | 4 | `resolve-ci-policy`, `stage-a-cpu` |
| `stage-b-2-gpu-h200` | 2× H200 | `["h200","2gpu"]` | 2 | both resolvers, `stage-a-cpu` |
| `stage-c-2-gpu-h200` | 2× H200 | `["h200","2gpu"]` | 4 | both resolvers, `stage-a-cpu` |
| `stage-c-4-gpu-h200` | 4× H200 | `["h200","4gpu"]` | 9 | both resolvers, `stage-a-cpu` |
| `stage-c-8-gpu-h100` | 8× H100 | `["h100","8gpu"]` | 2 regular / 3 otherwise | both resolvers, `stage-a-cpu` |
| `stage-c-8-gpu-h200` | 8× H200 | `["h200","8gpu"]` | 4 regular / 5 otherwise | both resolvers, `stage-a-cpu` |
| `stage-c-8-gpu-b200` | B200 pool | `["b200","<num_gpus>gpu"]` | one job per file | both resolvers, `stage-a-cpu` |
| `stage-c-4-gpu-b200` | same B200 pool | `["b200","<num_gpus>gpu"]` | one job per file | same plan, no 8-GPU stage barrier |
| `stage-c-2-gpu-mi350` | 2× MI355X | `["linux-miles-mi35x-spur"]` | 2 regular / 4 otherwise | both resolvers |
| `stage-c-4-gpu-mi350` | 4× MI355X | `["linux-miles-mi35x-spur"]` | 4 regular / 6 otherwise | both resolvers |
| `stage-c-8-gpu-mi350` | 8× MI355X | `["linux-miles-mi35x-spur"]` | 4 | both resolvers |

## What each stage does

The [shared section](#shared-by-cuda-and-rocm) is the single reference for rules used by both GPU backends. The [CUDA/CPU](#cuda-and-cpu-specifics) and [ROCm](#rocm-prnightlyweekly-mirror) sections describe their differences. CPU behavior is called out explicitly where it also uses a shared rule.

### Shared by CUDA and ROCm

**Workflow events.** `pr-test.yml` and `pr-test-rocm.yml` listen to both `pull_request.labeled` and `pull_request.unlabeled`. Adding or removing any PR label starts a new run in the same PR concurrency group, cancels the previous queued or running run, and resolves CI policy from the remaining labels. Independently dispatched `/rerun-test` runs are unaffected.

Both workflows treat `pull_request.closed` (including merges) as cancellation-only: each close event shares its workflow's PR concurrency group, cancels queued or running runs of that workflow, and starts no resolver or test jobs.

Both PR workflows expose reusable `workflow_call` entry points. Called runs group concurrency by `inputs.ref`, so redispatching the same release branch cancels the older run; literal `pr-test-` and `pr-test-rocm-` prefixes keep the workflows in separate concurrency groups.

**Image selection (`resolve-ci-image-<arch>` for CUDA, `resolve-ci-image` for ROCm).** Both workflows select a test image only when GPU tests are selected, using the same priority:

| Priority | Source |
|---|---|
| 1 | `image_tag` from the calling workflow |
| 2 | `ci_image_tag` from a manual run |
| 3 | The PR image matching the current build inputs |
| 4 | The backend's image-tag directive in the PR description |
| 5 | The backend's default image |

Specify a tag without a repository prefix. The backend-specific names are:

| Backend | PR tag | PR-body directive | Default tag | Repository |
|---|---|---|---|---|
| CUDA | `pr-<num>` | `ci-image-tag:` | `dev` | The running repository (`$GITHUB_REPOSITORY`; `radixark/miles` here) |
| ROCm | `pr-<num>-rocm10-mi35x` | `ci-rocm-image-tag:` | `miles-rocm10-mi35x` | `radixark/miles` for PR tags; `rocm/sgl-dev` otherwise |

**Image builds.** Each backend checks for image changes only when the run selects its GPU tests or the PR carries `rebuild-ci-image`. If a PR changes an image's inputs relative to the base, CI reuses a matching PR image or builds a new one. Any required build must succeed before image selection and GPU execution. See [Docker Build](/developer/ci/02-docker-build) for the input mapping and build process.

**Policy resolution (`resolve-ci-policy`).**

- `pull_request`, `schedule`, and `workflow_dispatch` only say how the workflow started; none itself implies a cadence or domain scope.
- Each Miles PR workflow passes trigger facts and, for PRs, the diff to `tests/ci/ci_policy.py`, which publishes the resolved policy and `skipped_stages` for `run_suite.py` and GPU job gates.
- A PR `nightly` label maps to nightly cadence.
- A scheduled run maps its exact UTC `github.event.schedule` cron: `0 15 * * 0-5` maps to nightly and `0 15 * * 6` maps to weekly; an unknown cron fails.
- A manual dispatch keeps regular cadence and has no PR labels. Both GPU workflows add `--match-all-labels` so an explicit manual operation runs the full regular GPU suites; CPU selection remains unchanged.
- A reusable `workflow_call` supplies an explicit cadence override because the called workflow inherits the caller's event name. Policy resolution deliberately runs the caller commit's `ci_policy.py`; only suite jobs check out the requested release ref.

A **nightly** policy selects every enabled tag except `long` and `ft-long`, admits both regular and `nightly=True` registrations, and disables fast-fail. **Weekly** and **release** select every enabled tag, admit both registration types, and disable fast-fail; release differs by never writing the rolling performance baseline. Regular cadence admits only regular registrations. All four cadences use the same stage inventory.

`run-ci-all` selects the full domain-tag set without changing cadence. `run-ci-image` selects every enabled tag except `long`, `ft-short`, and `ft-long`. If scope signals overlap, the precedence is `run-ci-all` > weekly/release full scope > nightly > `run-ci-image`. The resolved cadence and raw/synthetic labels are passed to `run_suite.py`, which computes one run policy (see [Labels](/developer/ci/01-label) for the subtraction semantics).

**PR GPU stage selection.** Before runner allocation, PR GPU stages are filtered by `runnable ∩ affected`: `runnable` reuses cadence/label selection, while `affected` maps changed registered tests to their suites. Known non-GPU paths affect none; unknown, missing, or malformed diffs affect all. `nightly`, `run-ci-all`, and `run-ci-image` add their runnable stages; `bypass-fastfail` does not. This path-based filter does not apply to CPU stages or scheduled/manual runs.

**Empty GPU jobs.** CUDA (except B200) and ROCm suite jobs first run their exact `--list-only` command on `ubuntu-latest`, including cadence, labels, disabled tests, partition arguments, and CUDA arch dispatch. `--github-output` publishes `has_tests` after partitioning. Only a successful plan with `has_tests=true` can allocate a GPU runner, which checks out the planned commit. Empty suites and shards skip GPU execution; planning errors fail the workflow. B200 file planning is described under [CUDA and CPU specifics](#cuda-and-cpu-specifics). `/rerun-test` instead uses its existing trusted resolver, which requires one enabled registration at the exact requested SHA.

**Execution contract.** Reusable GPU workflows own planning, checkout, environment setup, and test execution. Callers supply `execute_command`, `ref`, `runs_on`, and `container_image`. Planning uses only Python's standard library and runs before GPU container startup or dependency installation.

**Dependency refs (also CPU).** CUDA, CPU, and ROCm dependency refs resolve in this order: explicit dispatch input or PR-body directive, committed `release-lock.json`, then the moving `sglang-miles` / `miles-main` branch heads. A called release run therefore checks out its requested Miles `ref` and consumes the lockfile on that ref unless an explicit override exists.

During dependency setup, all three backends use the selected checkouts and install `radix_transfer_plugins` editable with `--no-deps`. The CUDA/CPU and ROCm sections below describe how each backend loads SGLang and Megatron-LM.

**Fork credentials.** PR runs test the merge commit, and GitHub withholds repository secrets from fork `pull_request` runs. Fork-head single-file reruns also withhold repository credentials; see [command identity](/developer/ci/05-command-identity) for that boundary.

**Sharding.** A stage with a `partition_id` matrix splits its tests across N shards; `run_suite.py` balances the shards by each test's `est_time`. Each shard is an independent job instance running the same `execute_command` with a different `--auto-partition-id`.

Shard sizing and concurrency policy live in [CI Maintenance](../../../.claude/rules/ci-maintenance.md).

### CUDA and CPU specifics

**PR admission.** In `pr-test.yml`, a PR event starts jobs only when the PR's base is the default branch or the PR carries a `run-ci*` label. Otherwise `resolve-ci-policy` is skipped, and because every other job needs it, the run starts nothing. A stacked PR therefore runs `PR Test` only while labeled; retargeting a PR to `main` is an `edited` event, which does not start a run.

**Dependencies / gating.** In `pr-test.yml`, `stage-a-cpu` requires only `resolve-ci-policy`; `stage-b-cpu` additionally waits for every `stage-a-cpu` shard to succeed and is skipped when `resolve-ci-policy` selects no `stage-b-cpu` tests (for example on an unlabeled PR). PR-image preparation requires the `stage-a-cpu` success/bypass gate and selected CUDA tests. Each CUDA stage declares its `image_arch` in `tests/ci/hardware.py::CUDA_STAGES` and requires `resolve-ci-image-<image_arch>`, which requires `docker-build-<image_arch>`. `stage-b-cpu` does not gate that chain and is skipped if `stage-a-cpu` fails, including with `bypass-fastfail` or broad cadences. Resolved nightly, weekly, or release cadence and the `bypass-fastfail` PR label admit the chain after an actual `stage-a-cpu` failure and make each suite continue after a test failure. This fast-fail exception does not itself select GPU tests or bypass policy or Docker/image failure; cadence and labels determine selection as described above. GPU tiers `b` and `c` describe cost/role and can run concurrently after these gates, subject to runner and GPU capacity. Scheduled, manual, and called release runs retain their existing image preparation.

**Release CI.** `release-branch-cut.yml` calls `pr-test.yml` for CPU and CUDA coverage and passes the prune-exempt `release-vX.Y.Z-ci` image tag recorded in `release-lock.json`.

**Runner selection and launch.** CUDA executes tests through `_run-ci.yml` on runners matching every label in `runs_on`. CPU uses `_run-cpu-ci.yml` on GitHub-hosted `ubuntu-latest`, with planning and execution in one job; when `plan_already_resolved` is false, it runs `--list-only --github-output` before freeing disk space, checking out dependencies, or installing packages, skipping empty shards without environment setup, and only frees preinstalled runner toolchains when root free space is below 40 GB.

**B200 allocation.** B200 jobs declare their minimum allocation with `register_cuda_ci(..., num_gpus=N)` for an integer N from 1 to 8. The existing B200 suite names select tests; they do not reserve a fixed partition. `_run-ci-b200.yml` plans all selected files at an exact commit and makes their jobs independently runnable. The host allocates any free GPU set, allowing `4+2+1+1`, `2+2+2+2`, or other fitting combinations. An 8-GPU job requires the whole host. Weekly uses the same scheduling rules. Jobs wait for GPUs after the runner accepts them, so their timeout includes admission waiting. The planner splits large selections into sequential batches whose summed file-timeout plus setup/cleanup allowances stay at or below 23 hours, below the 24-hour `GITHUB_TOKEN` lifetime and 5-day self-hosted job limit. Every job gets its batch's serial budget, covering even the last job to acquire GPUs; per-file execution timeouts remain unchanged. Files within each batch have no matrix parallelism cap.

The repository-wide `b200-oma` workflow queue keeps PRs and B200 file reruns in order. Host device locks remain necessary for old workflow revisions and hold through container cleanup. See `tests/ci/README.md` for deployment, cancellation, and orphan-container handling.

**Dependency boundary.** CUDA stages start from dependencies baked into `radixark/miles` and install `examples/multi_lora/requirements.txt` before `requirements.txt` through `tests/ci/reconcile_dependencies.py`. Installed CUDA 12/13 cuDNN versions are preserved with uv overrides; images without cuDNN use pip. Before each install, a dry-run gate rejects replacement of installed GPU runtimes or packages with at least 100 MiB of recorded installed files, including same-version reinstalls. These packages must be updated in the image.

CUDA and CPU load the selected SGLang/Megatron-LM sources through `PYTHONPATH` and verify import paths. Hosted CPU stages also install `requirements.txt` and the fully pinned `tests/ci/requirements-ci-cpu.txt`.

**Secrets.** CPU/CUDA stages call their reusable workflows with `secrets: inherit`.

**NVSHMEM on RoCE-only runners.** When the job container sees HCAs and all of them have link layer `Ethernet` (RoCE), `_run-ci.yml` sets `NVSHMEM_HCA_LIST` to a NIC name that matches nothing before any test command runs. NVSHMEM then skips its IB transports at init, and DeepEP keeps its intra-node traffic on NVLink. Runners with an InfiniBand HCA keep NVSHMEM's RDMA transports. The override relies on every CUDA stage being single-node; a multi-node stage on RoCE runners must not inherit it.

GitHub attaches every job container to a runner-created bridge network (`container.options` cannot set `--network`). On RoCE hosts NVSHMEM's IBRC/IBGDA connection setup then uses a GID that belongs to the host's network namespace: `ibv_modify_qp` fails with `ENODEV`, and DeepEP low-latency waits until the job timeout. The `aliyun` runner tag ([`manage-gh-runners`](https://github.com/radixark/miles/tree/main/.claude/skills/manage-gh-runners)) adds GIDs owned by the container, which Mooncake selects; NVSHMEM still fails.

**Arch dispatch.** `tests/ci/hardware.py::CUDA_STAGES` is the single source of truth for the CUDA taxonomy: each stage's GPU generation, GPU count, and runner labels. `CI_SUITES` and the default `/rerun-test` runner map derive from it. B200 overrides the runner width with the registration's `num_gpus`; a stage's `--suite` still names its generation.

A test normally runs at its home stage. A [dispatch label](/developer/ci/01-label) can send it to another generation instead, in which case the destination is the smallest stage on that generation with enough GPUs — a test declares its own GPU budget through `ray start --num-gpus` / `torchrun --nproc-per-node` rather than reading the devices it can see, so a wider stage simply leaves the surplus idle. Blackwell dispatch sends 2/4-GPU registrations to `stage-c-4-gpu-b200` and 8-GPU registrations to `stage-c-8-gpu-b200`.

Without a dispatch label nothing leaves its home stage, so scheduled and called runs are unaffected. Whatever stage actually executes a run is also the stage its performance baseline is keyed on, keeping the generations' numbers apart.

### ROCm PR/nightly/weekly mirror

The [shared CUDA/ROCm rules](#shared-by-cuda-and-rocm) apply. This section contains only ROCm-specific behavior:

- **Repository.** `pr-test-rocm.yml` runs only in `radixark/miles`, whose MI350 runners it uses: `resolve-ci-policy`, which every other job needs, requires that repository.
- **Spur execution.** `pr-test-rocm.yml` runs GPU suites through `_run-ci-rocm.yml`; see the [stage roster](#stage-roster) for suites and shard counts. `num_gpus` sets each Spur job's allocation on Miles' two 8-GPU MI355X nodes. A 1-, 2-, or 4-GPU job sees only its allocated GPUs and can share a node; an 8-GPU job reserves the whole node. Login-node runner processes limit concurrency. There are no CPU jobs, and ROCm is outside the `release-ci` gate.
- **No schedule for now.** `pr-test-rocm.yml` has no cron, so Spur's two runners stay with PRs. The external nightly below runs the nightly scope every day. The `long` ROCm registrations are `nightly=True`, so they run only on a PR labelled both `nightly` and `run-ci-long`, which runs the full weekly-sized scope, or one file at a time with `/rerun-test <file> --hw rocm`. Only PRs with the `nightly` label write the ROCm [metric baseline](03-metric-history-gate.md#rollout). Restore the cron when Spur gains runners or the external nightly stops covering that scope.
- **Dependencies.** ROCm installs the selected SGLang and Megatron-LM checkouts editable with `--no-deps` and prepares SGLang's pinned Rust toolchain.
- **Secrets.** ROCm explicitly passes `AMD_WANDB_API_KEY` and `AMD_NEON_DATABASE_URL`. The workflow maps the former to `WANDB_API_KEY` for W&B, and Spur forwards both into the container.
- **Test registration.** Tests opt in through `register_rocm_ci`; CUDA registrations do not select ROCm tests. Hardware-specific layouts belong in separate `test_amd_<name>.py` files.
- **Changed-file jobs.** PRs with `run-ci-amd`, `run-ci-all`, or `nightly` also run eligible changed ROCm files not covered by the selected suites, using the [single-file plan](01-label.md#manage-ci-from-pr-comments). `long` and `ft-long` require explicit `run-ci-long` and `run-ci-ft-long` labels. The planner reports skips and, if more than four files are eligible, lists them without launching file jobs. They use the suite's resolved image and share the ROCm `/rerun-test` queue. Without a usable diff, only suite selection applies.
- **External nightly.** SGLang's [ROCm 10 nightly](https://github.com/sgl-project/sglang/blob/main/.github/workflows/nightly-test-amd-miles-rocm10.yml) runs the same MI350 suites with `--cadence nightly`, still under their former `nightly-` names, which `run_suite.py` maps to them until that workflow moves, on its own MI35X runners and the newest Miles image.

## TPU smoke runs

`pr-test-tpu.yml` is outside the stage inventory: it runs whole `scripts/tpu` launchers rather than registered
test files. Each run is the launcher's `smoke` recipe (3 rollouts; the sglang-torchtpu engine and the torchtitan
trainer share one TPU v7x 2x2x1 host and sync weights over ICI) in a JobSet on sglang-torchtpu's GKE CI cluster,
queued through its Kueue PR-gate queue alongside sglang-torchtpu's PR jobs. A PR runs the Qwen3-4B smoke only while
it carries `run-ci-tpu`, again on every push; the nightly runs every launcher on `main`, and `workflow_dispatch` takes
a launcher list and any branch.
It runs only in `radixark/miles`, on every trigger: the cluster and the `dev-tpu` / `pr-<number>-tpu` images belong to that
repository. Fork PRs cannot run it either: the cluster's credentials come from GitHub's OIDC token through Workload
Identity Federation.

The image is the rolling `radixark/miles:dev-tpu`, or `radixark/miles:pr-<number>-tpu` for a PR that changes the TPU
image's inputs, chosen the way the CUDA suites choose theirs; the pod checks out the commit under test over the
image's `/root/miles`.
[`.github/tpu-ci/README.md`](../../../.github/tpu-ci/README.md) covers the cluster side.

## Assumptions

- Suite-to-stage mappings are maintained manually across `run_suite.py`, the Miles workflows, and the external MI350 nightly workflow.
- Runner placement assumes the live fleet actually carries the requested `runs_on` labels for each GPU class and count.
- `est_time` controls shard balancing, CPU [registration eligibility](01-label.md#registration-and-scan-scope), and timeouts for per-file runs.
