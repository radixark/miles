---
title: Contributing
description: Repository layout, the local loop, what enforces code style, what lives in .claude, and how to drive CI from a PR.
---
Miles is open source under the LICENSE file in the repo. Contributions of every size are
welcome: bug reports, doc fixes, new model recipes, full features.

## Repository layout

```text
miles/
├── train.py                  # synchronous entry point
├── train_async.py            # fully-async entry point
├── train_multi_lora_async.py # multi-LoRA async entry point
├── miles/                    # the package
│   ├── backends/
│   │   ├── megatron_utils/   # Megatron actor, weight sync, checkpointing, fp32 markers
│   │   ├── fsdp_utils/       # FSDP2 actor, per-arch adaptations, MoE kernels
│   │   ├── sglang_utils/     # SGLang engine + argument glue
│   │   └── training_utils/   # loss / GRPO / PPO / GSPO / REINFORCE++, shared ParallelState
│   ├── ray/                  # Ray actors, placement groups, the train and rollout groups
│   ├── rollout/              # rollout functions, data source, filters, fully-async buffer
│   ├── router/               # Miles Router (FastAPI proxy in router.py)
│   ├── dashboard/            # the run dashboard (collector + backend)
│   ├── true_on_policy/       # true-on-policy contracts and model profiles
│   └── utils/                # arguments.py, async / IO / distributed helpers, audit utils
├── miles_plugins/            # opt-in plugins, imported by name from flags
│   ├── models/               # per-architecture Megatron specs and HF wrappers
│   ├── mbridge/              # per-architecture weight bridges
│   ├── megatron_bridge/      # megatron.bridge shims
│   └── optimizers/           # optimizer plugins (NVMe streaming store)
├── scripts/                  # launchers, one per recipe; scripts/models/ holds MODEL_ARGS
├── tools/                    # checkpoint converters, quantizers, profilers
├── tests/                    # fast / fast-gpu / e2e / ci / manual (see Running CI)
├── docker/                   # Dockerfile, Dockerfile.rocm, build.py, patches
├── docs/                     # the source of this site, plus docs/ci internals
└── .claude/                  # rules and skills (see What lives in .claude)
```

## The local loop

```bash
git remote add me git@github.com:<your_user>/miles.git
git checkout -b feat/awesome

pip install -e . --no-deps       # editable install, deps come from the image
pre-commit install               # optional, runs the hooks on every commit

pytest tests/fast                # CPU suite, no GPU needed
pre-commit run --all-files       # what the pre-commit CI job runs

git commit -m "feat(rollout): add partial-rollout buffer"
git push me feat/awesome && gh pr create
```

## Code style

Formatting is not a matter of taste here, it is a hook. `.pre-commit-config.yaml` is the
enforcement, the `pre-commit` workflow runs `pre-commit run --all-files` on every PR, and
you can reproduce it exactly with the same command locally.

| Hook | What it enforces |
|---|---|
| `ruff-check --fix` | Pycodestyle errors, Pyflakes, bugbear, pyupgrade. `E402` and `E501` are ignored on purpose |
| `autoflake` | Removes unused imports in place |
| `isort` | Import order, `--profile=black`, first-party is `miles` and `miles_plugins` |
| `black` | Formatting at **line length 119** (`[tool.black]` in `pyproject.toml`) |
| `check-yaml`, `check-case-conflict`, `detect-private-key`, `check-added-large-files` | Hygiene; files cap at 1000 KB |
| `requirements-txt-fixer` | Keeps `requirements.txt` sorted |

Three hooks are Miles-specific bans, each pointing at the API you should use instead:

| Ban | Use instead | Why |
|---|---|---|
| `mpu.get_*` | `get_parallel_state()` from `miles.backends.training_utils.parallel` | The two backends share one `ParallelState`; reading Megatron's `mpu` directly does not work under FSDP |
| `AutoConfig.from_pretrained` / `AutoTokenizer.from_pretrained` | `load_hf_config` / `load_tokenizer` from `miles.utils` | The wrappers centralize trust, caching, and the multi-node file-system race |
| `huggingface-cli` | `hf` | The old CLI is deprecated upstream |

If a commit legitimately needs an exception, the hooks carry `exclude` patterns; extend
those in the same PR rather than disabling a hook.

Beyond formatting, the conventions a reviewer will hold you to live in
[`.claude/rules/general-code-style.md`](https://github.com/radixark/miles/blob/main/.claude/rules/general-code-style.md):
prefer stateless and immutable, keep functions under roughly 100 lines and files under
roughly 1000, initialize derived values once, keep imports at the top, use absolute
imports, prefer keyword arguments where they add clarity. It applies to `miles/**/*.py`,
`scripts/**/*.py`, `tools/**/*.py`, `train.py` and `train_async.py`.

## What lives in `.claude`

The `.claude` directory is how the repo hands its conventions to coding agents, and it is
worth reading even if you never run one, because it is where several rules are written
down exactly once.

**`.claude/rules/`** holds path-scoped conventions. Each file carries a `paths:` front
matter list, and the rule applies to any file matching it. `general-code-style.md` is the
one described above. `AGENTS.md` at the repo root points Codex at the same file, so both
agents and humans review against one document.

**`.claude/skills/`** holds procedures, one directory per skill with a `SKILL.md`. They
are workflows rather than style rules:

| Skill | What it is for |
|---|---|
| `doc-dev` | Keeping a file and its governing document in sync (see below) |
| `ci-fetch-log` | Pulling complete GitHub Actions logs and diagnosing a failed run from saved evidence |
| `ci-e2e-time-tune` | Recalibrating `register_cuda_ci(est_time=...)` from real run times |
| `mechanical-refactor-verify` | Reviewing a file split or move by requiring a reproducible transform script |
| `setup-ci-host`, `manage-gh-runners` | Provisioning a CI host and its self-hosted runners |

### The `doc-dev` sentinel

Some files are bound to a document. A `# doc-dev:` line in a file's own comment syntax is
the opt-in: bare, it binds the file's own header block; with a repo-relative path, it also
binds that central document. Editing such a file means updating its documentation in the
same change, and editing the document means finding the files that name it.

```python
# doc-dev: docs/ci/02-docker-build.md
```

Current sentinels, so you know when you have walked into one:

| File | Governing document |
|---|---|
| `.github/workflows/pr-test.yml`, `pr-test-rocm.yml` | `docs/ci/00-stage.md`, `docs/ci/01-label.md` |
| `docker/build.py` | `docs/ci/02-docker-build.md` |
| `tests/ci/metric_history/**` | `docs/ci/03-metric-history-gate.md` |

Grep for `doc-dev:` before editing anything under `.github/workflows/` or `docker/`. A
change that lands the code and leaves the document stale is the failure mode this
convention exists to prevent.

## Running CI

### What a PR runs

Two things start automatically on every PR: the `pre-commit` workflow, and `PR Test`
(`.github/workflows/pr-test.yml`). `PR Test` resolves a policy and an image, runs the two
CPU stages, and then the GPU stages, which are gated on `stage-a-cpu` succeeding so a
formatting or import error does not burn GPU time. A PR that touches `docker/Dockerfile`,
`docker/build.py`, `docker/verify_transformer_engine.py`, `docker/patch/**` or
`requirements.txt` additionally builds the image first and runs every GPU suite inside it.

### Registering a test

Selection is declared in the test file, never in the workflow YAML.

- **CPU tests go in `tests/fast/`.** Every `test_*.py` there is auto-registered as a CPU
  test in `stage-a-cpu` with no labels, and runs on every PR. A `register_cuda_ci` under
  `tests/fast/` is a hard error; move the file to `tests/fast-gpu/`.
- **Everywhere else, register explicitly.** One top-level call per file:

```python
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=600,                 # rough seconds; balances shards and sets the per-file timeout
    suite="stage-c-4-gpu-h200",   # the hardware bucket that runs it
    labels=["megatron"],          # [] or omitted means always-on
)
```

`register_cpu_ci`, `register_cuda_ci` and `register_rocm_ci` share that signature, plus
`nightly=True` (nightly and weekly cadence only) and `disabled="<reason + issue link>"` (reported as
skipped rather than deleted). The calls are parsed from the AST, so they must be
top-level, literal, and unaliased.

The runner scans `tests/fast`, `tests/fast-gpu`, `tests/e2e` and `tests/ci` for
`test_*.py`, and a file outside `tests/fast/` with no registration fails collection with
`No CI registry found`. Suites are `stage-<tier>-<gpus>-<hw>`; pick the one an existing
test like yours uses, because a typo'd suite has no job and silently never runs.

### Verify it actually runs

```bash
# list the plan for a suite, no GPU needed
python3 tests/ci/run_suite.py --hw cuda --suite stage-c-4-gpu-h200 --match-all-labels --list-only
python3 tests/ci/run_suite.py --hw cpu  --suite stage-a-cpu        --match-all-labels --list-only
```

Your file must appear under `Enabled N test(s)`. The command also validates registration
across every discovered test, so it fails here if any file is missing its declaration. Add
`--nightly` when checking a `nightly=True` registration. On the PR, the matching stage job
prints the same plan in its **Resolve suite plan** step.

### Labels

Labels are how a PR opts into the expensive matrix. A test's `labels=["megatron"]` is
triggered by the GitHub label `run-ci-megatron`: the workflow forwards the labels, Python
strips the `run-ci-` prefix and intersects with each test's list. The canonical set lives
in `tests/ci/labels.py`, and a value outside it is a collection-time error.

| Label | Effect |
|---|---|
| `run-ci-<x>` | Selects tests declaring `<x>` |
| `run-ci-all` | Every enabled tag |
| `nightly` | Nightly cadence: admits `nightly=True` tests, every tag except `long` and `ft-long`, fast-fail off |
| `run-ci-image` | Every tag except `long`, `ft-short`, `ft-long`; for validating an image bump |
| `bypass-fastfail` | Run GPU stages even if `stage-a-cpu` failed, and let each suite continue past the first failure |

If your fork PR sits waiting for approval, that is GitHub holding first-time contributor
runs; any maintainer-applied `run-ci-*` label doubles as the approval.

## PR-description CI tags

Three directives are read out of the PR description itself, one per line, matched at the
start of a line:

| Line in the PR description | Effect |
|---|---|
| `ci-image-tag: <tag>` | Run the GPU suites on `radixark/miles:<tag>` instead of `dev`. Must be a bare tag, not a full image reference |
| `ci-sglang-pr: <ref>` | Check the SGLang tree out at `<ref>` instead of the default `sglang-miles` branch |
| `ci-megatron-pr: <ref>` | Check Megatron-LM out at `<ref>` instead of the default `miles-main` branch |

For the two ref directives, `<ref>` is a branch or commit, and the shorthand `#1234`
resolves to `refs/pull/1234/head`, which is how you test against an unmerged SGLang or
Megatron-LM PR:

```text
ci-sglang-pr: #12345
ci-megatron-pr: my-fix-branch
ci-image-tag: dev-202608100600
```

Precedence, when several sources disagree: a `workflow_dispatch` input wins, then the
PR-description line, then the default. `ci-image-tag:` has one more rule: on a PR that
built its own image, the fresh `pr-<number>` tag outranks the directive, so a docker PR
always tests what it just built.

## PR conventions

Commit subjects follow conventional commits, under 70 characters, and the body explains
*why*:

```text
feat(rollout): add partial-rollout buffer
fix(megatron): correct fp32 marker on Qwen3.5 A_log
docs: clarify FP8 rationale for MoE
```

Before marking a PR ready for review:

- [ ] `pre-commit run --all-files` is clean.
- [ ] `pytest tests/fast` passes, plus `tests/fast-gpu` if you have a GPU.
- [ ] New behavior has a test, registered where CI will find it (verified with
  `--list-only`).
- [ ] A new flag appears in [CLI Reference](/user-guide/cli-reference), and
  `python3 train.py --help` still parses.
- [ ] A change to a `doc-dev:` governed file updates its document in the same PR.
- [ ] The PR description carries the CI directives your change needs.

Commenting `/run-lint` on a PR runs the hooks in CI and pushes the autofixes back to your
branch, which is the quick way out of a red `pre-commit` job.

## Issue triage

| Label | Meaning |
|---|---|
| `good first issue` | Self-contained, no system knowledge needed |
| `help wanted` | Community PRs welcome |
| `bug` | Reproducible breakage |
| `enhancement` | Feature request |
| `discussion` | Design conversation, not yet a task |
| `needs-repro` | Not reproducible yet, please add a minimal example |
| `ci-infra` | A CI machine or runner problem, not a code failure |
| `flaky` | A test that fails non-deterministically |

Comment to claim an issue before you start. For an infra failure or a flake, file the
issue with the job URL, the runner name, the suite, and the log line, so a maintainer can
map it to a host.

## Where to ask

* **Quick questions:** the Miles channel of the [SGLang Slack](https://slack.sglang.ai).
* **Design discussions:** a GitHub Discussion, or an Issue labeled `discussion`.
* **CI internals:** [Stage](/ci/00-stage) (stages), [Labels](/ci/01-label) (label
  semantics), [Docker build](/ci/02-docker-build) (images),
  [Metric history & regression gate](/ci/03-metric-history-gate) (metric gate), and the
  [CI Contributor Guide](/ci/contributor-guide) for the long-form version of this section.
