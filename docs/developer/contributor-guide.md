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
├── scripts/                  # launchers, one per recipe; scripts/models/ holds the architecture flags
├── tools/                    # checkpoint converters, quantizers, profilers
├── tests/                    # fast / fast-gpu / e2e / ci / manual (see Running CI)
├── docker/                   # Dockerfile, Dockerfile.rocm, build.py, patches
├── docs/                     # the source of this site, plus docs/developer/ci internals
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

**`.claude/rules/`** holds conventions. A file with a `paths:` front matter list applies
to any file matching it; a file without one applies everywhere. `general-code-style.md`
is the one described above, and `AGENTS.md` at the repo root points Codex at it too, so
both agents and humans review against one document. `ci-test-registration.md` holds the
rules for adding a CI test (see [Registering a test](#registering-a-test)), and
`ci-failure-triage.md` what to do when a CI check goes red.

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
# doc-dev: docs/developer/ci/02-docker-build.md
```

Current sentinels, so you know when you have walked into one:

| File | Governing document |
|---|---|
| `.github/workflows/pr-test.yml`, `pr-test-rocm.yml` | `docs/developer/ci/00-stage.md`, `docs/developer/ci/01-label.md` |
| `.github/workflows/bot-bump-miles-version.yml`, `bot-cherry-pick.yml`, `release-*.yml` | `docs/developer/ci/04-release.md` |
| `docker/build.py` | `docs/developer/ci/02-docker-build.md` |
| `tests/ci/metric_history/**` | `docs/developer/ci/03-metric-history-gate.md` |

Grep for `doc-dev:` before editing anything under `.github/workflows/` or `docker/`. A
change that lands the code and leaves the document stale is the failure mode this
convention exists to prevent.

## Running CI

### What a PR runs

Two things start automatically: the `pre-commit` workflow on every PR, and `PR Test`
(`.github/workflows/pr-test.yml`) on every PR based on `main`. A PR based on another branch,
such as a stacked PR, runs `PR Test` only with a `run-ci*` label (see
[Labels](/developer/ci/01-label)). `PR Test` resolves a policy and an image, runs the two
CPU stages, and then the GPU stages, which are gated on `stage-a-cpu` succeeding so a
formatting or import error does not burn GPU time. A PR that touches `docker/Dockerfile`,
`docker/build.py`, `docker/verify_transformer_engine.py`, `docker/patch/**` or
`requirements.txt` additionally builds the image first and runs every GPU suite inside it.

### Registering a test

Selection is declared in the test file, never in the workflow YAML. Every `test_*.py`
under `tests/fast/` is auto-registered as a CPU test in `stage-a-cpu`; a file under
`tests/fast-gpu`, `tests/e2e` or `tests/ci` declares a top-level `register_*_ci(...)` call,
or collection fails with `No CI registry found`.

Where the file goes, which stage runs it, how it is declared, which PRs run it, how
`est_time` is measured with `/rerun-test`, and when `disabled=` is allowed are rules,
written once in
[`.claude/rules/ci-test-registration.md`](https://github.com/radixark/miles/blob/main/.claude/rules/ci-test-registration.md).
Telling your failure from an infra one, and reporting either, is in
[`.claude/rules/ci-failure-triage.md`](https://github.com/radixark/miles/blob/main/.claude/rules/ci-failure-triage.md).

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
- [ ] New behavior has a test.
- [ ] Each new or moved CUDA test has its `CI timing:` line in the PR description, from a
  `/rerun-test` run (see `.claude/rules/ci-test-registration.md`).
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

Comment to claim an issue before you start. For an infra failure or a flake, file the
issue with the fields `.claude/rules/ci-failure-triage.md` lists (job URL, runner name,
suite, log line), so a maintainer can map it to a host.

## Where to ask

* **Quick questions:** the `#miles-rl` channel of the [SGLang Slack](https://slack.sglang.ai).
* **Design discussions:** a GitHub Discussion, or an Issue labeled `discussion`.
* **CI internals:** [Stage](/developer/ci/00-stage) (stages), [Labels](/developer/ci/01-label) (label
  semantics), [Docker build](/developer/ci/02-docker-build) (images), and
  [Metric history & regression gate](/developer/ci/03-metric-history-gate) (metric gate).
  The CI rules for contributors are `.claude/rules/ci-test-registration.md` and
  `.claude/rules/ci-failure-triage.md`.
