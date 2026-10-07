---
title: CI Contributor Guide
description: For community contributors — add a CI test and confirm it runs, tell an infra failure from your own, and report a machine issue or a flaky test.
---
This guide is for contributors landing small features and fixes. It answers three things: how to add a test to CI and be sure it actually runs, how to read a red check and decide whether it's your change or the infrastructure, and how to report a machine issue or a flaky test. You never edit the CI workflow YAML to add a test — read on.

## Add a test to CI

CI selection is driven by a one-line declaration at the top of each test file, not by the workflow. To add a test you drop a `test_*.py` file in the right place and declare it; the runner discovers it automatically.

**CPU / pure-Python tests** in `tests/fast/` need no declaration — every `test_*.py` there is auto-registered as a CPU test and runs in every `PR Test` run. **GPU tests** go under `tests/e2e/` (or `tests/fast-gpu/` for short GPU checks) and declare a top-level `register_cuda_ci(...)`.

The rules for adding a test — where the file goes, which stage runs it, how it is declared, how `est_time` is measured, and when `disabled=` is allowed — are written once, in [`.claude/rules/ci-test-registration.md`](https://github.com/radixark/miles/blob/main/.claude/rules/ci-test-registration.md), which coding agents load as well. In short: pick the cheapest stage that can run the test, run it once with `/rerun-test` before the PR merges, and set `est_time` from the measured **Run tests** time. The rest of this section covers the mechanics around those rules.

### Verify it definitely runs

A test that isn't picked up fails silently — it just never appears, and CI stays green. Confirm pickup **before** you rely on it:

1. **Locally**, from the repo root, list the plan for your suite (no GPU needed):
   ```bash
   python3 tests/ci/run_suite.py --hw cuda --suite stage-c-4-gpu-h200 --match-all-labels --list-only
   # CPU: python3 tests/ci/run_suite.py --hw cpu --suite stage-a-cpu --match-all-labels --list-only
   ```
   Your file must appear under `Enabled N test(s)`. Add `--nightly` when verifying a `nightly=True` registration. This command also validates registration across all tests — if any discovered file is missing its declaration, it errors here.
2. **On the PR**, open the matching stage job and read the **Resolve suite plan** step — it prints the same plan, so you can confirm your file is listed in the real environment.

If your file does **not** show up, check, in order:
- It's named `test_*.py` and lives under `tests/fast`, `tests/fast-gpu`, `tests/e2e`, or `tests/ci` (the only discovered roots).
- It has a top-level `register_*_ci(...)` call (GPU tests only; not inside a function, not import-aliased).
- The `suite=` string is a real stage: a CUDA suite in `tests/ci/hardware.py`'s `CUDA_STAGES`, or `stage-a-cpu` / `stage-b-cpu` for CPU (a typo'd suite has no job and never runs).

`/rerun-test <test-file>`, which the rules use to measure a new test, is described in [Manage CI from PR comments](/developer/ci/01-label#manage-ci-from-pr-comments).

### Will it run on my PR?

`labels` gates *which PRs* trigger your test within its eligible cadence. GPU registrations require at least one domain label:
- `labels=["megatron"]` → runs only when the PR carries the GitHub label **`run-ci-megatron`** (the `run-ci-` prefix is added on the PR side). This keeps the heavy GPU matrix off unrelated PRs.

Cadence is independent of labels: `nightly=True` excludes a registration from regular cadence, while nightly, weekly, and release runs include both ordinary and `nightly=True` registrations.

So if your test is gated and you don't see it run, add the matching `run-ci-<label>` label to your PR. To force the full suite regardless of labels, a maintainer can add `run-ci-all`. Valid labels live in `tests/ci/labels.py`; using one outside that list is a hard error at collection time.

### First PR from a fork: CI waits for approval

Until your first PR merges, GitHub holds every CI run of your fork PR behind a maintainer's "Approve and run" — after **every** push. There is no separate trust flag: any `run-ci-*` label a maintainer adds (the same labels that select tests) also approves the held runs, for that push and each one after. Removing the labels restores manual approval.

## When CI fails: yours or the infra?

Open the failing job and read the log first. Most failures fall cleanly into one of two buckets.

**Likely your change** — fix it locally before re-running:

| Signal | What it means |
|---|---|
| `ImportError` / `ModuleNotFoundError` / `SyntaxError` / `NameError` | your code doesn't load |
| `pre-commit` job red | formatting / lint; run `pre-commit run --all-files` locally |
| your new test's own assertion fails | reproduce with `python3 tests/e2e/.../test_yours.py` |

**Likely infrastructure, not your fault** — re-run the job once first (transient issues clear on retry); if it reproduces, report it:

| Signal in the log | Cause |
|---|---|
| Job stuck `Queued`, never starts | runner pool busy or a runner is offline |
| `nvidia-smi did not become ready after 120s` / CUDA error 802 | GPU subsystem not ready on the runner |
| `ENOSPC` / model or dataset download failure | runner disk full |
| Job needed N GPUs but ran with fewer | scheduling landed it on a smaller runner |
| A test **unrelated to your change** fails on an accuracy/score/latency assertion, then passes on re-run | flaky test (see below) |

Rule of thumb: if the failure is in code or tests your PR didn't touch, and a re-run behaves differently, it's infra or flakiness — not yours.

## Report a machine / infra issue

When a re-run still shows an infra signal from the table above, open a **GitHub Issue** labeled **`ci-infra`**. Include:

- The failing **job URL** (the Actions page for that job).
- The **runner name** (`runner_name`, shown at the top of the job log).
- The **suite / stage** (e.g. `stage-c-4-gpu-h200`) and the step that failed.
- A short **log snippet** of the error (the infra signal line).
- What you already tried (e.g. "re-ran twice, same `ENOSPC`").

A maintainer maps the runner to its host and fixes the machine; you don't need runner access. For a fast sanity check before filing, you can ask in the `#miles-rl` channel of the [SGLang Slack](https://slack.sglang.ai), but the **GitHub Issue is the tracked record**.

## Report a flaky test

A test is flaky when it fails non-deterministically — it passes on a re-run with no code change, usually on a numeric/accuracy/timing assertion. PR CI runs each test once, so a flake **will** fail your check; re-run the job to confirm the failure isn't your change. Report a test that flakes repeatedly so a maintainer can stabilize or quarantine it.

Open a **GitHub Issue** labeled **`flaky`** with:

- The **test file path** (e.g. `tests/e2e/megatron/test_x.py`).
- The **assertion that failed** (the `AssertionError` line).
- Run URLs for both a passing and a failing run, if you have them.

To unblock other PRs, a maintainer may temporarily set `disabled="<reason> (#<issue>)"` on the test's `register_*_ci(...)` — that reports it as skipped (not deleted) until the flake is fixed. The `disabled=` rules, including the required issue link, are in [`.claude/rules/ci-test-registration.md`](https://github.com/radixark/miles/blob/main/.claude/rules/ci-test-registration.md#disabled).
