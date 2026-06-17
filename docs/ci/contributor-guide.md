---
title: CI Contributor Guide
description: For community contributors — how to add a CI test and confirm it runs, how to tell an infra failure from your own, and how to report a machine issue or a flaky test.
---

# CI Contributor Guide

This guide is for contributors landing small features and fixes. It answers three things: how to add a test to CI and be sure it actually runs, how to read a red check and decide whether it's your change or the infrastructure, and how to report a machine issue or a flaky test. You never edit the CI workflow YAML to add a test — read on.

## Add a test to CI

CI selection is driven by a one-line declaration at the top of each test file, not by the workflow. To add a test you drop a `test_*.py` file in the right place and declare it; the runner discovers it automatically.

**CPU / pure-Python tests** go under `tests/fast/`. No declaration needed — every `test_*.py` there is auto-registered as a CPU test and runs on every PR.

**GPU tests** go under `tests/e2e/` (or `tests/fast-gpu/` for small single-file GPU tests) and need exactly one top-level declaration:

```python
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=600,                  # rough seconds the test takes; used to balance + time-out
    suite="stage-c-4-gpu-h200",    # which hardware bucket runs it (table below)
    labels=["megatron"],           # see "Will it run on my PR?"; use [] for always-on
)
```

Pick the `suite` by the hardware your test needs. The simplest reliable choice is to copy the `suite=` of an existing test most like yours.

| Suite | Runs on | Use for |
|---|---|---|
| (none — put file in `tests/fast/`) | CPU | pure-Python / CPU-only tests |
| `stage-b-2-gpu-h200` | 2× H200 | small 2-GPU tests |
| `stage-c-2-gpu-h200` | 2× H200 | 2-GPU tests |
| `stage-c-4-gpu-h200` | 4× H200 | 4-GPU tests |
| `stage-c-8-gpu-h100` | 8× H100 | 8-GPU tests |

### Verify it definitely runs

A test that isn't picked up fails silently — it just never appears, and CI stays green. Confirm pickup **before** you rely on it:

1. **Locally**, from the repo root, list the plan for your suite (no GPU needed):
   ```bash
   python3 tests/ci/run_suite.py --hw cuda --suite stage-c-4-gpu-h200 --match-all-labels --list-only
   # CPU: python3 tests/ci/run_suite.py --hw cpu --suite stage-a-cpu --match-all-labels --list-only
   ```
   Your file must appear under `Enabled N test(s)`. This command also validates registration across all tests — if any discovered file is missing its declaration, it errors here.
2. **On the PR**, open the matching stage job and read the **Resolve suite plan** step — it prints the same plan, so you can confirm your file is listed in the real environment.

If your file does **not** show up, check, in order:
- It's named `test_*.py` and lives under `tests/fast`, `tests/fast-gpu`, `tests/e2e`, or `tests/ci` (the only discovered roots).
- It has exactly one top-level `register_*_ci(...)` call (GPU tests only; not inside a function, not import-aliased).
- The `suite=` string is one of the suites in the table (a typo'd suite has no job and never runs).

### Will it run on my PR?

`labels` gates *which PRs* trigger your test:

- `labels=[]` (or omitted) → **always runs** on every PR.
- `labels=["megatron"]` → runs only when the PR carries the GitHub label **`run-ci-megatron`** (the `run-ci-` prefix is added on the PR side). This keeps the heavy GPU matrix off unrelated PRs.

So if your test is gated and you don't see it run, add the matching `run-ci-<label>` label to your PR. To force the full suite regardless of labels, a maintainer can add `run-ci-all`. Valid labels live in `tests/ci/labels.py`; using one outside that list is a hard error at collection time.

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

A maintainer maps the runner to its host and fixes the machine; you don't need runner access. For a fast sanity check before filing, you can ask in the Miles channel of the [SGLang Slack](https://slack.sglang.ai), but the **GitHub Issue is the tracked record**.

## Report a flaky test

A test is flaky when it fails non-deterministically — it passes on a re-run with no code change, usually on a numeric/accuracy/timing assertion. PR CI runs each test once, so a flake **will** fail your check; re-run the job to confirm the failure isn't your change. Report a test that flakes repeatedly so a maintainer can stabilize or quarantine it.

Open a **GitHub Issue** labeled **`flaky`** with:

- The **test file path** (e.g. `tests/e2e/megatron/test_x.py`).
- The **assertion that failed** (the `AssertionError` line).
- Run URLs for both a passing and a failing run, if you have them.

To unblock other PRs, a maintainer may temporarily set `disabled="<reason + issue link>"` on the test's `register_*_ci(...)` — that reports it as skipped (not deleted) until the flake is fixed. Don't disable a test in your own feature PR unless a maintainer asks.
