---
title: Labels
description: The three kinds of CI label — domain labels that gate tests, meta-labels that run everything, and bypass-fastfail.
---

# Labels

A label is a GitHub PR label that changes what CI runs or how it fails. Three kinds:

| Kind | Example | Effect |
|---|---|---|
| Domain label | `run-ci-megatron` | selects which tests run |
| Meta-label | `run-ci-image`, `run-ci-all` | run the full suite, ignoring domain labels |
| Behavior label | `bypass-fastfail` | opt out of fast-fail; one run surfaces every failure |

Only domain labels are declared by tests; the other two are workflow switches in `pr-test.yml`.

## Domain labels: `register_*_ci(labels=...)` ↔ `run-ci-<x>`

A test declares its labels: `register_cuda_ci(..., labels=["megatron"])`. The PR trigger for `<x>` is the GitHub label `run-ci-<x>`. The workflow passes every PR label to `run_suite.py --labels`; Python strips the `run-ci-` prefix and intersects with each test's labels.

| Test declares | Runs when |
|---|---|
| `labels=[]` (or omitted) | every PR (always-on) |
| `labels=["megatron"]` | PR has `run-ci-megatron` |
| `labels=["sglang"]` | PR has `run-ci-sglang` |
| `labels=["fsdp", "lora"]` | PR has `run-ci-fsdp` or `run-ci-lora` |

PR labels without the `run-ci-` prefix are ignored.

### The canonical label list

Domain labels live in `tests/ci/labels.py` (`KNOWN_LABELS`); a `labels=[...]` value outside it is a hard error. Current set: `megatron`, `model-scripts`, `sglang`, `fsdp`, `short`, `long`, `ckpt`, `lora`, `precision`, `weight-update`, `replay`.

To add one: add the entry to `KNOWN_LABELS`, then create the matching `run-ci-<key>` label on the PR. No workflow edit needed.

## Meta-labels: run everything

`run-ci-image` and `run-ci-all` are the same switch: both add `--match-all-labels` to run every enabled test in the suite, ignoring domain labels. Neither is in `KNOWN_LABELS`.

A manual `workflow_dispatch` run gets `--match-all-labels` too, because it has no PR labels to filter on.

## Registration and scan scope

Labels are optional; registration is not. The runner scans `tests/fast`, `tests/fast-gpu`, `tests/e2e`, `tests/ci` recursively for `test_*.py`. Every file must resolve to a registration or collection fails:

- A file outside `tests/fast/` with no `register_*_ci()` call → `No CI registry found`.
- A `labels=[...]` value not in `KNOWN_LABELS` → `unknown labels [...]`.

## `tests/fast/` auto-registers as CPU

Each `test_*.py` under `tests/fast/` is auto-registered as a CPU test (backend CPU, suite `stage-a-cpu`, `labels=[]`) with no `register_*_ci()` call, and runs on the GitHub-hosted `ubuntu-latest` runner. Here "CPU" is the hardware backend, not a label. A `register_cuda_ci()` under `tests/fast/` is a hard error — move it to `tests/fast-gpu/`.

## `bypass-fastfail`: opt out of fast-fail

By default CI fails fast on two levels:

- Cross-stage: GPU stages run only when `stage-a-cpu` succeeds — the `if` requires `needs.stage-a-cpu.result == 'success'`.
- Within-stage: each suite stops at the first failure (`pytest -x` for CPU; `run_unittest_files` breaks on the first failing file for CUDA).

The `bypass-fastfail` PR label turns both off so one run surfaces every failure:

- Cross-stage: each GPU stage's check becomes `(needs.stage-a-cpu.result == 'success' || (needs.stage-a-cpu.result == 'failure' && contains(..., 'bypass-fastfail')))`, so GPU stages run even after `stage-a-cpu` fails.
- Within-stage: each stage adds `--continue-on-error` (drops `pytest -x`; sets `continue_on_error=True` for CUDA). The stage still ends red — it changes coverage, not the verdict.

Like the meta-labels, `bypass-fastfail` is matched directly in `pr-test.yml` and is not in `KNOWN_LABELS`.
