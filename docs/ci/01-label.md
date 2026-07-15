---
title: Labels
description: The three kinds of CI label — domain labels that gate tests, scope labels that broaden selection, and bypass-fastfail.
---

# Labels

A label is a GitHub PR label that changes what CI runs or how it fails. Three kinds:

| Kind | Example | Effect |
|---|---|---|
| Domain label | `run-ci-megatron` | selects which tests run |
| Scope label | `run-ci-image` | run every enabled non-FT tag |
| Cadence/scope label | `nightly` | select nightly cadence and every enabled tag except `ft-long`, with fast-fail disabled |
| Scope label | `run-ci-all` | run every enabled tag |
| Behavior label | `bypass-fastfail` | opt out of fast-fail; one run surfaces every failure |

Only domain labels are declared in `labels=[...]`; scope and behavior labels are workflow inputs resolved by `tests/ci/ci_policy.py`. The separate `nightly=True` registration field is a cadence gate described below.

## Domain labels: `register_*_ci(labels=...)` ↔ `run-ci-<x>`

A test declares its labels: `register_cuda_ci(..., labels=["megatron"])`. The PR trigger for `<x>` is the GitHub label `run-ci-<x>`. The workflow forwards canonical CI labels to `run_suite.py --labels`; Python strips the `run-ci-` prefix and intersects with each test's labels.

| Test declares | Runs when |
|---|---|
| `labels=[]` (or omitted) | every run whose cadence admits the test (always-on within that cadence) |
| `labels=["megatron"]` | PR has `run-ci-megatron` |
| `labels=["sglang"]` | PR has `run-ci-sglang` |
| `labels=["fsdp", "lora"]` | PR has `run-ci-fsdp` or `run-ci-lora` |

PR labels without the `run-ci-` prefix are ignored.

### The canonical label list

Domain labels live in `tests/ci/labels.py` (`KNOWN_LABELS`); a `labels=[...]` value outside it is a hard error. Current set: `megatron`, `model-scripts`, `sglang`, `fsdp`, `short`, `long`, `ckpt`, `lora`, `precision`, `ft-short`, `ft-long`, `weight-update`, `replay`, `qwen35`.

To add one: add the entry to `KNOWN_LABELS`, then create the matching `run-ci-<key>` label on the PR. No workflow edit needed.

## Cadence eligibility

There are two CI cadences: `regular`, the ordinary mode; and `nightly`, which admits `nightly=True` tests, broadens the default scope, and bypasses fast-fail.

`register_*_ci(nightly=True)` means the test is eligible only under nightly cadence. It does not create a separate suite inventory and does not replace domain-label filtering. A regular run selects regular registrations only; a nightly run selects regular plus nightly-only registrations, then applies the same suite and domain-label filters to both. For example, a nightly-only test carrying only `ft-long` remains outside the standard nightly scope unless `run-ci-ft-long` or `run-ci-all` explicitly includes it.

## Broad CI scopes

The workflow's `resolve-ci-policy` job forwards trigger-specific facts to `tests/ci/ci_policy.py`; that module adapts them into explicit cadence and label inputs, then its shared `resolve_policy` maps those inputs to one effective include-label set and fast-fail policy. `run_suite.py` consumes the same resolved-policy function and never derives policy from `schedule` or `workflow_dispatch` event names. A broad scope is just a large include set (every registered label minus the scope's subtractions).

| Scope | Explicit source | Runs | Subtracts | Fast-fail |
|---|---|---|---|---|
| all | `run-ci-all` label | every enabled tag | — | determined by cadence |
| nightly | resolved nightly cadence from the PR label, exact nightly cron, or local `--nightly` | every enabled tag incl. `ft-short` | `ft-long` | disabled on both levels (within-stage only for local runs) |
| image | `run-ci-image` label | every enabled non-FT tag | `ft-short`, `ft-long` | determined by cadence |

Rows are in precedence order: when scope signals overlap, the higher row wins (`run-ci-all` > nightly > `run-ci-image`, the branch order of `resolve_policy`). `run-ci-all` widens only the domain scope; without nightly cadence it does not admit nightly-only registrations.

The generic triggers carry no policy. The current nightly schedule is identified by the exact cron string `0 15 * * *`; adding a weekly schedule requires a distinct cadence mapping rather than another `event_name == "schedule"` branch. A manual dispatch uses regular cadence and no PR labels, so it receives only the ordinary always-on scope; its existing operation inputs do not imply all or nightly.

A subtraction is not a per-test veto — it only stops that label from granting inclusion. A test carrying a subtracted label still runs when another of its labels is in the set, so a test that must never run at nightly must carry only FT labels.

A domain label explicitly requested on the PR wins over a scope subtraction: `run-ci-image` plus `run-ci-ft-short` runs the image scope *and* the ft-short tests, rather than silently dropping the explicit request.

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

- Cross-stage: each GPU stage consumes the shared `bypass_fastfail` policy output, so GPU stages run even after `stage-a-cpu` fails.
- Within-stage: `run_suite.py` derives continue-on-error from the same resolved policy (drops `pytest -x`; sets `continue_on_error=True` for CUDA). The stage still ends red — it changes coverage, not the verdict.

A resolved nightly cadence bypasses fast-fail on both levels because a nightly is meant to exercise every eligible test except `ft-long` and surface every failure (one datapoint per test), not stop at the first. This applies equally whether the cadence came from the PR `nightly` label or the explicitly mapped nightly cron. Local `--nightly` applies the same selection and within-stage behavior; cross-stage gating does not exist in a local invocation.

Like the scope labels, `bypass-fastfail` is a workflow-only input and is not in `KNOWN_LABELS`.
