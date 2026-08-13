---
title: Labels
description: The three kinds of CI label — domain labels that gate tests, scope labels that broaden selection, and bypass-fastfail.
---
A label is a GitHub PR label that changes what CI runs or how it fails. Three kinds:

| Kind | Example | Effect |
|---|---|---|
| Domain label | `run-ci-megatron` | selects which tests run |
| Scope label | `run-ci-image` | run every enabled tag except `long`, `ft-short`, and `ft-long` |
| Cadence/scope label | `nightly` | select nightly cadence and every enabled tag except `long` and `ft-long`, with fast-fail disabled |
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

Domain labels live in `tests/ci/labels.py` (`KNOWN_LABELS`); a `labels=[...]` value outside it is a hard error. Current set: `megatron`, `model-scripts`, `sglang`, `fsdp`, `short`, `long`, `ckpt`, `lora`, `precision`, `ft-short`, `ft-long`, `weight-update`, `replay`, `qwen35`, `mooncake`.

To add one: add the entry to `KNOWN_LABELS`, then create the matching `run-ci-<key>` label on the PR. No workflow edit needed.

## Cadence eligibility

There are three CI cadences: `regular`, the ordinary mode; `nightly`, which admits `nightly=True` tests, broadens the default scope, and bypasses fast-fail; and `weekly`, which also admits `nightly=True` tests, selects every enabled tag, and bypasses fast-fail.

`register_*_ci(nightly=True)` means the test is eligible under nightly and weekly cadence, but not regular cadence. It does not create a separate suite inventory and does not replace domain-label filtering. A regular run selects regular registrations only; nightly and weekly select regular plus `nightly=True` registrations, then apply their own domain-label scopes. For example, a `nightly=True` test carrying only `ft-long` remains outside the nightly scope unless `run-ci-ft-long` or `run-ci-all` explicitly includes it, while weekly includes it automatically.

## Broad CI scopes

The workflow's `resolve-ci-policy` job forwards trigger-specific facts to `tests/ci/ci_policy.py`; that module adapts them into explicit cadence and label inputs, then its shared `resolve_policy` maps those inputs to one effective include-label set and fast-fail policy. `run_suite.py` consumes the same resolved-policy function and never derives policy from `schedule` or `workflow_dispatch` event names. A broad scope is just a large include set (every registered label minus the scope's subtractions).

| Scope | Explicit source | Runs | Subtracts | Fast-fail |
|---|---|---|---|---|
| all | `run-ci-all` label | every enabled tag | — | determined by cadence |
| weekly | exact weekly cron | every enabled tag | — | disabled on both levels |
| nightly | resolved nightly cadence from the PR label, exact nightly cron, or local `--nightly` | every enabled tag except `long` and `ft-long`, incl. `ft-short` | `long`, `ft-long` | disabled on both levels (within-stage only for local runs) |
| image | `run-ci-image` label | every enabled tag except `long` and FT tags | `long`, `ft-short`, `ft-long` | determined by cadence |

Rows are in precedence order: when scope signals overlap, the higher row wins (`run-ci-all` > weekly > nightly > `run-ci-image`, the branch order of `resolve_policy`). `run-ci-all` widens only the domain scope; regular cadence still does not admit `nightly=True` registrations.

The generic triggers carry no policy. All scheduled runs use UTC: nightly is identified by the exact cron `0 15 * * 0-5`, and weekly by `0 15 * * 6`. Saturday weekly replaces that day's nightly rather than starting alongside it. A manual dispatch uses regular cadence and no PR labels, so it receives only the ordinary always-on scope; its existing operation inputs do not imply all, nightly, or weekly.

A subtraction is not a per-test veto — it only stops that label from granting inclusion. A test carrying a subtracted label still runs when another of its labels is in the set, so a test that must stay outside the standard nightly scope must carry only labels that nightly subtracts.

A domain label explicitly requested on the PR wins over a scope subtraction: `run-ci-image` plus `run-ci-long` or `run-ci-ft-short`, and nightly plus `run-ci-long`, add the explicitly requested tests back rather than silently dropping the request.

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

A resolved nightly or weekly cadence bypasses fast-fail on both levels so a scheduled broad run surfaces every failure rather than stopping at the first. For nightly this applies equally whether the cadence came from the PR `nightly` label or the explicitly mapped nightly cron. Local `--nightly` applies the same nightly selection and within-stage behavior; cross-stage gating does not exist in a local invocation.

Like the scope labels, `bypass-fastfail` is a workflow-only input and is not in `KNOWN_LABELS`.

## Labels double as fork-PR CI approval

GitHub holds a first-time contributor's fork-PR CI at "Approve and run" after every push. Any maintainer-applied `run-ci*` label is already that human decision, so the `Approve Trusted CI` workflow (on `pull_request_target`) auto-approves the held runs while such a label is present. Removing the labels restores manual approval; the friction ends permanently once the contributor's first PR merges.
