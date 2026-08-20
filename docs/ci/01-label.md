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
| CPU `labels=[]` (or omitted) | every run whose cadence admits the test (always-on within that cadence) |
| `labels=["megatron"]` | PR has `run-ci-megatron` |
| `labels=["sglang"]` | PR has `run-ci-sglang` |
| `labels=["fsdp", "lora"]` | PR has `run-ci-fsdp` or `run-ci-lora` |

Bare domain labels without the `run-ci-` prefix do not select domains. The workflow-only `nightly` and `bypass-fastfail` labels are handled separately as cadence and behavior inputs.

CUDA and ROCm registrations must declare at least one domain label. GPU runners are scarce and expensive, so an always-on GPU test would defeat the purpose of selecting only the GPU coverage a PR requests.

### The canonical label list

Domain labels live in `tests/ci/labels.py` (`KNOWN_LABELS`); a `labels=[...]` value outside it is a hard error. Current set: `megatron`, `model-scripts`, `sglang`, `fsdp`, `short`, `long`, `ckpt`, `lora`, `eval`, `precision`, `ft-short`, `ft-long`, `weight-update`, `fully-async`, `replay`, `qwen35`, `mooncake`, `miles-plugin`, `amd`.

To add one: add the entry to `KNOWN_LABELS`, then create the matching `run-ci-<key>` repository label in GitHub. No workflow edit is needed. To expose it through PR comments, also add that exact label to `commands.add_label.allowed_labels` in `.github/workflows/policies/comment-command-access.json`.

## Manage CI from PR comments

The PR-comment entrypoint is a command gateway rather than a label handler. Each recognized comment becomes a typed request, and a code-defined static registry selects its fixed handler, policy key, and token capability. The JSON policy controls only access groups and per-command resource allowlists. The registry implements exact-label additions plus `/clear-labels`, `/rerun-failed-ci`, and `/rerun-test`.

After the comment gateway is enabled, post `/<label>` as the entire comment on an open PR to append that exact label. The label must be a supported `run-ci-*` label or `bypass-fastfail` and must be listed in `.github/workflows/policies/comment-command-access.json`; for example, `/run-ci-short` appends only `run-ci-short`, while `/bypass-fastfail` appends only `bypass-fastfail`.

A label command permits leading and trailing whitespace only; it cannot include arguments, prose, or a second command. Only `/rerun-test` takes an argument, and that argument must be a single test-file path. If the label is already present, the request succeeds as a no-op and does not emit another `labeled` event or rerun CI.

`.github/workflows/policies/comment-command-access.json` is an exact, default-deny ACL. Its `commands.add_label.allowed_labels` array controls which exact labels can be added through comments; adding a `KNOWN_LABELS` entry does not expose it automatically. Other command entries select an access group but cannot select a handler, token capability, workflow, API endpoint, or shell command.

The `add_label_access` group controls every command that adds a label. A caller belongs to this group when either their live legacy permission on `radixark/miles` appears in `repository_permissions` or their stable numeric GitHub user ID appears in `user_ids`.

The checked-in policy accepts `write` and `admin` and has no explicit user IDs. Workflow owners can grant a contributor label-command access by adding only that numeric ID to the JSON policy, without granting repository write access. GitHub reports the `maintain` role as legacy `write`; custom roles follow their base repository access.

The `repo_write_access` group restricts `/clear-labels`, `/rerun-failed-ci`, and `/rerun-test <test-file>` to live `write` or `admin` permission. Only `add_label_access` can contain explicit `user_ids`; those IDs do not grant any non-label operation.

Unrecognized comments exit after trusted parsing with capability `none`; they do not load the access policy, call the GitHub API, or mint an App token. A malformed comment containing one of the recognized command markers still fails instead of being treated as an unrelated comment.

The gateway controls only the delegated comment path; it does not restrict users' existing GitHub UI/API label permissions and does not offer commands that add `run-ci-all`, `nightly`, or an arbitrary label absent from the policy.

Post `/clear-labels` as the entire comment to remove every current label whose name starts with `run-ci`, plus `nightly` and `bypass-fastfail`. All other PR labels are preserved. This stops stale CI scope, cadence, fast-fail, and fork-approval choices from carrying into later pushes. It neither suppresses the ordinary always-on CI triggered by `synchronize` nor cancels a run that has already started.

Post `/rerun-failed-ci` as the entire comment to request failed-job reruns for the current open PR head. The handler considers only the latest run of each allowlisted PR workflow: `pre-commit.yml`, `pr-test.yml`, and `pr-test-rocm.yml`. A latest run is rerun only when it belongs to this PR and exact head SHA and has completed with conclusion `failure`.

Successful, skipped, cancelled, queued, or in-progress runs are not rerun, and a newer run prevents an older failure of the same workflow from being revived. The command does not touch CodeQL, `pull_request_target`, cleanup, scheduled, manually dispatched, or other control-plane workflows.

GitHub can omit `pull_requests` from fork workflow-run payloads. For a fork, the handler therefore also requires an all-state lookup by exact head owner and ref to identify only the current open PR, then binds each run to the same head ref, SHA, and repository ID. An absent, reused, or otherwise ambiguous fork head fails without a rerun.

Post `/rerun-test <test-file>` as the entire comment to run one registered test file on the PR's current head, e.g. `/rerun-test tests/e2e/precision/test_hf_attention_cp_relayout.py`. Despite the name, this dispatches a fresh workflow and does not require the test to have run previously. The handler accepts only a repo-relative path under the registry scan roots (`tests/e2e`, `tests/fast`, `tests/fast-gpu`, `tests/ci`), then dispatches the fixed default-branch `.github/workflows/run-ci-file.yml` with the PR number, exact head SHA, and file path as inputs.

After GitHub confirms the workflow dispatch, the App reacts to the original command with 👍 and posts a separate PR comment containing the exact Actions run link:

```text
Started `/rerun-test tests/e2e/precision/test_hf_attention_cp_relayout.py`.

[View workflow run](https://github.com/radixark/miles/actions/runs/<run-id>)
```

The reaction acknowledges that the request was accepted, while the reply identifies the dispatched run. The reply is not updated as the run status changes; the final pass or failure remains in that `Rerun Test` Actions run.

An explicit file request is the selection: domain labels and the nightly cadence gate do not apply, while a symlinked or `disabled` test, an unregistered path, a ROCm-only registration, or a file with more than one CPU/CUDA registration fails the resolve job instead of silently running nothing.

The dispatched workflow and its reusable workflows come from the reviewed default-branch commit. A trusted resolver reads the separately checked-out PR snapshot without importing it and maps the requested registration to a fixed CPU/CUDA suite, runner, image, and timeout. The execution job then checks out that same exact SHA and uses the requested path as its command entrypoint: bounded `pytest` for CPU or bounded `python3` for CUDA. It does not depend on workflow or runner code from the PR snapshot.

The trust boundary ends after the resolver produces that fixed plan. The execution job intentionally installs and runs PR-controlled dependencies, configuration, imports, and test code; it is not a sandbox for hostile PR code. The command supports only same-repository PRs and requires the fixed `run-ci-file.yml` workflow on the default branch. A later push does not change the already dispatched snapshot.

A file run honors the PR body's `ci-megatron-pr` and `ci-sglang-pr` pins; CUDA file runs also honor `ci-image-tag`, while CPU file runs use the hosted bare environment. The gateway validates and forwards these inputs, and a pin whose value fails validation rejects the command. Without a `ci-image-tag` pin a CUDA run uses the released `dev` image even when the PR built a `pr-<N>` image, so docker-changing PRs should pin the tag explicitly. The run writes no perf baseline (regular cadence), and its result is informational: it does not appear among the PR's required checks.

The command gateway remains off until workflow owners complete the following steps and set the repository variable `CI_COMMAND_APP_ENABLED=true`:

1. Create a GitHub App, install it only on `radixark/miles`, and grant `Pull requests: read`, `Issues: write`, and `Actions: write`; do not grant `Contents: write`. Mutation tokens remain capability-specific: label commands request `Issues: write`, while rerun and file-run commands request `Actions: write`. After a successful `/rerun-test` dispatch, independent feedback jobs each request only `Issues: write` to add the 👍 reaction and workflow-link reply.
2. Store the App client ID in the repository variable `CI_COMMAND_APP_CLIENT_ID` and its private key in the repository secret `CI_COMMAND_APP_PRIVATE_KEY`.
3. Protect the final bytes under `.github/workflows/` that implement the command gateway—its workflows, handler, and policy: require code-owner review, enable stale-review dismissal or last-push approval, and explicitly accept administrators who can still bypass the rule as external trust roots.
4. In the target repository, compare manually adding a test label with adding the same label through the App. Confirm that both trigger the expected CUDA, ROCm, and held-run approval consumers.
   Then run `/clear-labels`; confirm that it removes only the CI control labels and does not start another CUDA, ROCm, or held-run approval workflow.
   Then create a disposable failed run on the current PR head and confirm that `/rerun-failed-ci` reruns only its failed jobs and dependent jobs.
   Finally, post `/rerun-test` with one registered CUDA test file and confirm the dispatched run uses that file as the command entrypoint from the recorded PR head SHA on its registered suite's runner, the original command receives a 👍 reaction only after dispatch succeeds, and a separate reply links that exact workflow run.

The handler evaluates the caller against the checked-in access policy before minting the App token and again in the capability-specific job before the command mutates GitHub state. An explicit `add_label_access.user_ids` match uses the numeric comment-author identity bound to the event; otherwise the handler checks the caller's live repository permission.
Actions command comments for one PR are serialized in GitHub's queued concurrency mode. Up to GitHub's 100-pending limit, commands wait instead of replacing an older pending comment.
Before each rerun request, the handler rechecks the permission, PR head, and latest run of that workflow. Each lookup is a point-in-time result, so a small race remains between the final check and the mutation request.

The comment handler runs only fixed, reviewed code from the default branch and never checks out or executes PR code, dependencies, artifacts, or configuration. The file-run resolver is a separate trusted job; its execution jobs intentionally run the exact PR snapshot, receive only the test secrets required by the selected reusable workflow, and never receive the command gateway App private key.

Initial authorization, PR, policy, or run-state errors fail before any mutation. A recheck error before a later rerun request stops that request, while any earlier accepted reruns remain applied.

If the additive label `POST`, a label `DELETE`, a failed-job rerun `POST`, a workflow-dispatch `POST`, the later reaction `POST`, or the workflow-reply `POST` was sent but its response timed out, was malformed, or could not be confirmed, GitHub may already have applied the change. A feedback failure after dispatch does not cancel the file run.

`/clear-labels` and `/rerun-failed-ci` can issue multiple requests and are not atomic: if a later request fails, earlier changes remain applied. The handler does not retry or roll back automatically; inspect the PR's current labels, comments, or Actions runs before deciding whether to retry. Manually rerunning a reply job after an ambiguous response can create a duplicate link comment.

GitHub reruns failed jobs and their dependent jobs with the original run's `GITHUB_SHA`, `GITHUB_REF`, event payload, and triggering actor privileges. A rerun therefore does not inherit the commenter's or App token's privileges; consumers of the original event payload do not see labels changed afterward. GitHub permits reruns for up to 30 days after the original run and limits a workflow run to 50 attempts.

## Cadence eligibility

There are four CI cadences: `regular`, the ordinary mode; `nightly`, which admits `nightly=True` tests and broadens scope; `weekly`, which admits those tests and selects every enabled tag; and `release`, an explicit called-workflow cadence with weekly's selection but no rolling performance-baseline writes. Nightly, weekly, and release all bypass fast-fail.

`register_*_ci(nightly=True)` means the test is eligible under nightly, weekly, and release cadence, but not regular cadence. It does not create a separate suite inventory and does not replace domain-label filtering. A regular run selects regular registrations only; nightly, weekly, and release select regular plus `nightly=True` registrations, then apply their own domain-label scopes. For example, a `nightly=True` test carrying only `ft-long` remains outside the nightly scope unless `run-ci-ft-long` or `run-ci-all` explicitly includes it, while weekly and release include it automatically.

## Broad CI scopes

The workflow's `resolve-ci-policy` job forwards trigger-specific facts or a called workflow's explicit cadence override to `tests/ci/ci_policy.py`; that module adapts them into cadence and label inputs, then its shared `resolve_policy` maps those inputs to one effective include-label set and fast-fail policy. `run_suite.py` consumes the same resolved-policy function and never derives policy from `schedule`, `workflow_dispatch`, or the caller's inherited event name. A broad scope is just a large include set (every registered label minus the scope's subtractions).

| Scope | Explicit source | Runs | Subtracts | Fast-fail |
|---|---|---|---|---|
| all | `run-ci-all` label | every enabled tag | — | determined by cadence |
| weekly | exact weekly cron | every enabled tag | — | disabled on both levels |
| release | `release-branch-cut.yml` calling with `cadence=release` | every enabled tag | — | disabled on both levels |
| nightly | resolved nightly cadence from the PR label, exact nightly cron, or local `--nightly` | every enabled tag except `long` and `ft-long`, incl. `ft-short` | `long`, `ft-long` | disabled on both levels (within-stage only for local runs) |
| image | `run-ci-image` label | every enabled tag except `long` and FT tags | `long`, `ft-short`, `ft-long` | determined by cadence |

Release and weekly have the same selection and fast-fail policy. Release is separate because frozen release refs must never write the rolling performance baseline; see [Metric history & regression gate](/ci/03-metric-history-gate#trust-cleanup-who-writes).

Rows are in precedence order: when scope signals overlap, the higher row wins (`run-ci-all` > weekly/release full scope > nightly > `run-ci-image`, the branch order of `resolve_policy`). `run-ci-all` widens only the domain scope; regular cadence still does not admit `nightly=True` registrations.

The generic triggers carry no policy. All scheduled runs use UTC: nightly is identified by the exact cron `0 15 * * 0-5`, and weekly by `0 15 * * 6`. Saturday weekly replaces that day's nightly rather than starting alongside it. A manual dispatch uses regular cadence and no PR labels; the GPU workflow commands explicitly add `--match-all-labels` so that operation runs the full regular GPU suites without changing cadence. A called workflow instead supplies its cadence explicitly, which is how `release-branch-cut.yml` selects release.

A subtraction is not a per-test veto — it only stops that label from granting inclusion. A test carrying a subtracted label still runs when another of its labels is in the set, so a test that must stay outside the standard nightly scope must carry only labels that nightly subtracts.

A domain label explicitly requested on the PR wins over a scope subtraction: `run-ci-image` plus `run-ci-long` or `run-ci-ft-short`, and nightly plus `run-ci-long`, add the explicitly requested tests back rather than silently dropping the request.

## Registration and scan scope

Labels are optional for CPU registrations and required for GPU registrations; registration itself is not optional. The runner scans `tests/fast`, `tests/fast-gpu`, `tests/e2e`, `tests/ci` recursively for `test_*.py`. Every file must resolve to a registration or collection fails:

- A file outside `tests/fast/` with no `register_*_ci()` call → `No CI registry found`.
- A CUDA or ROCm registration with missing, `None`, or empty `labels` → `labels ... must contain at least one domain label`.
- A `labels=[...]` value not in `KNOWN_LABELS` → `unknown labels [...]`.

## `tests/fast/` auto-registers as CPU

Each `test_*.py` under `tests/fast/` is auto-registered as a CPU test (backend CPU, suite `stage-a-cpu`, `labels=[]`) with no `register_*_ci()` call, and runs on the GitHub-hosted `ubuntu-latest` runner. Here "CPU" is the hardware backend, not a label. A `register_cuda_ci()` under `tests/fast/` is a hard error — move it to `tests/fast-gpu/`.

## `bypass-fastfail`: opt out of fast-fail

By default CI fails fast on two levels:

- Cross-stage: CUDA GPU stages run only when `stage-a-cpu` succeeds — the `if` requires `needs.stage-a-cpu.result == 'success'`.
- Within-stage: each suite stops at the first failure (`pytest -x` for CPU; `run_unittest_files` breaks on the first failing file for CUDA).

The `bypass-fastfail` PR label turns both off so one run surfaces every failure:

- Cross-stage: each CUDA GPU stage consumes the shared `bypass_fastfail` policy output, so CUDA GPU stages run even after `stage-a-cpu` fails.
- Within-stage: `run_suite.py` derives continue-on-error from the same resolved policy (drops `pytest -x`; sets `continue_on_error=True` for CUDA). The stage still ends red — it changes coverage, not the verdict.

The ROCm workflow has no CPU cross-stage gate, so `bypass-fastfail` changes only its within-stage behavior.

A resolved nightly, weekly, or release cadence bypasses fast-fail on both levels so a broad run surfaces every failure rather than stopping at the first. For nightly this applies equally whether the cadence came from the PR `nightly` label or the explicitly mapped nightly cron; release comes from the called workflow's explicit override. Local `--nightly` applies the same nightly selection and within-stage behavior; cross-stage gating does not exist in a local invocation.

Like the scope labels, `bypass-fastfail` is a workflow-only input and is not in `KNOWN_LABELS`.

## Labels double as fork-PR CI approval

GitHub holds a first-time contributor's fork-PR CI at "Approve and run" after every push. Any maintainer-applied or comment-gateway-authorized `run-ci*` label is already that human decision, so the `Approve Trusted CI` workflow (on `pull_request_target`) auto-approves those held runs while such a label is present. Removing the labels restores manual approval. This automation covers the first-time-contributor hold only; GitHub may separately hold a workflow it identifies as potentially malicious, and that hold requires approval through an authenticated web session.
