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

PR labels without the `run-ci-` prefix are ignored.

CUDA and ROCm registrations must declare at least one domain label. GPU runners are scarce and expensive, so an always-on GPU test would defeat the purpose of selecting only the GPU coverage a PR requests.

### The canonical label list

Domain labels live in `tests/ci/labels.py` (`KNOWN_LABELS`); a `labels=[...]` value outside it is a hard error. Current set: `megatron`, `model-scripts`, `sglang`, `fsdp`, `short`, `long`, `ckpt`, `lora`, `eval`, `precision`, `ft-short`, `ft-long`, `weight-update`, `fully-async`, `replay`, `qwen35`, `mooncake`, `miles-plugin`, `amd`.

To add one: add the entry to `KNOWN_LABELS`, then create the matching `run-ci-<key>` label on the PR. No workflow edit needed.

## 通过 PR 评论添加 CI label

启用 comment gateway 后，在 open PR 下单独评论 `/run-ci-<key>`，等价于追加同名的 `run-ci-<key>` label。例如，`/run-ci-short` 只会请求 `run-ci-short`。命令只允许首尾 whitespace，不能携带参数、正文、第二条命令或任意其他 label；已有 label 会作为成功的 no-op，不会产生新的 `labeled` event 或 rerun。

`.github/workflows/policies/ci-label-access.json` 是默认拒绝的精确 ACL：每个 label 映射到允许使用它的 GitHub numeric user ID。新增 `KNOWN_LABELS` 不会自动开放评论权限，必须另行添加精确 policy entry。v1 初始 policy 开放所有当前 domain label 和 `run-ci-image`，但只授权当前 workflow owner：`yushengsu-thu`（`11704492`）、`guapisolo`（`48857426`）和 `yueming-yuan`（`112649537`）。fork PR 还要求调用者出现在 `fork_actor_ids` 中；初始同样只有这三人。

gateway 只管理通过评论委托添加 label 的路径，不限制用户原有的 GitHub UI/API label 权限。它只追加 label，不移除 label，也不提供 `/run-ci-all`、`nightly`、`bypass-fastfail` 或通用 `/label` 命令。成功添加的 label 会跨后续 push 保留，撤销授权需要人工移除 label。

workflow 默认关闭。workflow owner 完成以下步骤后才能设置 repository variable `CI_LABEL_APP_ENABLED=true`：

1. 创建并仅在 `radixark/miles` 安装 GitHub App，授予 `Pull requests: read` 和 `Issues: write`；不授予 `Actions` 或 `Contents: write`。
2. 将 App client ID 存为 repository variable `CI_LABEL_APP_CLIENT_ID`，将 private key 存为 repository secret `CI_LABEL_APP_PRIVATE_KEY`。
3. 保护 `.github/workflows/comment-ci-label.yml`、handler 和 policy 的最终 bytes：要求 code-owner review，并启用 stale-review dismissal 或 last-push approval；显式接受仍可 bypass 的管理员作为外部信任根。
4. 在目标仓库对比人工添加与 App 添加同一测试 label，确认两者都触发预期的 CUDA、ROCm 和 held-run approval consumer，然后移除测试 label。

handler 只运行默认分支的固定代码，不 checkout 或执行 PR 代码、依赖、artifact 或配置。未授权、未知、格式错误、关闭 PR、删除 fork、身份不一致、policy 错误，以及读取 PR 时的 GitHub API 错误都会在 label mutation 前失败。如果追加 label 的 POST 已发出但响应超时、损坏或无法确认，GitHub 可能已经完成变更；handler 不自动重试或 rollback，操作者必须先查看 PR 的当前 label 再决定是否重试。

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

- Cross-stage: GPU stages run only when `stage-a-cpu` succeeds — the `if` requires `needs.stage-a-cpu.result == 'success'`.
- Within-stage: each suite stops at the first failure (`pytest -x` for CPU; `run_unittest_files` breaks on the first failing file for CUDA).

The `bypass-fastfail` PR label turns both off so one run surfaces every failure:

- Cross-stage: each GPU stage consumes the shared `bypass_fastfail` policy output, so GPU stages run even after `stage-a-cpu` fails.
- Within-stage: `run_suite.py` derives continue-on-error from the same resolved policy (drops `pytest -x`; sets `continue_on_error=True` for CUDA). The stage still ends red — it changes coverage, not the verdict.

A resolved nightly, weekly, or release cadence bypasses fast-fail on both levels so a broad run surfaces every failure rather than stopping at the first. For nightly this applies equally whether the cadence came from the PR `nightly` label or the explicitly mapped nightly cron; release comes from the called workflow's explicit override. Local `--nightly` applies the same nightly selection and within-stage behavior; cross-stage gating does not exist in a local invocation.

Like the scope labels, `bypass-fastfail` is a workflow-only input and is not in `KNOWN_LABELS`.

## Labels double as fork-PR CI approval

GitHub holds a first-time contributor's fork-PR CI at "Approve and run" after every push. Any maintainer-applied `run-ci*` label is already that human decision, so the `Approve Trusted CI` workflow (on `pull_request_target`) auto-approves the held runs while such a label is present. Removing the labels restores manual approval; the friction ends permanently once the contributor's first PR merges.
