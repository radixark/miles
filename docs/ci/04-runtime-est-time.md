---
title: Runtime estimate calibration
description: How Miles records scheduled CUDA e2e runtimes in Neon and turns them into reviewable est_time updates.
---
CI uses each test registration's `est_time` to balance shards and derive its per-file timeout. Runtime estimate calibration records scheduled-main CUDA e2e attempts, derives stable estimates from PASS history, and proposes literal updates in a pull request. It does not change a test result, dynamically alter a running job's timeout, or update CPU and ROCm registrations.

## Data and failure boundaries

Collection is enabled only when all of these conditions hold:

- `GITHUB_EVENT_NAME=schedule`.
- `GITHUB_REF=refs/heads/main`.
- The selected backend is CUDA, the registered file is under `tests/e2e/`, and `NEON_DATABASE_URL` is set.

For each selected file, the harness buffers every completed PASS, FAIL, and TIMEOUT attempt, including retries. Each row carries `(test_path, backend, suite, test_attempt)`, measured and registered seconds, the commit SHA, and the GitHub run ID and attempt. The harness writes the batch after `run_unittest_files` finishes; cancellation or an unexpected harness exception before that write may leave no runtime rows.

Runtime-history provenance and batch-write failures are fail-open: the harness logs `[CI Runtime] history write failed` and preserves the test result. FAIL and TIMEOUT rows remain available for audit but never contribute to an estimate.

## Runtime-history storage

Runtime history reuses the repository's hosted Postgres connection but stays in the independent `ci_test_runtime_attempts` table; it does not share metric-history tables or trust semantics. Runtime code performs DML only and never creates or migrates the table.

The table definition is versioned in `tests/ci/runtime_estimate/runtime_history_schema.sql`. The repository Actions secret `NEON_DATABASE_URL` must use a role that can `SELECT`, `INSERT`, and `UPDATE` this table. The same secret is passed to scheduled CUDA jobs for collection and to the calibration workflow for reads.

The idempotency key is `(github_run_id, github_run_attempt, test_path, backend, suite, test_attempt)`. Replaying the same payload is safe; a different payload for an existing key fails and rolls back the batch instead of silently choosing one value.

After a scheduled main CUDA run, verify collection with the Neon console or `psql`:

```sql
SELECT test_path, backend, suite, test_attempt, status, elapsed_seconds,
       github_run_id, github_run_attempt, recorded_at
FROM ci_test_runtime_attempts
WHERE event_name = 'schedule'
  AND git_ref = 'refs/heads/main'
ORDER BY recorded_at DESC
LIMIT 20;
```

Do not enable automatic publishing until this query shows the expected scheduled-main rows.

## Preview an estimate update

Set `NEON_DATABASE_URL` in the local environment, then run a deterministic dry run from the repository root:

```bash
python3 -m tests.ci.runtime_estimate.update_est_time \
  --dry-run \
  --as-of 2026-08-12 \
  --report-file /tmp/ci-runtime-est-time.md
```

`--as-of` is the exclusive UTC date at the end of the history window: the example reads timestamps in `[2026-07-22T00:00:00+00:00, 2026-08-12T00:00:00+00:00)`. Omitting it uses the current UTC date. `--dry-run` prints and optionally writes the Markdown report without changing `tests/e2e`.

For each `(test_path, backend, suite)` identity, the updater:

1. Reads only PASS rows from scheduled runs on `refs/heads/main` in the 21-day half-open window.
2. Keeps the latest 15 attempts and requires at least 3.
3. Computes the inclusive p90 runtime.
4. Rounds upward to a 10-second bucket through 200 seconds or a 100-second bucket above 200 seconds.
5. Proposes both increases and decreases when the bucket differs from the registered literal.

The report includes the sample count, p90, old and new values, and a link to every contributing GitHub run attempt. The updater matches active CUDA registrations under `tests/e2e` by `(test_path, backend, suite)` and replaces only the numeric `est_time` AST literal. CPU, ROCm, disabled, unmatched, and already-current registrations remain byte-identical.

The calibration path is fail-closed: a database read, registry or AST validation, file write, or report-write error exits nonzero. The workflow does not continue to the publish step after such an error.

To apply the same update locally, omit `--dry-run`, then review only the intended literals:

```bash
python3 -m tests.ci.runtime_estimate.update_est_time --as-of 2026-08-12 --report-file /tmp/ci-runtime-est-time.md
git diff --check
git diff -- tests/e2e
```

## Enable the weekly pull request

The `Update CI runtime estimates` workflow runs every Monday at 12:00 UTC. Its scheduled job remains skipped until the repository variable `CI_RUNTIME_EST_TIME_BOT_ENABLED` is exactly `true`; manual dispatch remains available while the variable is absent or false, and defaults to `dry_run=true`.

Enable publishing in this order:

1. Configure the `NEON_DATABASE_URL` Actions secret with the required DML permissions and verify scheduled-main collection.
2. Wait until each identity to be calibrated has at least three recent PASS attempts.
3. Manually dispatch `Update CI runtime estimates` with `dry_run=true` and inspect its step summary.
4. In repository **Settings → Actions → General → Workflow permissions**, enable **Allow GitHub Actions to create and approve pull requests** as described in the [GitHub Actions settings documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository).
5. Set the repository Actions variable `CI_RUNTIME_EST_TIME_BOT_ENABLED=true`.

A manual dispatch with `dry_run=false` can publish even before the scheduled-job variable is enabled. Publishing happens only when `tests/e2e` changed. The workflow refreshes the fixed `jiajun/ci-est-time-update` branch with `--force-with-lease`, updates its existing open pull request or creates one, and never merges it.

Current limitation: the publish step authenticates with the repository `GITHUB_TOKEN`. GitHub therefore creates CI runs for the bot pull request in an approval-required state; a user with write access must approve them after each opened or synchronize event. The workflow does not implement a GitHub App or personal-access-token path for running those checks without approval; see [When `GITHUB_TOKEN` triggers workflow runs](https://docs.github.com/en/actions/concepts/security/github_token#when-github_token-triggers-workflow-runs).

## Diagnose an empty or failed update

| Symptom | Check |
|---|---|
| No runtime rows | Confirm the source was a scheduled run on `main`, the job was CUDA, the file was under `tests/e2e`, and `NEON_DATABASE_URL` reached the job. PR runs, nightly-labeled PRs, CPU, and ROCm do not collect runtime history. |
| `[CI Runtime] history write failed` | Check that `ci_test_runtime_attempts` exists, along with DSN reachability, provenance variables, and `SELECT`/`INSERT`/`UPDATE` privileges. The associated test outcome is unchanged, but its runtime evidence was not persisted. |
| The report has no changes | Check for three PASS attempts within the window. FAIL and TIMEOUT rows are excluded, and a p90 that rounds to the current literal needs no edit. |
| The scheduled workflow is skipped | Confirm `CI_RUNTIME_EST_TIME_BOT_ENABLED` is the string `true`. |
| The publish step cannot create a pull request | Confirm both the workflow's write permissions and the repository setting that allows Actions to create pull requests. |

Repository unit tests cover the collection gate, SQL and idempotency behavior, estimator, AST targeting, dry-run behavior, and workflow publish condition with fake database connections. They do not prove that the Neon table is provisioned, credentials are valid, or a live round trip succeeds; the storage query and manual dry run above are the activation checks.
