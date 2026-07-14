---
name: ci-e2e-time-tune
description: Tune literal `register_cuda_ci(est_time=...)` values in `tests/e2e` from complete GitHub Actions logs produced by `ci-fetch-log`. Use when asked to calibrate E2E timing estimates, produce an evidence-backed runtime CSV, or apply estimate-only changes. Aggregate passed samples by test file, apply a 1.25 safety factor with mandatory upward bucket rounding, and never commit or push changes.
---

# Tune E2E CI Time Estimates

Use only complete GitHub Actions artifacts produced by the repository's [`ci-fetch-log`](../ci-fetch-log/SKILL.md) skill. Produce an auditable CSV and, unless the user requests CSV-only output, update eligible `est_time` literals.

Accept a GitHub Actions run URL, job URL, or `ci-fetch-log` artifact directory. After full-run collection is verified, optional suite, job-name, or partition filters define the selected evidence set used for the CSV and edits. They never narrow log collection, and every grouping rule below applies across all rows in that selected set.

## Obtain complete logs

For a URL, follow `ci-fetch-log` in complete-log collection mode for every job in the run. Treat a job URL only as a hint identifying the run. Never pass `--only`, `--job`, `--job-name`, a conclusion filter, or another selector that narrows collection. Do not fetch logs independently or use pasted excerpts.

For an existing artifact directory, require a `manifest.json` with:

- `producer="ci-fetch-log"`.
- `schema_version=1`.
- `artifact_kind="complete-gha-job-logs"`.
- `mode="complete-log collection"`.
- A positive `run_attempt` that matches the `attempt` recorded in `run.json`.
- An existing `run.json` whose full job-ID set exactly matches the job-ID set in `manifest.json`, proving that no selector narrowed collection.
- A manifest `repo` that matches one of the current checkout's configured GitHub remotes; stop on a repository mismatch or an ambiguous checkout identity.
- A recorded `head_sha` and current `HEAD`. A mismatch is allowed when `head_sha` is an ancestor of `HEAD`; otherwise default to report-only mode unless the user explicitly approves applying cross-revision edits.
- Every job selected for parsing marked `log_complete=true`.
- An existing, non-empty `log_path` for each selected job whose actual line count matches `line_count`.

Stop and report the failed identity, scope, or completeness check. Apply any CSV/edit-view filter only after obtaining and validating the complete-run manifest, then parse every job admitted to that selected evidence set.

## Parse scheduled tests and outcomes

Consider only scheduled files under `tests/e2e`. Ignore `tests/fast`, `tests/fast-gpu`, unit tests, setup spans, and other runtimes.

For each selected complete job log:

- Recover `suite`, `partition_id`, and `partition_size` from the job name when available.
- Parse the `Enabled N test(s)` block for scheduled files and logged estimates. A job can print the plan during both a list-only step and the real run, so deduplicate identical plan entries and prefer the block associated with the execution step.
- Parse `End (...)` timing records containing `filename='tests/e2e/...', elapsed=<seconds>, _estimated_time=<seconds>` and the corresponding `[i/N] <test_file> PASS|FAIL|TIMEOUT elapsed=<seconds>s` attempt-status lines.
- Tolerate ISO 8601 timestamp prefixes and ANSI escapes while parsing; do not rewrite the saved log.
- Parse the final `PASSED:` and `FAILED:` summary sections.
- Classify every scheduled file using the precedence below so each CSV row receives exactly one status.

Associate each `End (...)` record with its attempt-status line by filename and attempt order, not by the displayed index: legacy `End` indices are zero-based while attempt-status indices are one-based. For retries, exclude earlier failed or timed-out attempts and use only the timing associated with the passing attempt.

1. `passed`: The associated attempt has `PASS`, an `End (...)` timing record, and the file appears under the final `PASSED:` section. A passed test remains valid evidence when another test later fails the same job.
2. `timeout`: A file-level attempt has `TIMEOUT`; this takes precedence over the same file appearing under the generic `FAILED:` section.
3. `unknown`: An `End (...)` record exists but the final summary is missing, so the evidence cannot establish a successful sample.
4. `failed`: The file appears under `FAILED:` or has an associated failure record not already classified as `timeout` or `unknown`.
5. `cancelled`: The job was cancelled before a more specific file result became available.
6. `not_run`: The file was scheduled but has no timing record or final result and no job-level cancellation or timeout explains the absence.

For a job-level `timed_out` conclusion without a file-level result, classify the affected scheduled row as `timeout` before considering `cancelled` or `not_run`. Outside the explicit missing-summary rule above, use `unknown` only as the final fallback for contradictory or insufficient evidence.

## Write the evidence CSV

Write one row per scheduled `tests/e2e` file to the requested path or `<artifact_dir>/e2e-runtime.csv` by default. Use these columns:

```text
workflow_run_id,job_id,job_name,suite,partition_id,partition_size,test_file,status,elapsed_s,logged_est_s,current_est_s,proposed_est_s,action,source_log,timing_line,status_line,note
```

Populate `logged_est_s` from the complete log and `current_est_s` from the current checkout. Cite the saved log path and exact line numbers supporting both timing and status. For scheduled-only rows, cite the relevant `Enabled N test(s)` or summary line.

Compute one grouped action per `test_file` after parsing every selected job, then copy that action to every row for the file. Fill `proposed_est_s` only on `passed` rows. Use these actions:

- `raise`: the proposal is greater than the current estimate and edits apply.
- `lower`: the proposal is smaller and edits apply unconditionally.
- `keep`: the proposal equals the current estimate.
- `report_only`: passed evidence exists but edits were disabled.
- `skip`: no passed evidence exists, the file has multiple `register_cuda_ci` calls, or `est_time` is missing or non-literal.

Leave `proposed_est_s` empty for non-passed and `skip` rows.

## Compute estimates by test file

Group rows by `test_file` across all selected jobs, partitions, retries, and reruns represented in the run. Never group by `(job_id, test_file)`.

Use only `passed` rows as runtime evidence:

```text
max_elapsed_s = max(elapsed_s for passed rows of this test_file)
raw_s = ceil(max_elapsed_s * 1.25)

if raw_s <= 200:
    proposed_est_s = ceil(raw_s / 10) * 10
else:
    proposed_est_s = ceil(raw_s / 100) * 100
```

Always round upward to the bucket. Never use `raw_s` directly or round down. `raw_s == 200` proposes `200`; `raw_s == 201` proposes `300`.

Failed, timed-out, cancelled, not-run, and unknown rows never contribute to `max_elapsed_s`. They never cancel a raise or block a lower. When the bucket-rounded proposal is below `current_est_s`, apply the lower unconditionally even if sibling rows did not pass. Timeout tuning is outside this workflow unless the user requests it separately.

## Apply only literal estimate edits

For an eligible file, replace only the numeric literal assigned to `est_time` in its single `register_cuda_ci(...)` call. Preserve all surrounding formatting and code.

Do not change suites, labels, disabled flags, timeouts, skips, assertions, test logic, imports, or runner configuration. If the call is ambiguous or `est_time` is computed rather than literal, record `skip` instead of editing around it.

Do not stage, commit, push, or open a pull request. Leave edits in the working tree.

## Verify and report

Verify that:

- The CSV parses successfully.
- Every `passed` row has `elapsed_s`, `source_log`, `timing_line`, and `status_line`.
- Grouped proposals exactly follow the formula and action rules.
- The diff changes only literal `est_time` values.

Run the registry's list-only command for affected suites and partitions, for example:

```bash
PYTHONPATH=. python tests/ci/run_suite.py --hw cuda --suite <suite> --match-all-labels --auto-partition-id <id> --auto-partition-size <size> --list-only
```

Confirm the new estimates appear and partition totals remain sensible. Do not run GPU E2E tests solely to verify timing metadata unless the user explicitly asks.

Report the CSV path, raises and lowers as `max_elapsed_s -> proposed_est_s`, skipped files and reasons, verification commands, and that all edits remain uncommitted.
