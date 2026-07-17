---
name: ci-fetch-log
description: Fetch complete GitHub Actions job logs with `gh`, write an auditable completeness manifest, and optionally diagnose failed jobs from saved evidence. Use when asked to collect full CI logs, save a workflow run for later analysis, or explain failed jobs. A PR number resolves to the newest workflow run for the PR's current head SHA. PR numbers, run URLs, and job URLs default to the whole run; `--only` requires an explicit single-job request.
---

# Fetch Complete GitHub Actions Logs

Produce complete GitHub Actions job logs, a machine-readable manifest, and an optional failure diagnosis. Do not edit code, commit, push, rerun jobs, or apply a proposed fix.

## Choose the scope

Require a PR number for the current checkout's GitHub repository, or a `github.com` Actions run URL or job URL:

```text
<pr_number>
https://github.com/<owner>/<repo>/actions/runs/<run_id>
https://github.com/<owner>/<repo>/actions/runs/<run_id>/job/<job_id>
```

Choose one mode:

- **Complete-log collection:** Save logs and the manifest. Use this by default for requests involving complete logs, successful-job logs, timing data, or later parsing.
- **Failure diagnosis:** Save logs for failed-like jobs and explain the failures from the saved evidence. Use this when the user asks why CI failed.

Selection defaults to the whole run regardless of input shape:

The selector names below are request-level controls for this skill, not flags to forward to `gh run view` or `gh api`.

- Complete-log collection selects every job in the pinned run attempt.
- Failure diagnosis selects jobs in the pinned attempt concluded as `failure`, `cancelled`, `timed_out`, or `action_required`.
- A PR number selects the newest workflow run for the PR's current head SHA, then follows the same whole-run rules as a run URL. It does not narrow selection to one job.
- A whole-run URL selects its latest attempt unless the user supplies the request-level selector `--attempt <number>`.
- A job URL resolves and pins its attempt from the job metadata endpoint. Its job ID is a hint only and does not narrow the default to one job.
- `--only` selects exactly one job and requires a job URL or an explicit `--job <id>` or `--job-name <glob>` selector.
- Match `--job-name` as a case-sensitive glob against each full enumerated job name. Without `--only`, it may select multiple jobs; with `--only`, zero or multiple matches are a hard error and the response must list the candidates.
- Reject `--only` without a job URL or explicit job selector. Do not choose the first failed job or silently fall back to all jobs.
- Explicit job ID, job-name, or conclusion selectors narrow the whole-run default.

Before fetching, state the mode, selected jobs, output directory, and completion condition. The operation is complete only when every selected job appears in the manifest and every completeness claim is backed by a complete saved log.

## Authenticate and use one transport

Run this in the exact shell environment that will execute every later `gh` command:

```bash
gh auth status --hostname github.com
```

Authentication is a hard gate. If it fails, stop before fetching metadata or logs. Check `whoami`, `HOME`, `GH_CONFIG_DIR`, whether `GH_TOKEN` or `GITHUB_TOKEN` is present without printing its value, and shell startup rules that may skip credential setup. Ask the user to make `gh auth status` pass in that environment.

Use `gh` as the only metadata and log transport. Do not substitute `curl`, `wget`, browser output, connector output, copied UI text, or pasted log snippets; those sources cannot establish completeness.

## Artifact contract

Use a caller-provided directory or create a fresh directory outside the repository, for example:

```text
${TMPDIR:-/tmp}/ci-fetch-log/<owner>-<repo>/<run_id>-attempt-<run_attempt>-<timestamp>/
├── run.json
├── manifest.json
├── summary.md
└── jobs/
    └── <job_id>/
        ├── job-<job_id>.log
        └── report.md        # diagnosis mode only
```

`manifest.json` is the consumer contract. Use these stable identity fields and include one entry for every selected job:

```json
{
  "producer": "ci-fetch-log",
  "schema_version": 1,
  "artifact_kind": "complete-gha-job-logs",
  "artifact_dir": "/absolute/path/to/artifacts",
  "run_url": "https://github.com/owner/repo/actions/runs/123",
  "repo": "owner/repo",
  "run_id": "123",
  "run_attempt": 2,
  "workflow_name": "CI",
  "head_sha": "abc123",
  "mode": "complete-log collection",
  "jobs": [
    {
      "job_id": 456,
      "job_name": "tests / shard 1",
      "status": "completed",
      "conclusion": "success",
      "log_path": "/absolute/path/to/artifacts/jobs/456/job-456.log",
      "log_complete": true,
      "line_count": 1234,
      "bytes": 567890,
      "fetch_command": "gh api /repos/owner/repo/actions/jobs/456/logs > ...",
      "fetched_at": "2026-05-20T05:00:00Z",
      "fetch_error": null,
      "incomplete_reason": null
    }
  ]
}
```

Set `log_complete=true` only when the job status is `completed`, a supported `gh` fetch command exits successfully, and the saved file is non-empty. Mark queued or running jobs, failed fetches, and empty files incomplete. Record the exact failure in `fetch_error` and a concise reason in `incomplete_reason`. Consumers must reject incomplete entries and manifests whose identity fields do not match this contract.

## Collect the logs

### 1. Resolve run metadata

After the authentication gate passes, resolve the input to a repository, run ID, and optional job ID, then pin the workflow attempt before enumerating jobs.

For a PR number, resolve the current repository and PR head SHA, then list runs for that exact commit:

```bash
repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
head_sha=$(gh pr view <pr_number> --repo "$repo" --json headRefOid --jq .headRefOid)
gh run list --repo "$repo" --commit "$head_sha" --limit 100 --json databaseId,attempt,createdAt,headSha,url,workflowName --jq 'sort_by(.createdAt, .databaseId) | last'
```

Require a non-null run whose `headSha` exactly matches the PR head SHA. Define newest by the maximum `(createdAt, databaseId)` pair, use its `databaseId` and `attempt`, and require a user-supplied `--attempt` to match. If the current PR head has no run, stop; do not fall back to an earlier PR commit. Record the PR number, head SHA, and chosen run URL in `summary.md`, and store the chosen run URL in `manifest.json`.

For URL inputs:

- For a job URL, query `gh api /repos/<owner>/<repo>/actions/jobs/<job_id>` and read `run_id` and `run_attempt`. Require the returned run ID to match the URL. If the user also supplied `--attempt`, require it to match `run_attempt`.
- For a whole-run URL with `--attempt <number>`, use that positive attempt number.
- For a whole-run URL without `--attempt`, read the latest attempt with `gh run view <run_id> --repo <owner>/<repo> --json attempt --jq .attempt`.

Save metadata for that pinned attempt:

```bash
gh run view <run_id> --attempt <run_attempt> --repo <owner>/<repo> --json jobs,conclusion,workflowName,headSha,url,attempt > "<artifact_dir>/run.json"
```

If the installed `gh` version rejects a field, retry only with fields it reports as supported and record the omitted field and original error in `summary.md`. Apply the mode and selectors. If no jobs match, write the selection mismatch to `summary.md` and stop.

### 2. Fetch each selected job

Create the job directory and try:

```bash
gh api /repos/<owner>/<repo>/actions/jobs/<job_id>/logs > "<artifact_dir>/jobs/<job_id>/job-<job_id>.log"
```

If that command fails, remove its failed output and try:

```bash
gh run view <run_id> --attempt <run_attempt> --repo <owner>/<repo> --log --job <job_id> > "<artifact_dir>/jobs/<job_id>/job-<job_id>.log"
```

Do not append retries to an existing log. Capture stderr separately so it does not contaminate the saved log, and preserve the exact fetch error in the manifest. Record the successful command, absolute output path, UTC fetch time, byte count, and line count. Attempt every selected job, then write `manifest.json`; never describe a partial or pasted log as complete.

### 3. Finish collection mode

Write `summary.md` with the run URL, pinned attempt, workflow name, head SHA, selected/complete/incomplete counts, and one row per selected job showing its name, ID, conclusion, completeness, and log path. Include exact errors and reasons for every incomplete log.

Return the artifact directory, `manifest.json`, `summary.md`, and the complete/incomplete counts.

## Diagnose failed jobs

Diagnose only failed-like jobs whose manifest entry has `log_complete=true`. List incomplete failed jobs separately and do not infer their cause from partial evidence.

When several jobs failed, use a fresh, separate analysis pass for each saved log and run those passes in parallel when available. Each pass must inspect only its assigned saved log and write `jobs/<job_id>/report.md` with:

- `error_class`: exactly one of `setup_or_install`, `test_failure`, `runtime_exception`, `assertion`, `timeout`, `cancelled`, `infra`, or `external_provider`.
- `error_location`: a source file and line, or a CI step and exact saved-log line span.
- `raw_log_link`: a relative link such as `[job-456.log:2871-2895](./job-456.log#L2871-L2895)`. Keep the plain line span in the link text because viewers that do not support GitHub-style fragments may ignore the anchor.
- `error_snippet`: a fenced, verbatim excerpt with each line prefixed by its saved-log line number and enough context to understand the failure.
- `command_or_step`: the exact failing step and command when available.
- `hypothesis`: at most one paragraph, explicitly labeled as a hypothesis when evidence is incomplete.
- `local_fix_sketch`: one to three sentences naming the likely file, function, threshold, configuration, or instrumentation point. If evidence is too weak, say `instrument first, fix later`. Do not include runnable code, a diff, Git commands, or a patch.

Every diagnosis claim must cite an exact line span from a complete saved log. Use numbered inspection when preparing evidence, for example:

```bash
nl -ba "<log_path>" | sed -n '<start>,<end>p'
```

Verify that every expected `report.md` exists and is non-empty before consolidation. Group reports by `error_class` in `summary.md`, but do not claim a shared cause merely because two jobs share a class. For every job, inline the verbatim failure excerpt and link to both its report and raw-log line citation so the reader can evaluate the conclusion without reconstructing it from prose.

End the diagnosis summary with a `## Next Step` section stating whether the evidence supports a targeted local fix, requires additional instrumentation and root-cause investigation, or indicates a broader CI or infrastructure repair.

Return the summary path, failed-job count, complete-log count, number of error-class groups, one evidence-backed sentence per group, and the recommended next action.
