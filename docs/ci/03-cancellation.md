---
title: Run cancellation
description: When an in-flight CI run is cancelled automatically — superseded within an open PR, or swept when the PR closes.
---

# Run cancellation

Two mechanisms stop CI runs that no longer matter. Within an open PR, a new run supersedes the old one; when the PR closes, everything still in flight is swept. GPU stages hold self-hosted H100/H200 runners for up to hours, so a stale run is not just noise — it delays every queued PR behind it.

## Within an open PR: supersede

`pr-test.yml` declares a `concurrency` group keyed by PR number with `cancel-in-progress: true`. A new run on the same PR (push, reopen, `run-ci-*` label) cancels the previous one. Distinct nightly schedules and manual dispatches on the default branch use different group keys and never cancel one another.

## On PR close: sweep

Merging or closing a PR does not, by itself, stop anything: GitHub cancels an in-progress run only when a *new* run enters its concurrency group, and `closed` is not a `pr-test.yml` trigger. Without a sweeper, a run started just before merge keeps occupying GPU runners to completion.

`cancel-pr-workflows-on-close.yml` (ported from sgl-project/sglang) closes that gap. On `pull_request_target: closed` — merged and unmerged alike — it lists every unfinished run on the PR's head branch **repo-wide** and cancels each one:

- Listing is by branch, not by workflow file, so workflows added later are covered automatically.
- Runs are matched against the PR's head repository id, so same-named branches in other forks are untouched.
- Every non-terminal status is swept (`queued`, `in_progress`, `waiting`, `pending`, `requested`, `action_required`), and a second pass 20 s later catches runs that were still materializing during the first.
- A run that rejects a plain cancel (stuck behind an approval or deployment protection rule) is force-cancelled via the API.

The sweep never touches runs on `main` (nightly cron, `workflow_dispatch`) or on branches without a closing PR. It uses `pull_request_target` so the job has `actions: write` even when the PR comes from a fork; the job never checks out PR code.
