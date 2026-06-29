---
title: Metric history & regression gate
description: How CI collects per-test training metrics across commits, runs a two-layer regression gate (fixed safety limit + data-driven history check), attributes a regression to the PR that introduced it, and how an operator cleans a bad data point.
---

# Metric history & regression gate

Each training test emits a few numbers per run — train-rollout logp abs diff, grad norm, raw rewards, KL values at each step. Today CI checks those against fixed thresholds someone hand-picked (`--ci-<metric>`), or compares two runs inside one test. It never compares a number against how the *same test* behaved on earlier commits, so a slow drift — logp diff creeping from 0.007 to 0.009 across a few PRs — slips through as "noise" while a real regression hides under a loose threshold.

This feature **keeps the per-test, per-metric numbers from every CI run in our own store, and runs a two-layer gate against that history** — catching both gross breakage and slow drift, and pointing at the PR that moved a number.

The numbers already go to wandb every run. wandb stays a **write-only sink**: it keeps receiving the numbers unchanged, but the gate **never reads from wandb**. The baseline we check against lives in our own database. (wandb is a logging dashboard, not a clean baseline; removing a bad point there means deleting whole runs, and the API has access limits.)

## Identity: what counts as "the same test"

A metric value is keyed at two levels.

**Run-level** — `(test_path, backend, suite, test_file_hash)`. This is the series a value belongs to: one test file, on one hardware backend, in one suite. `test_file_hash` is the **sha256 of the test file contents** — so editing the test resets its baseline, and a metric shift caused by changing the test itself never reads as a regression against the old code's history.

**Value-level** — `(metric_key, sub_label)` within a run. `metric_key` names the metric (e.g. `train_rollout_logp_abs_diff`, `grad_norm`); `sub_label` distinguishes points under one metric (e.g. the per-step `ppo_kl` at step 0, step 1, …). A run carries many `(metric_key, sub_label)` values.

## Storage: Neon, two normalized tables

History lives in **Neon** (managed serverless Postgres). Serverless and queryable means the gate asks the DB directly for what it needs (mean / std over the trusted tail of a series) instead of pulling rows back to compute locally; it has a browsable web table for inspecting trends and fixing a bad point; it is cheap for this workload; and CI has no cloud DB today, so something is needed either way — Neon needs just one project and one password held as a CI secret.

Two normalized tables sit behind a `MetricHistoryStore` abstraction (the gate and collector talk to that interface, never to raw SQL):

**`runs`** — one row per CI run of one test series.

| Column | Meaning |
|---|---|
| `run_id` | primary key |
| `test_path`, `backend`, `suite`, `test_file_hash` | the run-level identity above |
| `commit_sha`, `pr_number` | which commit / PR produced the run (for attribution) |
| `github_run_id`, `github_run_attempt` | the Actions run + attempt, to trace a row back to its job |
| `event_name`, `ref` | how the run was triggered (`schedule`, `pull_request`, …) and on what ref |
| `created_at` | insertion time, used for the trend order |
| `trusted` | whether this run counts toward the baseline — **run-level** |

**`metric_values`** — one row per value within a run.

| Column | Meaning |
|---|---|
| `run_id` | foreign key into `runs` |
| `metric_key`, `sub_label` | the value-level identity above |
| `value` | the number |

`trusted` lives on `runs`, not on `metric_values`: a run is admitted or rejected as a whole, so every value it produced shares one trust verdict. The read path is served by a composite index `runs(test_path, backend, suite, test_file_hash, trusted, created_at DESC)` — the gate fetches the trusted tail of one series in series order.

**No runtime DDL.** The schema is created once by a versioned migration, never by the CI process. The CI database role is **DML-only** — `INSERT` / `SELECT` / `UPDATE`, no `CREATE` / `ALTER` / `DROP` — so a CI run can write rows and mark a row untrusted but can never reshape the schema.

## The gate: two layers

After a test **passes**, the gate evaluates each `(metric_key, sub_label)` value two ways. Both use the same rel-OR-abs tolerance: a value fails when

```
|cur - ref| > max(rel * |ref|, abs_floor)
```

`rel` defaults to **0.20** (a value may move 20% before it's flagged). `abs_floor` is a small fixed wiggle that only matters for metrics normally near zero — e.g. step-0 `ppo_kl`, where a percentage of ~0 is meaningless; for ordinary metrics `rel * |ref|` dominates and `abs_floor` is effectively off.

**Hard gate — always on.** A hardcoded safety limit ("this should never happen"), with `ref` the fixed limit. It runs even for a brand-new test with zero history, catching gross breakage. This is the role the existing `--ci-<metric>` thresholds play; the hard gate generalizes them and is always present.

**Historical gate — data-driven, activates with ≥1 trusted same-hash point.** `ref` is the **mean of the trusted runs in the same series** (same `test_file_hash`). A value that deviates from that mean beyond tolerance fails. This is the layer that catches slow drift and produces the trend.

**Cold start.** With **0 trusted points** in the series, the historical gate is simply inactive — the run is checked by the hard gate only, and the absence of history is **not** an error. The historical gate switches on as soon as the series has at least one trusted point.

## Trusted admission and cleanup

A run is **trusted iff it passed ALL active gates** for every value. A run that trips the historical gate (a drifting run) is **still recorded** — its rows go into `runs` / `metric_values` — but with `trusted = false`, so it can't quietly drag the baseline along. Accepting a drifted value as the new normal is a deliberate act, not an automatic one.

**Cleanup** is a single `UPDATE runs SET trusted = false` (call it `mark_untrusted`) on the offending run. Because the historical gate's mean reads only trusted rows through the composite index, the next gate evaluation excludes the marked run **immediately** — there is no rebaseline step, no row deletion, and the run stays in the table as a record of what happened. This is the "clean a bad data point" operator action.

## Which runs write a baseline

Ordinary PR runs are **read-only**: they fetch history, run the gate, and (a PR run records its result as an untrusted-eligible row per the rollout policy below) — but a normal PR **cannot move the baseline**. Only **nightly-marked** runs write baselines, where nightly is detected as:

- `event_name == 'schedule'` (the daily cron), **or**
- the PR carries a `nightly` label.

The nightly signal — the `schedule` cron (`0 15 * * *`) and the `nightly` label — **already shipped on `main`** (PR #1491, commit `edfa2e0e1`). The harness reads it from the `GITHUB_EVENT_NAME` environment variable; it does **not** use the separate `--nightly` / `NIGHTLY_SUITES` flag. A nightly run is the full suite with fast-fail off, so it produces one datapoint per test in a single pass.

Coverage: short runs must be covered by the historical gate; long regression tests continue to track via wandb.

## Collection: alongside wandb, process-safe

Metrics are collected **while the test runs**, on the same `log()` fan-out that already feeds wandb. A `CIHistoryBackend` runs **alongside** `WandbBackend`: every `log()` call reaches both. The `CIHistoryBackend` does not talk to the DB during the run — it appends a **process-safe local NDJSON record per process**, which is safe under Ray multi-process training where several processes log concurrently (each writes its own file, no shared-handle contention).

After the test passes, the harness:

1. merges the per-process NDJSON files into one per-run record,
2. assigns the run-level identity (`test_path`, `backend`, `suite`, `test_file_hash`) and the GitHub provenance columns,
3. reads the trusted history for the series, runs the two-layer gate,
4. writes the run (`runs` + `metric_values`) with its `trusted` verdict, and fails CI if the gate failed.

Nothing is fetched from wandb at any step; the local NDJSON files are the only source for a run's own numbers.

## Rollout: shadow-first

The gate rolls out **shadow-first**: it collects, stores, and evaluates, but **never blocks any PR** in the initial milestone — a historical-gate failure is recorded (the run lands untrusted) and surfaced, not enforced. Enforcement arrives in a later milestone behind an **allowlist** of opted-in tests plus a **global kill-switch**, so the gate can be turned hard for a known-good set and disabled wholesale if it misbehaves.

## Relationship to the rest of CI

This design needs **no edit to `.github/workflows/pr-test.yml`**. That file carries `# doc-dev:` sentinels pointing at `docs/ci/00-stage.md` and `docs/ci/01-label.md`, so a documented-behavior change to its flagged regions would have to land in those docs first. But the nightly trigger this feature relies on (the `schedule` cron and the `nightly` label) is **already upstream** and already documented in `00-stage.md` / `01-label.md`, and nightly detection here is **harness-side**, reading `GITHUB_EVENT_NAME` rather than adding any workflow logic. So this feature triggers no doc-first change to `00-stage.md` or `01-label.md`, and no workflow edit.

## Assumptions

- `test_file_hash` is sha256 of the test file **contents**; any edit to the test file is an intentional baseline reset for that series.
- The Neon schema is owned by a versioned migration; the CI role is DML-only and never issues DDL at runtime.
- The historical gate reads only `trusted` rows; `mark_untrusted` is the only cleanup primitive and takes effect on the next gate read with no rebaseline.
- wandb is write-only for the gate; it is never read as a baseline.
- Nightly is the only writer of baselines, detected via `GITHUB_EVENT_NAME == 'schedule'` or the `nightly` PR label — not the `--nightly` / `NIGHTLY_SUITES` flag.

## Open questions

- Whether a brand-new test's first few baselines should need a human to confirm before they count as trusted. Preference for v1: no — the hard gate covers cold start, and the historical gate activates automatically from the first trusted point.
- Reserved customization space per metric (per-series `rel` / `abs_floor` overrides) beyond the global defaults.
