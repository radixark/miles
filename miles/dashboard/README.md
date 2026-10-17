# miles dashboard

Training-dynamics and efficiency dashboards for miles runs, built on top of
`--dump-details` plus a live telemetry collector.

User-facing documentation lives at `docs/user-guide/dashboard.md`; this file is the
implementer's summary.

## Collect

```bash
python train.py ... \
    --dump-details /path/to/dump \
    --use-miles-dashboard \
    --use-rollout-entropy        # optional: per-token entropy in the token view
```

`--use-miles-dashboard` adds a named collector actor on the driver node, one
NVML sampler per GPU node, per-rank phase interval sinks on the existing
`Timer` instrumentation, and an sglang engine-metric scraper. Everything is
appended under `{dump-details}/dashboard/` (append-only JSONL). Overhead on
the training path is a few milliseconds per step; all pushes are
fire-and-forget and a dead collector never affects training.

`--dump-details` alone (no `--use-miles-dashboard`) still writes the rollout
`.pt`, a per-token column mirror (`dashboard_columns/`, so the token view
never full-loads the `.pt`), and — for session / multi-turn runs — a
raw-conversation sidecar (`trajectory/`, for the conversation view).

## View

```bash
pip install fastapi uvicorn polars   # already present in the training image
python -m miles.dashboard.serve --dump-details /path/to/dump [--follow] [--port 7788]
```

Works on any machine that can see the directory (login node over NFS, the
training node itself); `--follow` tails a still-running job, typically over an
SSH port-forward. `--use-utilization-overview` forces the fleet overview
(auto-enabled above 64 lanes). Runs recorded *without* `--use-miles-dashboard`
still get the training-dynamics views; only the timeline is absent (metrics
fall back to dump-derived `dump/*` aggregates).

Views:

- **Metrics** — a wandb-style category sidebar over every logged metric, plus
  an `sglang` category (scraped engine series) when present; hover for values,
  drag to zoom.
- **Compute Utilization** — per-GPU lanes below 64 GPUs (phase band, NVML
  utilization, sglang overlay, click-to-zoom bubble strip); above that, a
  scale-invariant fleet overview (phase composition + utilization band) with a
  lane-selection grammar (`g:` / `rank:` / `node:` / `every:`) and outlier
  quick-picks.
- **Rollouts** — per-step trajectory table and scatter, GRPO group degeneracy
  (`zero_std`), average weight-version staleness, eval tab. Opens on the newest
  step that has samples, skipping one still being dumped; a step named in the
  URL is always shown as asked.
- **sample view** — a `conversation` tab (role-tagged turns with thinking /
  tool calls, from the trajectory sidecar) and a lazily-loaded `tokens` tab
  (the whole sequence in one scrollable metric-colored strip, opened at the
  first generated token, + a selectable metric-vs-position chart, loss-masked
  regions dimmed).

- **Sandboxes** — shown once an agent function reports sandbox events (see
  below): trials per current stage, failures per exception type over time,
  and stale trials (no event for longer than a configurable number of
  minutes, also settable as `#/sandboxes?stale_min=N`; for a finished run,
  measured from its last event).

## Report sandbox events

Agentic rollouts that run each trial in a sandbox can make the trial's
lifecycle visible by calling `report_sandbox_event` from the agent / custom
generate function. The call is fire-and-forget and a no-op when the
dashboard is disabled. It works with any sandbox backend: `provider` and
`sandbox_id` are free-form labels, and miles never talks to the provider.

```python
from miles.dashboard.hooks import report_sandbox_event
from miles.dashboard.store import SandboxStage


async def run_trial(trial_id: str, task: str, env) -> float:
    report_sandbox_event(trial_id, SandboxStage.CREATED, task=task, provider=env.name, sandbox_id=env.id)
    try:
        await env.setup()
        report_sandbox_event(trial_id, SandboxStage.SETUP_DONE, task=task)
        report_sandbox_event(trial_id, SandboxStage.AGENT_STARTED, task=task)
        await run_agent(env)
        report_sandbox_event(trial_id, SandboxStage.AGENT_DONE, task=task)
        reward = await verify(env)
        report_sandbox_event(trial_id, SandboxStage.VERIFIER_DONE, task=task, reward=reward)
    except Exception as e:
        await env.close()
        report_sandbox_event(trial_id, SandboxStage.FAILED, task=task, exception_type=type(e).__name__)
        raise
    await env.close()
    report_sandbox_event(trial_id, SandboxStage.TORN_DOWN, task=task)
    return reward
```

A trial's current stage is its latest event, so a failed trial reports
`FAILED` after its teardown. Events land in
`{dump-details}/dashboard/sandbox_events.jsonl`; `/api/sandboxes` serves the
aggregate (`stale_after_s`, `bucket_s` query parameters).

## Develop

```bash
python -m miles.dashboard.serve --demo    # generated demo data, no cluster needed
python -m pytest tests/fast/dashboard/ -q
MILES_DASHBOARD_REALDATA_DIR=/path/to/real/dump python -m pytest tests/fast/dashboard/ -q
```

Architecture:

```
producers (Timer sinks / rollout hooks / NVML samplers / sglang scraper)
    -> DashboardCollector (named actor, driver node)          [live JSONL streams]
dump .pt + dashboard_columns/*.parquet + trajectory/*.jsonl   [written by training]
    -> serve.py: MetricStore + DumpReader -> FastAPI -> static SPA
```
