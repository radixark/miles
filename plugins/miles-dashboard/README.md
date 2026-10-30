# miles dashboard

Training-dynamics and efficiency dashboards for miles runs, built on top of
`--dump-details` plus a live telemetry collector. It watches a run through
miles' run observers (`miles.utils.observers`).

## Collect

```bash
MILES_DASHBOARD=1 python train.py ... \
    --dump-details /path/to/dump \
    --use-rollout-entropy        # optional: per-token entropy in the token view
```

A launcher sets `MILES_DASHBOARD=1` in its `extra_env_vars`. The settings are
environment variables too; an unset one keeps its default in `CollectorConfig`
(`collector.py`):

| Variable | Controls |
|---|---|
| `MILES_DASHBOARD_FLUSH_INTERVAL` | collector disk flush cadence (s) |
| `MILES_DASHBOARD_GPU_SAMPLE_INTERVAL` | NVML sampling cadence (s) |
| `MILES_DASHBOARD_SGLANG_SCRAPE_INTERVAL` | engine scrape cadence (s) |
| `MILES_DASHBOARD_SGLANG_METRICS` | comma-separated override of the scraped sglang metrics |
| `MILES_DASHBOARD_FORWARD_PROMETHEUS` | `1` also pushes dashboard gauges to the `--use-prometheus` collector for external Grafana |

`MILES_DASHBOARD=1` turns on the dashboard's run observer (`observer.py`). It starts a named collector actor on the
driver node, which runs one NVML sampler per GPU node and an sglang
engine-metric scraper, and forwards what every process of the run reports
(timer phases, sample lifecycles, metrics, engine topology, sandbox health)
to it. Everything is appended under `{dump-details}/dashboard/` (append-only
JSONL). Overhead on the training path is a few milliseconds per step; all
pushes are fire-and-forget and a dead collector never affects training.
The dashboard cannot serve a run without `--dump-details` or on
`--cluster-backend kubernetes`, so such a run stops at startup.

`--dump-details` alone (without `MILES_DASHBOARD`) still writes the rollout
`.pt` and — for session / multi-turn runs — a raw-conversation sidecar
(`trajectory/`, for the conversation view). The viewer writes a per-token
column mirror into its cache directory (`--cache-dir`) the first time it reads
a step, so later token views never full-load the `.pt`.

## View

```bash
python -m miles.dashboard.serve --dump-details /path/to/dump [--follow] [--port 7788]
```

Works on any machine that can see the directory (login node over NFS, the
training node itself); `--follow` tails a still-running job, typically over an
SSH port-forward. `--use-utilization-overview` forces the fleet overview
(auto-enabled above 64 lanes). Runs recorded *without* `MILES_DASHBOARD`
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

- **Sandboxes** — shown once an agent function reports a sandbox (see
  below): how many sandboxes are healthy, erroring or stale, and a table of
  them with their last error and when they last ran a command. A healthy
  sandbox is stale when it has not reported for longer than a configurable
  number of minutes (also settable as `#/sandboxes?stale_min=N`; for a
  finished run, measured from the last snapshot).

## Report sandbox health

Agentic rollouts that run each trial in a sandbox can make its health
visible by calling `report_sandbox` from the agent / custom generate
function: when the sandbox starts, after each command it runs, on an error,
and on teardown. The call is cheap, never raises, and is a no-op when the
run has no observers. It works with any sandbox backend: `provider` and
`sandbox_id` are free-form labels, and miles never talks to the provider.
Under `--custom-agent-function-mode subproc` the agent process sends each
report to the rollout process as it is made, so a hung call still shows up.

```python
from miles.rollout.agentic.sandbox_reports import report_sandbox


async def run_trial(trial_id: str, task: str, env) -> float:
    report_sandbox(trial_id, task=task, provider=env.name, sandbox_id=env.id)
    try:
        for command in commands_for(task):
            await env.exec(command)
            report_sandbox(trial_id)  # heartbeat: sets "last executed"
        return await verify(env)
    except Exception as e:
        report_sandbox(trial_id, status="error", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        await env.close()
        report_sandbox(trial_id, status="finished")
```

Only the latest state of each sandbox is kept; finished sandboxes drop out
after 10 minutes. The collector overwrites
`{dump-details}/dashboard/sandboxes.json` on every flush, and
`/api/sandboxes?stale_after_s=N` serves the counts and rows.

## Develop

```bash
python -m miles.dashboard.serve --demo    # generated demo data, no cluster needed
python -m pytest tests/fast/dashboard/ -q
MILES_DASHBOARD_REALDATA_DIR=/path/to/real/dump python -m pytest tests/fast/dashboard/ -q
```

Architecture:

```
producers (the run observer in every process / NVML samplers / sglang scraper)
    -> DashboardCollector (named actor, driver node)          [live JSONL streams]
dump .pt + trajectory/*.jsonl                                [written by training]
    -> serve.py: MetricStore + DumpReader -> FastAPI -> static SPA
```

## Read bounds

Live streams are hourly JSONL partitions; the server keeps the newest
`MetricStore.PARTITION_CACHE_BLOCKS` (24) parsed per stream and tails the open
one. Every request is sized by its window, never by the run's length:

- Timeline windows are at most `MAX_WINDOW_S` (4 h). Phases read the window's
  partitions plus `MAX_WINDOW_S` past its end (a phase lands in its end hour);
  intervals still open come from an index built once and advanced on new bytes.
- Run-long engine series (the Metrics `sglang` category) come from a view
  thinned to at most `2 * SCRAPES_PER_ENGINE` scrapes per engine, built from
  appended bytes only. Metric names come from `engine_catalog.json`, written by
  the collector as it sees new names.
- The engine overlay fetches only the engines on the selected lanes.
- Advisories judge the trailing `MAX_WINDOW_S` before the newest engine scrape;
  the abort alarm reads run totals only once it fires. The timeline renders
  without waiting for them and, in follow mode, refreshes them every 30 s.
- The Rollouts table builds 200 rows at a time.

What still scales with history or step size: the first Metrics `sglang` query
(it folds the whole run once), step summaries (one `torch.load` of the step's
train dumps), and the lifecycle heatmap and trajectory reads, which parse
trajectory rows as objects.

## Simulating a large run

`tests/manual/dashboard_simulator/` replays a real `--dump-details` run widened to N
GPUs: every calibration GPU, engine and train rank becomes N/calibration copies,
each rollout step carries that many times the samples, and the actor's DP size
grows with it. All bytes are written by miles' own code (sinks, collector,
`save_debug_rollout_data`, the train-data DP split and dump functions) on a
virtual clock, so a 26 h run generates in minutes to hours; only per-token
training values are synthetic. A x1 replay matches its calibration row for row.

```bash
python -m tests.manual.dashboard_simulator generate --gpus 256 --hours 26 --out <dir> \
    --calibration <real dump> --actor-tp-size 2 --actor-pp-size 4 --actor-cp-size 4 --max-tokens-per-gpu 8192
python -m tests.manual.dashboard_simulator bench-static --dump-details <dir>      # every endpoint, cold / warm
python -m tests.manual.dashboard_simulator bench-follow --dump-details <copy>     # serve --follow + live appends
python -m tests.manual.dashboard_simulator bench-collector --metrics-page <page> --engines 128
python tests/manual/dashboard_simulator/browser.py --url http://127.0.0.1:7788 --dump-details <dir>
```

`bench-follow` appends new hour partitions, so run it on a hardlink copy
(`cp -al`). The actor flags are the calibration launcher's: a dashboard dump
does not record them. `--rollout-dumps-last-hours` / `--train-dumps-last-hours`
keep only recent steps' dumps; the reader opens one step at a time, so older
steps cost disk, not request time.
