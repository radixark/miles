# miles dashboard

Training-dynamics and efficiency dashboards for miles runs, built on top of
`--dump-details` plus a live telemetry collector.

## Collect

```bash
python train.py ... \
    --dump-details /path/to/dump \
    --use-miles-dashboard \
    --use-rollout-entropy        # optional: per-token entropy in the token view
```

`--use-miles-dashboard` adds a named collector actor on the driver node, one
NVML + host-memory sampler per GPU node, per-rank phase interval sinks on the
existing `Timer` instrumentation, and an sglang engine-metric scraper.
Everything is appended under `{dump-details}/dashboard/` (append-only JSONL).
Overhead on the training path is a few milliseconds per step; all pushes are
fire-and-forget and a dead collector never affects training.

`--dump-details` alone (no `--use-miles-dashboard`) still writes the rollout
`.pt`, a per-token column mirror (`dashboard_columns/`, so the token view
never full-loads the `.pt`), and — for session / multi-turn runs — a
raw-conversation sidecar (`trajectory/`, for the conversation view).

## View

```bash
pip install -e .[dashboard]          # fastapi/uvicorn; already in the training image
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
- **Compute Utilization** — one node-level CPU memory-pressure row plus per-GPU
  lanes below 64 GPUs (phase band, NVML utilization, sglang overlay,
  click-to-zoom bubble strip); hover CPU memory for used/available/total GiB.
  Above that, a scale-invariant fleet overview (phase composition + utilization
  band) with a lane-selection grammar (`g:` / `rank:` / `node:` / `every:`)
  and outlier quick-picks.
- **Rollouts** — per-step trajectory table and scatter, GRPO group degeneracy
  (`zero_std`), average weight-version staleness, eval tab.
- **sample view** — a `conversation` tab (role-tagged turns with thinking /
  tool calls, from the trajectory sidecar) and a lazily-loaded `tokens` tab
  (per-token metric strips + a selectable metric-vs-position chart,
  loss-masked regions dimmed).

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
