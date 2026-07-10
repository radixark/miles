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
NVML sampler per GPU node, per-rank phase interval sinks on the existing
`Timer` instrumentation, and an sglang engine-metric scraper. Everything is
appended under `{dump-details}/dashboard/` (append-only JSONL). Overhead on
the training path is a few milliseconds per step; all pushes are
fire-and-forget and a dead collector never affects training.

## View

```bash
pip install -e .[dashboard]          # fastapi/uvicorn; already in the training image
python -m miles.dashboard.serve --dump-details /path/to/dump [--follow] [--port 7788]
```

Works on any machine that can see the directory (login node over NFS, the
training node itself); `--follow` tails a still-running job. Typical remote
usage is an SSH port-forward. Runs recorded *without* `--use-miles-dashboard`
still get the full training-dynamics views (metrics fall back to dump-derived
aggregates under the `dump/` namespace); only the timeline is absent.

Views:

- **metrics** — every logged metric plus `dump/*` per-step aggregates; click
  a rollout-axis point to drill into that step.
- **timeline** — per-GPU lanes: phase band (rollout / train / update_weights,
  hatched `train_wait` idle), NVML utilization, sglang engine overlay
  (running requests, throughput, KV usage), bubble strip with click-to-zoom.
- **step drill-down** — per-trajectory table and scatter, GRPO group
  degeneracy (`zero_std`), generation-time columns, eval tab.
- **token view** — per-token importance ratio / entropy / advantage strips
  with loss-masked regions dimmed.

## Develop

```bash
python -m miles.dashboard.serve --demo    # generated demo data, no cluster needed
python -m pytest tests/fast/dashboard/ -q
MILES_DASHBOARD_REALDATA_DIR=/path/to/real/dump python -m pytest tests/fast/dashboard/ -q
```

Design and implementation-plan documents live with the dashboard workspace
(`miles-dashboard-design.md`); the short version of the architecture:

```
producers (Timer sinks / rollout hooks / NVML samplers / sglang scraper)
    -> DashboardCollector (named actor, driver node)  [writes JSONL streams]
dump_details .pt files (existing dump path)           [written by training]
    -> serve.py: MetricStore + DumpReader -> FastAPI -> static SPA
```
