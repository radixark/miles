---
title: Miles Dashboard
description: Design and usage of the built-in dashboard for training dynamics and compute efficiency.
---

The miles dashboard is a self-hosted web UI for inspecting a run's training dynamics and
compute efficiency. It answers two classes of question that stdout and wandb do not cover
well: what every GPU was doing during a given step, and what an individual trajectory
actually contained at the token level.

It reads files from disk and never connects to the training job, so it can be pointed at a
finished run, or tailed against a live one from a login node.

## Design

The dashboard draws on two independent data sources. Either one alone produces a usable
view, which matters because they are enabled by different flags.

### Live telemetry

`--use-miles-dashboard` starts a `DashboardCollector` as a named Ray actor pinned to the
driver node. Four kinds of producer push records to it:

| Producer | Stream | Fields |
|---|---|---|
| Phase sinks on the existing `Timer`, on every rank | `phases` | `name`, `t0`, `t1`, `node`, `gpus`, `rank`, `role` |
| One NVML sampler actor per GPU node | `gpu_util` | `ts`, `node`, `gpu`, `util`, `mem_mb`, `power_w` |
| One NVML sampler actor per GPU node | `gpu_processes` | `ts`, `node`, `gpu`, `pid`, `name`, `mem_mb` |
| sglang scraper thread | `engine_series` | `ts`, `addr`, `metric`, `labels`, `value` |
| sglang scraper thread | `topology` | Per engine `addr`, `worker_type`, `engine_rank`, `gpus`, `gpu_uuids` |
| Rollout manager hooks | `trajectory` | `ts`, `kind`, `sample_index`, `group_index`, `turn`, `weight_version`, `detail` |
| Rollout manager hooks | `data_buffer` | `ts`, `length` (queued sample count) |
| The tracking backend | `metrics` | `ts`, `step_key`, `step`, and the metric dictionary |

The collector buffers these and appends them to JSONL streams under
`{dump-details}/dashboard/` on a flush cadence. It can also forward a latest-value
snapshot to the Prometheus collector for external Grafana.

The phase names the timeline knows how to colour are `initialize`, `rollout`,
`eval_rollout`, `actor_train`, `train_log_probs`, `log_probs`, `ref_log_probs`,
`data_preprocess`, `train_wait`, `update_weights`, `ref_model_update`, `save_model`,
`sleep` and `wake_up`. Anything else the `Timer` emits still appears, in a neutral colour.

The scraped sglang whitelist covers queue and throughput gauges
(`sglang_num_running_reqs`, `sglang_num_queue_reqs`, `sglang_gen_throughput`,
`sglang_token_usage`, `sglang_cache_hit_rate`), cumulative token and request counters,
the latency histograms (time to first token, inter token latency, time per output token,
end to end request latency), and the PD disaggregation queue and KV transfer families,
which are simply absent when PD is off. Override it with
`--dashboard-sglang-metrics` when you need something outside that set.

Three properties of this path are worth knowing because they determine what happens when
something goes wrong:

* **Producers are fire and forget.** Nothing on the training path waits on the collector.
  A collector that is slow, wedged, or dead does not affect training. Overhead on the
  training path is a few milliseconds per step.
* **The collector class is Ray free.** `backend.py` wraps it in the named actor and spawns
  the per-node samplers, so the collector itself only ever sees plain method calls. Every
  behavior in it is unit testable without a cluster.
* **Write failures are loud.** If the disk write fails, for example on a full disk or an
  NFS hiccup, the error is logged on every flush attempt rather than silently dropping
  telemetry.

### Training artifacts

`--dump-details` writes the per-step artifacts the trajectory views read, independently of
whether the collector is enabled:

| Path | Contents |
|---|---|
| `rollout_data/{rollout_id}.pt` | The full sample batch of one rollout step |
| `train_data/{rollout_id}_{rank}.pt` | That rank's data-parallel shard, with per-token tensors and a `sample_indices` map back to `Sample.index` |
| `dashboard_columns/` | A per-token column mirror, so the token view never has to load a whole `.pt` |
| `trajectory/` | A raw conversation sidecar, written for session and multi turn runs |

`DumpReader.load_joined()` reunites the rollout and train sides: every rollout sample plus,
where a train dump exists, its per-token training row, deduplicated across tensor-parallel
duplicate rank files.

### Read side

`serve.py` loads a `MetricStore` over the JSONL streams and a `DumpReader` over the dumps,
then wires both into a FastAPI app that serves a static single page application. The server
is strictly read only over files on disk. Live viewing is the same application with a
follow loop tailing the store every two seconds.

```
producers (Timer sinks, rollout hooks, NVML samplers, sglang scraper)
    -> DashboardCollector (named actor on the driver node)   -> JSONL streams
dump .pt + dashboard_columns/*.parquet + trajectory/*.jsonl  -> written by training
    -> serve.py: MetricStore + DumpReader -> FastAPI -> static SPA
```

### Why the storage layout looks the way it does

Every stream is append only. That single constraint is what makes `follow()` a plain byte
offset tail and makes concurrent reads from request handlers safe without locking: a reader
may miss the newest records, but it can never see a torn one.

The two high rate streams, `gpu_util` and `engine_series`, are held in memory as columnar
polars frames rather than lists of dataclasses. This costs roughly 16 bytes per row instead
of roughly 600, and allows vectorized parsing and numpy queries. Those two streams plus
`phases` are written as hourly partition files, `{stream}/{YYYYMMDD_HH}.jsonl`, and parsed
lazily, so opening a long run does not require reading its entire history.

### Reading a run that is still being written

Two layers keep a live run from looking like a corrupt one. `DumpReader.rollout_ids()`
hides dump files younger than ten seconds unless the train companion already exists, and a
`torch.load` failure on a fresh file raises `DumpStillWriting`, which the server maps to
HTTP 503 so the client retries. Other failures map to conventional statuses: a missing file
or key returns 404, and a bad argument returns 400.

## Collecting telemetry

Add both flags to the training command. `--use-miles-dashboard` asserts that
`--dump-details` is set, because the telemetry lives under that directory and the
trajectory views read the dumps.

```bash
python3 train.py ... \
    --dump-details /path/to/dump \
    --use-miles-dashboard \
    --use-rollout-entropy
```

`--use-rollout-entropy` is optional. Without it the run still records everything else, and
the launcher logs a warning that per token entropy will be missing from the token view.

Cadence and scope can be tuned, though the defaults are appropriate for most runs:

| Flag | Default | Purpose |
|---|---|---|
| `--dashboard-flush-interval` | `5.0` | Collector disk flush cadence, in seconds |
| `--dashboard-gpu-sample-interval` | `1.0` | NVML sampling cadence, in seconds |
| `--dashboard-sglang-scrape-interval` | `2.0` | Engine scrape cadence, in seconds |
| `--dashboard-sglang-scrape-mode` | `auto` | `auto` scrapes `{router}/engine_metrics`, or each engine's `/metrics` under `--use-miles-router`. `router` and `direct` force one or the other |
| `--dashboard-sglang-metrics` | whitelist | Comma separated override of the scraped sglang metric whitelist |
| `--dashboard-forward-prometheus` | off | Also push dashboard gauges to the `--use-prometheus` collector for external Grafana |

A curated subset of the run's arguments, including the wandb identifiers, the parallelism
layout, and the key sglang settings, is persisted into `meta.json` for the dashboard header.

## Viewing

The three runtime dependencies (`fastapi`, `uvicorn`, `polars`) are already present in the
training image. To view from a machine that does not have them, install those three.

```bash
python -m miles.dashboard.serve --dump-details /path/to/dump
```

Then open `http://localhost:7788`. Any machine that can see the directory will do, whether
that is a login node over NFS or the training node itself. For a remote run, forward the
port over SSH:

```bash
ssh -L 7788:localhost:7788 <training-or-login-node>
```

| Flag | Default | Purpose |
|---|---|---|
| `--dump-details` | required | The run's `--dump-details` directory |
| `--follow` | off | Tail the telemetry streams of a still running job |
| `--port` | `7788` | Listen port |
| `--host` | `0.0.0.0` | Listen address |
| `--tensor-lru` | `2` | Rollout steps kept resident in tensor memory |
| `--cache-dir` | `<dump>/dashboard/cache` | Summary cache directory |
| `--use-utilization-overview` | auto | Always show the fleet overview instead of the per rank carpet. Enabled automatically above 64 lanes |
| `--demo` | off | Serve generated demo data, which needs a repository checkout |
## Views

Three tabs, in the order you normally walk them: **Metrics** for whether the run is learning,
**Compute Utilization** for whether the cluster is busy, and **Rollouts** for what the model
actually did. The screenshots below all come from one real run — GLM-5.2 744B on
terminal-bench-2, 100 steps over 11 hours, 32 training GPUs and 32 engine GPUs in a
disaggregated (non-colocated) layout, eight samples per prompt.

### Metrics

![The Metrics view, rollout category](/assets/images/dashboard-metrics-rollout.png)

A wandb-style category sidebar over every logged metric. Categories are just key prefixes, so
`rollout/`, `perf/`, `train/` and `eval/` appear when the run logs them and are absent when it
does not — the run above has no `eval` category because it ran no evaluations. The filter box
narrows within a category, and every chart shares the same x-axis, `rollout/step`.

This is the tab that answers "is it learning". In the screenshot `rollout/raw_reward` climbs
from roughly 0.3 to 0.9 across the hundred steps, which is the headline for this run.
`rollout/prefix_cache_hit_rate` drifting down from 0.96 to 0.94 over the same window is the
kind of second-order detail the view is good at surfacing next to it.

Metric keys from `metrics.jsonl` are served as recorded. Per-step aggregates derived from the
dumps are namespaced under `dump/`, which is what allows this view to work for runs where the
collector was never enabled: `dump/reward_mean`, `dump/reward_std`,
`dump/response_length_mean`, `dump/truncated_frac`, `dump/zero_std_group_frac`,
`dump/mean_abs_lp_diff`, `dump/mean_entropy` and `dump/mixed_version_frac`.

Two of those are worth calling out because nothing else reports them per step:
`dump/zero_std_group_frac` is the fraction of GRPO groups whose reward standard deviation
collapsed to zero, so a degenerating run is visible as that fraction climbing, and
`dump/mixed_version_frac` is the fraction of samples that spanned more than one weight
version, which is the staleness signal that matters in async runs.

![The Metrics view, sglang category](/assets/images/dashboard-metrics-sglang.png)

The `sglang` category is different in three ways and it is worth knowing which. Its series come
from the engine scrape rather than from `metrics.jsonl`; its x-axis is **wall clock**, not
`rollout/step`, because engines are sampled on their own cadence; and it gains an **Engines**
legend on the right with one checkbox per engine. Unchecking an engine hides it everywhere on
the page, including from the y-scale, which is how you stop one outlier engine from flattening
every other line.

The four coloured series above are the run's four inference engines. The PD-disaggregation
families (`sglang_num_decode_prealloc_queue_reqs` and friends) are flat zero here because this
run did not use prefill/decode disaggregation — those charts are present but empty rather than
hidden, so their absence is legible.

### Compute Utilization

![The Compute Utilization view](/assets/images/dashboard-compute-utilization.png)

The densest view, and the one that pays off most on a run that is slower than it should be. It
stacks three things, top to bottom.

**Fleet overview.** One scale-invariant summary of all 64 lanes: phase composition on top, and
a utilization p10–p90 band with median and minimum below. This is what you read first, because
it is the only element whose shape does not change with cluster size.

**Wait ratio per step.** One tile per training step, shaded by how much of that step went to
`train_wait`. In the screenshot the first handful of steps are visibly darker than the rest —
early steps waiting on rollout while the pipeline fills. Scanning this strip is the fastest way
to find the step worth zooming into.

**Per-lane detail.** Below 64 GPUs, one lane per GPU, and each lane stacks four things:

* **Phase band.** Which phase that rank was in, at that moment. Because the band is per rank
  rather than per run, a straggler shows up as one lane whose `actor_train` starts late, and a
  rank stuck in `train_wait` while its peers compute is visible directly.
* **NVML utilization and memory**, sampled once per second by default, so a phase that holds the
  GPU without using it is distinguishable from one that is genuinely busy.
* **An sglang overlay**, selectable between `sglang_num_running_reqs`, `sglang_gen_throughput`,
  `sglang_token_usage` and `sglang_cache_hit_rate`, drawn against the same time axis as the
  phases. This is what connects a rollout that ran long to the engine state at the time, for
  example concurrency collapsing or KV cache saturating.
* **A request lifecycle strip**, coloured by whether each request was queued, generating, or
  waiting on a tool call, which separates slow generation from time spent outside the model.

The screenshot shows a disaggregated run, and the two roles are immediately distinguishable:
lanes `g0`–`g24` are training GPUs, with blue `actor_train` bands and a sawtooth utilization
trace that drops between steps; lanes `g32`–`g56` are engine GPUs, with orange `rollout` markers
and the orange engine overlay riding on top of a much noisier utilization trace. The green band
at the right edge of every training lane is `save_model` — the checkpoint written at the end of
the run. On a colocated run both patterns would share the same lanes instead.

Typical questions it answers: which rank is late into a weight update, whether a phase boundary
lines up with a utilization dip, whether rollout and training actually overlap in an async run,
and how much of a step went to `train_wait`.

`gpu_processes` samples additionally record which PIDs hold memory on each GPU, which is how a
colocated run shows the trainer and the engine sharing a device.

Above 64 GPUs a per-lane rendering stops being readable, so the view switches to the fleet
overview alone. Lanes are selected with a small grammar (`g:`, `rank:`, `node:`, `every:`)
alongside outlier quick picks — `pick: lowest util` and `pick: slowest update_weights` — so a
specific subset can still be brought up on a large cluster. The eight lanes in the screenshot
are the spaced default, `g:0` through `g:56`, which the view seeds on first sight of the
topology.

This view also carries a configuration advisory panel, which compares what the engines actually
did against what the run was configured to allow:

| Trigger | Suggestion |
|---|---|
| Peak `sglang_num_running_reqs` stayed below 30% of `--sglang-max-running-requests` | Lower it, and under `--colocate` note that this frees memory for training |
| Mean `sglang_cache_hit_rate` below 10%, non colocated runs only | Raise `--sglang-mem-fraction-static` for a bigger KV cache |
| Mean `sglang_token_usage` above 95% | Warns that KV cache is the throughput bottleneck; more GPUs or a smaller rollout batch |

These are heuristics rather than a guarantee, and the thresholds are expected to be tuned as
real runs surface false positives and negatives. The panel is empty when no sglang series was
scraped, since it has nothing to compare against.

### Rollouts

![The Rollouts view for one training step](/assets/images/dashboard-rollout-step.png)

One training step at a time, reached by step number and walked with Prev/Next. The header tiles
summarise the batch before you look at anything else: sample count, reward mean, truncated
fraction, how many GRPO groups collapsed to zero reward standard deviation, mixed-version
fraction, average staleness, and — when train dumps are present — mean absolute log-prob
difference and mean entropy. A tile reading `—` means that column is absent from this run's
dumps rather than zero.

For the step above: 64 samples, reward mean 0.844, nothing truncated, and 6 of 8 groups with
zero reward std.

**Batch anatomy** is the top panel, and it is the one to read first on an agentic run. One row
per sample, drawn on wall-clock time, with three colours: orange while the model is generating
(the hue steps with each weight version, so staleness is visible as a colour change mid-row),
green while the sample is blocked on a tool call, and pale grey while it is queued or retrying.
A vertical marker shows when the batch was consumed by the trainer. Sort by submit order,
staleness, wall span, reward or turns — sorting by wall span puts the long tail at one edge,
which is usually the thing you opened the view to find. In the screenshot the green tool-wait
segments dominate, which is the expected shape for a terminal-agent task where most of the wall
clock is spent running shell commands rather than generating tokens.

Below it, a scatter of reward against response length, with truncated samples in red, and then
the per-sample table: sample and group index, both raw and shaped reward, response length,
truncation flag, turn and tool-call counts, and the per-token statistics when train dumps exist.
Click any row to open the sample view. The reward axis in the screenshot is binary — every
sample scored exactly 0 or 1 — which is what a pass/fail task harness looks like here.

![The Rollouts view, Groups tab](/assets/images/dashboard-rollout-groups.png)

The **Groups** tab re-aggregates the same step by GRPO group. Rows whose reward standard
deviation collapsed to zero are drawn in red, because those groups contribute no gradient
signal at all: every sample in the group got the same reward, so the advantages vanish. Six of
the eight groups here are red — five where every sample succeeded and one where every sample
failed. That is the concrete form of the `6/8 zero-std groups` tile above, and a run where this
fraction climbs is a run whose effective batch size is shrinking.

### Sample view

![The sample view, conversation tab](/assets/images/dashboard-sample-conversation.png)

One sample, reached by clicking a row in the step table, with Prev/Next walking the other seven
samples of the same GRPO group — which is the comparison that matters, since those samples share
a prompt and differ only in sampling.

The lifecycle strip at the top is the same three-colour encoding as the batch anatomy, scoped to
this one sample. Below it, two tabs.

**Conversation** renders the turns as they were exchanged, with the status and reward as chips.
The screenshot shows the first two exchanges of a terminal-agent episode: the system prompt, the
task, the model's reasoning and its first shell command, the shell's reply, and the model's next
command. Reasoning blocks are set apart from the message body, so a run where the model reasons
at length but acts rarely is visible at a glance.

**Tokens** is the same sample at token granularity, with per-token log-probs, entropy and the
rollout-versus-train log-prob difference where the train dump supplies them. It loads a window
at a time rather than the whole sequence, so a 36k-token episode like this one opens without
pulling the entire `.pt`.

Two things about the token view are easy to misread. Training statistics exist only for
positions the loss covered, so prompt positions have text but no statistics — that is expected,
not missing data. And the rollout-versus-train log-prob difference is the true-on-policy check:
it should be near zero, and a systematically non-zero band is worth chasing.

## Runs recorded without the collector

A run that set `--dump-details` but not `--use-miles-dashboard` still gets the training
dynamics views, because those read the dumps. The timeline is the one view that is absent,
since it has no phase or GPU telemetry to draw, and the metrics view falls back to the
`dump/*` aggregates.

## Development

```bash
# generated demo data, no cluster needed
python -m miles.dashboard.serve --demo

python -m pytest tests/fast/dashboard/ -q

# run the same tests against a real dump
MILES_DASHBOARD_REALDATA_DIR=/path/to/real/dump python -m pytest tests/fast/dashboard/ -q
```

`--demo` builds its fixture with the dummy generators from the test suite, which are
deliberately not shipped in the wheel, so it requires a repository checkout.

The HTTP surface the SPA consumes is available to scripts as well, with the caveat that it
carries no compatibility guarantee. `/api/meta` describes the run, `/api/metrics` serves the
catalog and series, `/api/advisory` returns the configuration suggestions, the
`/api/timeline/*` family covers topology, phases, GPU samples, heatmap, fleet, outliers,
engine series, and bubbles, and the `/api/rollout/{rollout_id}/*` family covers per step
summaries, groups, trajectories, and per sample messages and tokens.
