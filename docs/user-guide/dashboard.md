---
title: Miles Dashboard
description: A self-hosted web UI for a run's training dynamics, compute efficiency, and per-token trajectories.
---

The miles dashboard is a self-hosted web UI for inspecting a run. It answers two kinds of
question that stdout and wandb do not cover well: what every GPU was doing during a given step,
and what an individual trajectory actually contained at the token level.

It reads files from disk and never connects to the training job. You can point it at a finished
run, or tail a live one from a login node. Nothing you do in the UI can affect training.

| If you are asking | Open |
|---|---|
| Is the run learning? Is reward moving, are responses growing? | [Metrics](#metrics) |
| Why is a step slow? Which rank is late, are the engines starved? | [Compute Utilization](#compute-utilization) |
| What did the model do on this batch? | [Rollouts](#rollouts) |
| Why did this one sample fail? What did it generate, token by token? | [Sample view](#sample-view) |

## What it shows

The screenshots below all come from the same run: GLM-5.2 744B on terminal-bench-2, 100 steps
over 11 hours, 32 training GPUs and 32 engine GPUs in a disaggregated (non-colocated) layout,
eight samples per prompt. Using one run throughout means the panels line up across figures.

### Metrics

![The Metrics view, rollout category](/assets/images/dashboard/metrics-rollout.png)

This is the tab that tells you whether the run is learning. It shows every logged metric, with a
wandb-style category sidebar on the left.

Categories are just key prefixes. `rollout/`, `perf/`, `train/` and `eval/` appear when the run
logs them, and are absent when it does not. The run above has no `eval` category because it ran
no evaluations. The filter box narrows the list within a category, and every chart uses the same
x-axis, `rollout/step`.

In the screenshot, `rollout/raw_reward` climbs from about 0.3 to about 0.9 over the hundred
steps. That is the headline for this run. `rollout/prefix_cache_hit_rate` drifting down from 0.96
to 0.94 over the same window is the kind of second-order detail this view is good at putting
right next to it.

Metric keys from `metrics.jsonl` are served as recorded. Per-step aggregates computed from the
dumps are namespaced under `dump/`, which is what lets this view work for runs where the
collector was never enabled: `dump/reward_mean`, `dump/reward_std`,
`dump/response_length_mean`, `dump/truncated_frac`, `dump/zero_std_group_frac`,
`dump/mean_abs_lp_diff`, `dump/mean_entropy` and `dump/mixed_version_frac`.

Two of those are worth calling out, because nothing else reports them per step.
`dump/zero_std_group_frac` is the fraction of GRPO groups whose reward standard deviation
collapsed to zero. If that fraction climbs, the run is degenerating.
`dump/mixed_version_frac` is the fraction of samples that spanned more than one weight version,
which is the staleness signal that matters in async runs.

The `perf` category holds the per step throughput series, including `perf/actor_train_mfu`
and the `perf/mfu_peak_tflops` it was divided by. Those two are covered in
[Model FLOPs utilization](#model-flops-utilization) below, which is worth reading before
comparing the number against anything published.

![The Metrics view, sglang category](/assets/images/dashboard/metrics-sglang.png)

The `sglang` category behaves differently from the other three, in ways that will confuse you if
you do not expect them.

1. Its data comes from the engine scrape, not from `metrics.jsonl`.
2. Its x-axis is wall clock, not `rollout/step`, because engines are sampled on their own
   schedule rather than once per training step.
3. It adds an Engines legend with one checkbox per engine. Unchecking an engine hides it
   everywhere on the page, including from the y-axis scale. That is how you stop one outlier
   engine from flattening every other line.

The four coloured series above are this run's four inference engines. The PD-disaggregation
charts (`sglang_num_decode_prealloc_queue_reqs` and the others like it) are flat zero because
this run did not use prefill/decode disaggregation. They are drawn empty rather than hidden, so
you can tell the difference between "no data" and "zero".

### Compute Utilization

![The Compute Utilization view](/assets/images/dashboard/compute-utilization.png)

This is the most detailed view, and the one to open when a run is slower than it should be. It
has three parts, top to bottom.

**Fleet overview.** A single summary of all lanes: phase composition on top, and a utilization
band (p10 to p90, with median and minimum) below. Read this first. Its shape does not change with
cluster size, so it works the same on 8 GPUs as on 800.

**Wait ratio per step.** One tile per training step, shaded by how much of that step went to
`train_wait`. In the screenshot the first few steps are darker than the rest, because early steps
wait on rollout while the pipeline fills. Scanning this strip is the fastest way to find the step
worth zooming into.

**Per-lane detail.** Below 64 GPUs you get one lane per GPU. Each lane stacks four things:

* **Phase band.** Which phase that rank was in at that moment. The band is per rank rather than
  per run, so a straggler shows up as one lane whose `actor_train` starts late, and a rank stuck
  in `train_wait` while its peers compute is visible directly.
* **NVML utilization and memory**, sampled once per second by default. This lets you tell a phase
  that holds the GPU without using it apart from one that is genuinely busy.
* **An sglang overlay**, on the same time axis as the phases. You can switch it between
  `sglang_num_running_reqs`, `sglang_gen_throughput`, `sglang_token_usage` and
  `sglang_cache_hit_rate`. This is what connects a rollout that ran long to what the engines were
  doing at the time, such as concurrency collapsing or the KV cache filling up.
* **A request lifecycle strip**, coloured by whether each request was queued, generating, or
  waiting on a tool call. This separates slow generation from time spent outside the model.

The screenshot is a disaggregated run, so the two roles look different at a glance. Lanes `g0`
through `g24` are training GPUs: blue `actor_train` bands, and a sawtooth utilization trace that
drops between steps. Lanes `g32` through `g56` are engine GPUs: orange `rollout` markers, the
engine overlay on top, and a much noisier utilization trace. The green band at the right edge of
every training lane is `save_model`, the checkpoint written at the end of the run. On a colocated
run both patterns appear on the same lanes instead.

Above 64 GPUs, one lane per GPU stops being readable, so the view shows only the fleet overview.
You can still bring up a specific subset with the lane selection grammar (`g:`, `rank:`, `node:`,
`every:`) or with the two quick picks, `pick: lowest util` and `pick: slowest update_weights`.
The eight lanes in the screenshot are the evenly spaced default the view picks on its own.

`gpu_processes` samples also record which PIDs hold memory on each GPU. That is how a colocated
run shows the trainer and the engine sharing a device.

This view also carries a configuration advisory panel. It compares what the engines actually did
against what the run was configured to allow:

| Trigger | Suggestion |
|---|---|
| Peak `sglang_num_running_reqs` stayed below 30% of `--sglang-max-running-requests` | Lower it. Under `--colocate` this also frees memory for training |
| Mean `sglang_cache_hit_rate` below 10%, on non-colocated runs only | Raise `--sglang-mem-fraction-static` for a bigger KV cache |
| Mean `sglang_token_usage` above 95% | KV cache is the throughput bottleneck. Add GPUs or use a smaller rollout batch |
| Mean `perf/actor_train_mfu` below `--low-mfu-threshold`, default 15%, excluding the first step | The training step itself is computing slowly. See the caveats below, and set the threshold to suit the run |

These are heuristics, not guarantees, and the thresholds will be tuned as real runs surface false
positives and negatives. The panel is empty when no sglang series was scraped, since it has
nothing to compare against, except for the MFU rule, which reads the metrics stream and
therefore stands on its own.

### Model FLOPs utilization

The Compute Utilization view carries an MFU tile fed by two keys the trainer logs per step:

```
perf/actor_train_mfu   = perf/actor_train_tflops / perf/mfu_peak_tflops
perf/actor_train_tflops = 3 x forward FLOPs of the model / training world size / actor train time
```

The denominator is published alongside the ratio on purpose. A percentage whose peak is not
stated cannot be checked by the person reading it, so the tile shows `25.3% of 989 TFLOP/s`
rather than `25.3%`.

`perf/mfu_peak_tflops` comes from a small device table in `miles/utils/device_flops.py`, keyed
on the words of `torch.cuda.get_device_name()`, and `--mfu-peak-tflops` overrides it for a
device the table does not know or to report against a different precision's peak. The table
holds **dense** BF16 figures. Vendor datasheets headline the 2:4-sparsity number, which is
exactly twice the dense one, so extending the table from the headline would halve every MFU
reported: an H100 SXM is 989 TFLOP/s dense and 1979 with sparsity, and every published number
this would be compared against uses dense. A board variant gets its own row only when it
clocks differently from its family's flagship, which today means H100 PCIe at 756 against the
SXM's 989; H100 NVL and H200 NVL match their SXM siblings and resolve through the family row.
When neither the table nor the override yields a peak, both keys are simply not logged, so a
run with an unrecognised device shows no tile at all rather than a percentage against an
assumed denominator.

#### Reading the number

This is model FLOPs utilization, not hardware FLOPs utilization. The `3x` counts one forward
and one backward of the model, which is the work the model required; it does not count
activation recompute, which is work the hardware did that the model did not require. Three
things therefore lower the number legitimately, and none of them are bugs:

* **Activation recompute** makes the hardware perform roughly four forward passes where the
  model needed three.
* **The optimizer step** sits inside `actor_train_time` and contributes no model FLOPs. With
  `--optimizer-cpu-offload` this is the dominant term, because the adam step moves the whole
  optimizer state across PCIe.
* **Heads the FLOPs model does not cover.** `calculate_fwd_flops` models the base model only,
  so a run with `--enable-mtp-training` does real work that never reaches the numerator. This
  affects the pre-existing `perf/actor_train_tflops` in the same way.

The first step of a run is always an outlier, because it carries kernel autotuning and
compilation. Measured below: 9.4% against a 25% steady state on a dense run, 3.8% against 5.4%
on an MoE one. The advisory rule drops step 0 for this reason, and so should you when reading
the curve.

#### What the numbers look like

Measured on 8xH200, three rollout steps each, all with gradient checkpointing:

| | colocate | fully-async |
|---|---|---|
| Qwen3-4B dense, FSDP | 25.3 / 24.6% | 26.0 / 26.2% |
| Qwen3.5-35B-A3B MoE, expert parallel, `--optimizer-cpu-offload`, MTP | 5.0 / 5.8% | 5.4% |

Two things to take from this. First, both rows are healthy runs and they differ five-fold, so
no single absolute threshold separates "slow by configuration" from "slow because something
broke". Long-context MLA training is a third point on that spread: a well-tuned Kimi-K2.5 run
on GB300 tops out around 17 to 18%, close enough to the 15% default that a slightly different
config would trip it. That is why the threshold is `--low-mfu-threshold` rather than a
constant: set it from what your own run does in a healthy step, or pass `0` to turn the rule
off and read the tile directly.

Second, and more useful day to day: **MFU barely moves between colocate and fully-async while
`perf/wait_time_ratio` collapses**, from 0.708 to 0.060 on the dense run and from 0.760 to
0.027 on the MoE one. The two metrics answer different questions and both are needed. MFU
asks how efficiently the training step computes, and fully-async does not change that because
the same kernels do the same work. `wait_time_ratio` asks how much of the step was spent
waiting for rollout data, which is exactly what fully-async removes. A single blended
end-to-end utilization number would have read as "async made training three times more
efficient", which is false.

Because `actor_train_time` is the only denominator, MFU is immune to rollout starvation: a
pipeline stalled on data shows a healthy MFU and a `wait_time_ratio` near one. That is the
signature to look for when a run is slow but the training step is not the reason.

### Rollouts

![The Rollouts view for one training step](/assets/images/dashboard/rollout-step.png)

This view shows one training step at a time. You reach a step by number and walk between steps
with Prev and Next.

The header tiles summarise the batch: sample count, reward mean, truncated fraction, how many
GRPO groups collapsed to zero reward standard deviation, mixed-version fraction, average
staleness, and, when train dumps are present, mean absolute log-prob difference and mean entropy.
A tile showing `—` means that column is absent from this run's dumps. It does not mean zero.

The step above has 64 samples, a reward mean of 0.844, nothing truncated, and 6 of 8 groups with
zero reward std.

**Batch anatomy** is the top panel, and on an agentic run it is the one to read first. Each row
is one sample, drawn on wall-clock time, in three colours:

* orange while the model is generating. The shade changes with each weight version, so you can
  see staleness as a colour change partway through a row.
* green while the sample is blocked on a tool call.
* pale grey while it is queued or retrying.

A vertical marker shows when the trainer consumed the batch. You can sort by submit order,
staleness, wall span, reward or turns. Sorting by wall span moves the long tail to one edge,
which is usually what you opened the view to find. In the screenshot the green tool-wait segments
dominate. That is the expected shape for a terminal-agent task, where most of the wall clock goes
to running shell commands rather than to generating tokens.

Below that is a scatter of reward against response length, with truncated samples in red, and
then the per-sample table: sample and group index, raw and shaped reward, response length,
truncation flag, turn and tool-call counts, and the per-token statistics when train dumps exist.
Click any row to open the sample view. The reward axis in the screenshot only has values at 0 and
1, because this task harness scores pass or fail with nothing in between.

![The Rollouts view, Groups tab](/assets/images/dashboard/rollout-groups.png)

The Groups tab re-aggregates the same step by GRPO group. Rows whose reward standard deviation
collapsed to zero are drawn in red, because those groups contribute no gradient signal at all:
every sample in the group got the same reward, so the advantages cancel out.

Six of the eight groups here are red. Five are groups where every sample succeeded, one is a
group where every sample failed. This is the detail behind the `6/8 zero-std groups` tile above.
If that fraction climbs over a run, the effective batch size is shrinking.

### Sample view

![The sample view, conversation tab](/assets/images/dashboard/sample-conversation.png)

This view shows one sample. You reach it by clicking a row in the step table. Prev and Next walk
the other seven samples of the same GRPO group, which is the useful comparison: those samples
share a prompt and differ only in sampling.

The strip at the top uses the same three colours as the batch anatomy, for this one sample.
Below it are two tabs.

**Conversation** renders the turns as they were exchanged, with the status and reward shown as
chips. The screenshot shows the first two exchanges of a terminal-agent episode: the system
prompt, the task, the model's reasoning and its first shell command, the shell's reply, and the
model's next command. Reasoning blocks are styled apart from the message body, so you can see at
a glance whether the model is reasoning at length but rarely acting.

**Tokens** shows the same sample at token granularity, with per-token log-probs, entropy, and the
difference between rollout and train log-probs where the train dump supplies them. It loads one
window at a time rather than the whole sequence, so a 36k-token episode like this one opens
without reading the entire `.pt`.

Two things here are easy to misread:

* Training statistics only exist for positions the loss covered. Prompt positions have text but
  no statistics. That is expected, not missing data.
* The difference between rollout and train log-probs is the true-on-policy check. It should be
  near zero. A band that is consistently non-zero is worth chasing.
* Positions the loss ignores, such as tool output in an agentic session, have no rollout
  log-prob because the engine never generated them. Both log-prob sides read zero there, so
  `lp_diff` is zero and `imp_ratio` is one by construction: judge train and rollout agreement
  on the loss-covered tokens only.

## Turning it on

Add both flags to the training command. `--use-miles-dashboard` requires `--dump-details`,
because the telemetry is written under that directory and the trajectory views read the dumps.

```bash
python3 train.py ... \
    --dump-details /path/to/dump \
    --use-miles-dashboard \
    --use-rollout-entropy
```

`--use-rollout-entropy` is optional. Without it the run still records everything else, and the
launcher warns that per-token entropy will be missing from the token view.

Cadence and scope can be tuned, though the defaults suit most runs:

| Flag | Default | Purpose |
|---|---|---|
| `--dashboard-flush-interval` | `5.0` | Collector disk flush cadence, in seconds |
| `--dashboard-gpu-sample-interval` | `1.0` | NVML sampling cadence, in seconds |
| `--dashboard-sglang-scrape-interval` | `2.0` | Engine scrape cadence, in seconds |
| `--dashboard-sglang-scrape-mode` | `auto` | `auto` scrapes `{router}/engine_metrics`, or each engine's `/metrics` under `--use-miles-router`. `router` and `direct` force one or the other |
| `--dashboard-sglang-metrics` | whitelist | Comma-separated override of the scraped sglang metric whitelist |
| `--dashboard-forward-prometheus` | off | Also push dashboard gauges to the `--use-prometheus` collector for external Grafana. No-op under `--cluster-backend kubernetes`, where that collector is not started |

A curated subset of the run's arguments is persisted into `meta.json` for the dashboard header:
the wandb identifiers, the parallelism layout, and the key sglang settings.

## Viewing a run

The three runtime dependencies (`fastapi`, `uvicorn`, `polars`) are already in the training
image. To view from a machine that does not have them, install those three.

```bash
python -m miles.dashboard.serve --dump-details /path/to/dump
```

Then open `http://localhost:7788`. Any machine that can see the directory will do, whether that
is a login node over NFS or the training node itself. For a remote run, forward the port over
SSH:

```bash
ssh -L 7788:localhost:7788 <training-or-login-node>
```

| Flag | Default | Purpose |
|---|---|---|
| `--dump-details` | required | The run's `--dump-details` directory |
| `--follow` | off | Tail the telemetry streams of a still-running job |
| `--port` | `7788` | Listen port |
| `--host` | `0.0.0.0` | Listen address |
| `--tensor-lru` | `2` | Rollout steps kept resident in tensor memory |
| `--cache-dir` | `<dump>/dashboard/cache` | Summary cache directory |
| `--use-utilization-overview` | auto | Always show the fleet overview instead of the per-rank carpet. Turns on automatically above 64 lanes |
| `--low-mfu-threshold` | `0.15` | Fraction below which the MFU advisory fires. `0` turns the rule off |
| `--demo` | off | Serve generated demo data, which needs a repository checkout |

Two notes if you are opening someone else's run. Leave `--follow` off for a finished run: the
static read is faster, and the follow loop has nothing to tail. And the server writes parquet
summary caches under `--cache-dir`, which defaults to a path inside the dump directory. When you
do not own the dump, point `--cache-dir` somewhere you can write:

```bash
python -m miles.dashboard.serve \
    --dump-details /shared/someone-elses-run/dump_details \
    --cache-dir ~/dash-cache --port 7803
```

## How it works

The dashboard draws on two independent data sources. Either one alone produces a usable view,
which matters because they are enabled by different flags.

```
producers (Timer sinks, rollout hooks, NVML samplers, sglang scraper)
    -> DashboardCollector (named actor on the driver node)   -> JSONL streams
dump .pt + dashboard_columns/*.parquet + trajectory/*.jsonl  -> written by training
    -> serve.py: MetricStore + DumpReader -> FastAPI -> static SPA
```

### Live telemetry

`--use-miles-dashboard` starts a `DashboardCollector` as a named Ray actor pinned to the driver
node. Four kinds of producer push records to it:

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

The collector buffers these and appends them to JSONL streams under `{dump-details}/dashboard/`
on a flush cadence. It can also forward a latest-value snapshot to the Prometheus collector for
external Grafana.

The phase names the timeline knows how to colour are `initialize`, `rollout`, `eval_rollout`,
`actor_train`, `train_log_probs`, `log_probs`, `ref_log_probs`, `data_preprocess`, `train_wait`,
`update_weights`, `ref_model_update`, `save_model`, `sleep` and `wake_up`. Anything else the
`Timer` emits still appears, in a neutral colour.

The scraped sglang whitelist covers queue and throughput gauges (`sglang_num_running_reqs`,
`sglang_num_queue_reqs`, `sglang_gen_throughput`, `sglang_token_usage`,
`sglang_cache_hit_rate`), cumulative token and request counters, the latency histograms (time to
first token, inter-token latency, time per output token, end-to-end request latency), and the PD
disaggregation queue and KV transfer families, which are simply absent when PD is off. Override
it with `--dashboard-sglang-metrics` when you need something outside that set.

Three properties of this path decide what happens when something goes wrong:

* **Producers are fire and forget.** Nothing on the training path waits on the collector. A
  collector that is slow, wedged or dead does not affect training. Overhead on the training path
  is a few milliseconds per step.
* **The collector class is Ray-free.** `backend.py` wraps it in the named actor and spawns the
  per-node samplers, so the collector itself only ever sees plain method calls. Every behavior in
  it is unit-testable without a cluster.
* **Write failures are loud.** If the disk write fails, on a full disk or an NFS hiccup, the
  error is logged on every flush attempt rather than silently dropping telemetry.

### Training artifacts

`--dump-details` writes the per-step artifacts the trajectory views read, whether or not the
collector is enabled:

| Path | Contents |
|---|---|
| `rollout_data/{rollout_id}.pt` | The full sample batch of one rollout step |
| `train_data/{rollout_id}_{rank}.pt` | That rank's data-parallel shard, with per-token tensors and a `sample_indices` map back to `Sample.index` |
| `dashboard_columns/` | A per-token column mirror, so the token view never has to load a whole `.pt` |
| `trajectory/` | A raw conversation sidecar, written for session and multi-turn runs |

`DumpReader.load_joined()` reunites the rollout and train sides: every rollout sample plus, where
a train dump exists, its per-token training row, deduplicated across tensor-parallel duplicate
rank files.

### Read side

`serve.py` loads a `MetricStore` over the JSONL streams and a `DumpReader` over the dumps, then
wires both into a FastAPI app that serves a static single-page application. The server is
strictly read-only over files on disk. Live viewing is the same application with a follow loop
tailing the store every two seconds.

### Why the storage layout looks the way it does

Every stream is append-only. That single constraint is what makes `follow()` a plain byte-offset
tail, and what makes concurrent reads from request handlers safe without locking: a reader may
miss the newest records, but it can never see a torn one.

The two high-rate streams, `gpu_util` and `engine_series`, are held in memory as columnar polars
frames rather than as lists of dataclasses. That costs about 16 bytes per row instead of about
600, and allows vectorized parsing and numpy queries. Those two streams plus `phases` are written
as hourly partition files, `{stream}/{YYYYMMDD_HH}.jsonl`, and parsed lazily, so opening a long
run does not require reading its entire history.

### Reading a run that is still being written

Two layers keep a live run from looking like a corrupt one. `DumpReader.rollout_ids()` hides dump
files younger than ten seconds unless the train companion already exists, and a `torch.load`
failure on a fresh file raises `DumpStillWriting`, which the server maps to HTTP 503 so the
client retries. Other failures map to conventional statuses: a missing file or key returns 404,
and a bad argument returns 400.

## Runs recorded without the collector

A run that set `--dump-details` but not `--use-miles-dashboard` still gets the training dynamics
views, because those read the dumps. Compute Utilization is the one view that is missing, since
it has no phase or GPU telemetry to draw, and Metrics falls back to the `dump/*` aggregates.

## Development

```bash
# generated demo data, no cluster needed
python -m miles.dashboard.serve --demo

python -m pytest tests/fast/dashboard/ -q

# run the same tests against a real dump
MILES_DASHBOARD_REALDATA_DIR=/path/to/real/dump python -m pytest tests/fast/dashboard/ -q
```

`--demo` builds its fixture with the dummy generators from the test suite, which are deliberately
not shipped in the wheel, so it needs a repository checkout.

The HTTP surface the SPA consumes is available to scripts as well, with the caveat that it
carries no compatibility guarantee. `/api/meta` describes the run, `/api/metrics` serves the
catalog and series, `/api/advisory` returns the configuration suggestions, the `/api/timeline/*`
family covers topology, phases, GPU samples, heatmap, fleet, outliers, engine series and bubbles,
and the `/api/rollout/{rollout_id}/*` family covers per-step summaries, groups, trajectories, and
per-sample messages and tokens.
