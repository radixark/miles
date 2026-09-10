# Multi-LoRA DAPO pressure test

The experiment compares the GPU slot probe against the largest client count
that completes the configured DAPO workload. A passing count is qualified by
the number of completed steps, sequence limit, adapter targets and topology;
it is not a guarantee that every possible future workload fits.

This PR is based on Miles PR #2846 at
`3b97ddc308bf10581d803617aa42f75bf7b88831`, integrating PR #3170 at
`f159f9baa3cf9433c6b31d9f155bebdb721afe26`. The GPU environment uses SGLang
PR #38165 at `c27edd9949f079c07ab66091219865af454b5cad`.

## GPU E2E acceptance

`pressure_client.py` uses real Tinker SDK clients against `serve_tinker.py`.
Each client creates a separate LoRA model, publishes its sampler weights,
samples through SGLang, builds DAPO training data, and awaits both GPU
forward/backward and an Adam update. Subsequent steps publish the updated
weights and repeat the same loop. No fake backend is used in this path.

Clients synchronize before each step and after collecting their rollout batch.
A client that has finished sampling therefore waits for the slowest client
before submitting forward/backward. Phase files expose that wait.

All N clients must complete three full steps before N passes. The one-slot
memory probe, CPU unit tests and launcher snapshots do not satisfy that gate.
Checkpoint saving during continuous training is included; checkpoint restore
correctness is covered separately by the gateway tests, not by this pressure loop.
The earlier GPU source estimated 121 slots but has not completed qualification;
it is not evidence that this rebased PR passes GPU E2E.

## Workload

- Default: full Qwen3-30B-A3B BF16, matching `run_gateway.py`. Four nodes with
  eight H200 each: 16 training GPUs (TP2, DP8, EP8, expert TP1), and 16 rollout
  GPUs (eight TP2/EP2 engines).
- `--model-name glm5.2` selects the original 78-layer GLM-5.2 recipe: training
  TP16/EP16 and one rollout TP16/EP16 engine on the same 16 + 16 GPU split.
- Both recipes use PP1, CP1, packed sequences and full activation recomputation.
- LoRA rank 16, alpha 32, attention and per-expert MLP adapters.
- DAPO math with 8192 **total** prompt and response tokens; prompts at most 2048.
- Each client uses four accepted prompt groups, eight samples per prompt,
  accuracy-based dynamic sampling, reward standardization, response-token-mean
  loss, asymmetric PPO clipping `[0.8, 1.28]` and a 1024-token overlength buffer.
- Clients use a common step boundary and a common forward/backward boundary.
  Every client must complete three optimizer steps to pass a default trial.

The SDK loop uses the async submission/future pattern in this directory's
`client.py`, with the publish/sample/train cycle from the official
[minimal Tinker Cookbook RL recipe](https://github.com/thinking-machines-lab/tinker-cookbook/blob/1f962eda3a2cec8de284725f2adc9978e93dfcd3/tinker_cookbook/recipes/rl_loop.py).
It is adapted for DAPO, each model's tokenizer and concurrent tenants, using the
official `tinker==0.26.2` SDK and the Miles DAPO answer grader.

## Probe and startup

`--n-adapters auto` first starts one trainer slot without rollout engines.
The current Bridge model and LayerWise optimizer have fixed pools, so startup
must give them a positive count. The probe includes their preallocated CUDA
storages in the residency measurement. A zero-learning-rate warmup materializes
Adam state and shared runtime workspaces; a second maximum-size loss pass and
optimizer step through `MilesBackend` measures steady-state activation memory.
The global probe batch has one maximum-length row per data-parallel replica,
so DP > 1 also exercises every rank.
Inactive allocator cache is released
before reading resident memory. The precision-derived byte model is only a
cross-check.

`slot-capacity.json` records every rank's memory measurements. The smallest
rank capacity wins, with a default 2 GiB margin. An optional host budget caps
the count using two BF16 engine versions per slot; the engine registry is
configured to that same limit. The host budget is per engine process, not a
whole-node RAM budget shared by all TP processes.

Once measured, the probe workers are stopped and rebuilt with the resolved
count. This releases their optimizer/DDP state completely and **reloads the
base checkpoint**. Engine launch specs are generated after resolution. An
explicit count skips both the probe and this rebuild.

For GLM only, the first hardware attempt found that the installed GLM TileLang absorption
path assumes a fused norm and a single LoRA. The pressure recipe uses the
Megatron DSA backend so each token's adapter goes through the normal projection
forward path. It does not drop the KV adapter to make TileLang run.

The experiment hook selects Core's packed DSA and cross-layer index sharing
(runtime Core commit `8c1e05747eb612b382df2632783df5c83a853646`), bypassing the
older Bridge compatibility implementation. It uses memory-efficient SDPA
with exactly the indexer's selected-key mask: Core's unfused reference gathers
large per-key tensors for backward. This computes dense attention with a
sparse mask and **does not measure fused sparse-kernel throughput**. Run
`python -m examples.multi_lora.verify_glm52_dsa` on a GPU to check output and
gradient parity, packed sequence isolation, and 8K forward/backward memory.

## Run

Install `tinker==0.26.2` in the client environment (`wandb` is optional). Place the same
Miles, SGLang and Megatron sources on every node, with a shared model cache and
a shared checkpoint directory that supports atomic directory rename. Join a
dedicated four-node Ray cluster before starting this recipe.

```bash
export MILES_SCRIPT_EXTERNAL_RAY=1
export MASTER_ADDR=<head-ip>
export RAY_ADDRESS=http://<head-ip>:8265
```

Create `launcher.json` containing a JSON argv list, for example:

```json
["python", "examples/multi_lora/run_pressure.py",
 "--model-name", "qwen3-30B-A3B",
 "--hf-checkpoint", "/models/Qwen3-30B-A3B",
 "--sglang-pythonpath", "/sources/sglang/python",
 "--max-running-requests", "128",
 "--sglang-mem-fraction-static", "0.95",
 "--output-dir", "/shared/pressure/runs",
 "--extra-env-vars", "{\"RAY_ADDRESS\":\"<head-ip>:6379\"}"]
```

```bash
python examples/multi_lora/pressure_test.py \
  --experiment-dir /shared/pressure --launcher-command-file launcher.json \
  --dashboard-address http://<head-ip>:8265 --ray-address <head-ip>:6379 \
  --model /models/Qwen3-30B-A3B --dataset /datasets/dapo-math-17k.jsonl \
  --run-id qwen3-pressure
```

After each trial, the supervisor waits for the Ray job to stop and then reaps
remaining processes on every GPU node. Both the exact Ray job ID and source
checkout must match; a cleanup failure stops the search. This prevents orphaned
SGLang schedulers from contaminating the next trial's memory measurement.

N always means N concurrent clients, N training slots, and N GPU LoRA buffers
on every rollout engine. Miles derives the GPU LoRA pool from the resolved
training count; do not override it with a smaller pool during this experiment.
The CPU registry keeps two published versions per slot. The Qwen command above
allows 128 running requests per engine and reserves 95% of GPU memory for
weights, LoRA buffers and KV cache. These settings do not establish that N fits:
the complete 8K rollout and training workload must still pass. Record the
request and memory settings alongside any capacity result.

The search begins at the measured count, moves downward after a confirmed CUDA
OOM or upward after a pass, and requires a passing N plus an OOM at N+1. A
timeout, slot-admission rejection, nonfinite update or non-OOM exception stops
the search for investigation; none of those is a memory-capacity result.

W&B is disabled by default; local metrics and error records are always retained.
To enable uploads, install `wandb`, set `WANDB_API_KEY` and `WANDB_ENTITY`, then
add `--wandb` to the supervisor or standalone client command.

After qualification it launches fresh clients at N with unlimited steps.
With uploads enabled, each LoRA has its own W&B run with loss, reward, accuracy, gradient norm,
lengths, throughput and train/rollout log-prob difference. `wandb.json` contains
the capacity run URL; `clients/<trial>/lora_*-wandb.json` contains each LoRA URL.

Clients write metrics to local `*-telemetry.jsonl` journals. A separate uploader
owns W&B, with per-run console capture and system metrics disabled. Network
backpressure cannot block sampling or training. Upload failures are visible in
`telemetry-upload.log`; restart the uploader on the same experiment with
`python -m examples.multi_lora.pressure_telemetry --root /shared/pressure`.
Stop the previous uploader before replaying; the journals retain each run ID.
Per-client `lora_*-phase.json` files show sampling attempts, accepted groups,
barrier waits and forward/backward progress. Failures write `lora_*-error.json`
and abort the shared barrier before any telemetry cleanup.
Sampler exports created by this harness are pruned to its two latest completed
versions after all older sampling finishes. Training checkpoints are saved
every 100 steps; the harness keeps its latest two completed training checkpoints.
On a later CUDA OOM, the supervisor reduces N, requalifies it, and resumes
continuous training. Credentials stay in process environments, not launch argv.

## Per-LoRA step-time distribution

After N is qualified, each continuous-training LoRA records every completed
step's elapsed time, running mean, P50/P90/P95, min/max, population standard
deviation and a W&B histogram. The capacity run also compares the LoRA means.
These summaries contain only that continuous trial, including its first step;
capacity-search trials and failed larger-N trials are kept separate.

A step starts after the common start barrier, just before publication, and
ends when its optimizer result arrives. Timings split publication, rollout
and batch construction, the client barrier, forward/backward, and optimizer
wait. They are client-observed wall times including server queueing; checkpoint
saving, metric logging and waiting for the next step's start are outside this
interval. Raw rows are in `lora_*-step-times.jsonl`, per-LoRA summaries in
`lora_*-timing-summary.json`, and the comparison in `timing-summary.json`.

Stopping the controller does not release GPU allocations. Capacity renewal is
the allocation owner's responsibility, separate from this experiment.
