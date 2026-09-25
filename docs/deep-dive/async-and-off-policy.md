---
title: Async Execution and Off-Policy Corrections
description: How each Miles driver overlaps rollout and training, where staleness comes from, how weight sync works, and which log-probs anchor the PPO ratio.
---
This page traces Miles' asynchronous machinery through the code. It covers when generation runs relative to training and weight updates, which policy version produced each token, what the trainer uses as the "old" policy in the PPO ratio, and what the trainer and the SGLang engines each do during a weight sync. Every claim cites `path:line` against the tree this page was written from. Where the code does not settle a question, the page says so.

It is written for readers who already know PPO, GRPO, and importance sampling, and who want to know how Miles implements them. For how to *use* the features, read these pages; this one links to them rather than repeating them:

| Topic | Page |
|---|---|
| Fully async flags, buffer semantics, eval backends, metric glossary | [Fully Async RL](/user-guide/fully-async) |
| Disaggregated rollout and the `disk-delta` transfer walkthrough | [Disaggregated RL Rollout](/advanced/disaggregated-rollout) |
| P2P/RDMA transfer design and benchmarks | [P2P Weight Transfer](/advanced/p2p-weight-transfer) |
| TIS/MIS helper recipes | [Train-infer mismatch helper](/examples/infra-features/train-infer-mismatch-helper) |
| Fault tolerance (the page marks itself outdated; see section 6 below) | [Fault Tolerance](/advanced/fault-tolerance) |

Line numbers describe the current tree and will drift. The symbol names are the stable reference.

## 1. Vocabulary

- **Rollout id** (`rollout_id`). The driver's loop index. Each iteration consumes one rollout batch of `--rollout-batch-size` prompt groups and runs `--num-steps-per-rollout` optimizer steps on it.
- **Weight version**. An integer counter on the trainer's `WeightUpdater`. It starts at `0` (`miles/backends/training_utils/weight_update/updater.py:66`) and increments once per successful sync (`updater.py:98-100`). After the transfer, rank 0 stamps the new value on every engine with `update_weight_version` (`miles/backends/training_utils/weight_update/session.py:73-79`). The driver then publishes it to the `RolloutExecutor` (`miles/ray/placement_group.py:162-164`, `miles/ray/rollout/rollout_executor.py:270-277`). Both drivers sync once before the first rollout (`train.py:59`, `train_async.py:53`), so the first training data carries version `1`. The `disk-delta` protocol is the exception: its first sync only captures a baseline and returns `False` from `begin_sync` (`protocols/delta.py:118-125`), which leaves the first data at version `0`.
- **Staleness**. Measured in weight versions, not optimizer steps: `current_version - span.version` (`miles/rollout/filter_hub/common_filters.py:21-34`). One version is one optimizer step only when `--update-weights-interval 1` and `--num-steps-per-rollout 1`.

## 2. Execution modes

Miles has two drivers, `train.py` and `train_async.py`, and three rollout functions that matter here. The standard `InferenceRolloutFn` generates one batch on demand and aborts the leftovers. `FullyAsyncRolloutFn` keeps a persistent background producer. The `random_async` example keeps a background producer thread of its own.

| Mode | Entry | Colocate | Overlap | Staleness (versions) |
|---|---|---|---|---|
| Sync | `train.py` | Yes or no | None | 0, plus partial-rollout carry-over |
| One-step async | `train_async.py` | No (`train_async.py:25`) | Generation N+1 overlaps training N | Exactly 1 for interval 1 |
| Fully async | `train_async.py --fully-async` | No (`miles/utils/arguments.py:70`) | Continuous generation | Variable. Bounded only by `--max-weight-staleness` |
| Random async (example) | `train_async.py` + custom rollout fn | No | Continuous (own thread) | Variable, unfiltered |

### 2.1 `train.py`, not colocated

In each iteration the loop generates, trains, and then updates the weights. The call sequence is `prepare_rollout` → `rollout_executor.get` → `actor_model.train` → `update_weights` (`train.py:112-162`):

```
version:   v1                     v2                     v3
engines :  [G0 ~~~~~~~~]          [G1 ~~~~~~~~]          [G2 ~~~
trainer :              [T0][U→v2]             [T1][U→v3]
```

`G_N` is generated entirely under the weights that `T_{N-1}` produced, so the batch is on-policy up to the train/inference numerical mismatch. At the end of the batch, `abort()` sends `/abort_request {"abort_all": true}` to every worker (`miles/rollout/inference_rollout/inference_rollout_train.py:26-55`, called at `:163`). This clears the over-sampled groups still in flight, so no request is running when `U` happens.

`train.py` never reads `--update-weights-interval`. It syncs after every rollout except the last, and after the last one only when an eval is due (`train.py:150-162`). The interval flag only affects `train_async.py:88` and the `keep_old_actor` bookkeeping (see section 4.4).

### 2.2 `train.py`, colocated (`--colocate`)

Training and rollout take turns on the same GPUs. `--colocate` turns on `--offload-train` and `--offload-rollout` (`miles/utils/arguments.py:172-212`). The handoff inside one iteration (`train.py:116-164`) runs as follows:

```
G_N on engines → release engine memory (cuda_graph [+kv_cache] [+weights] per --offload-rollout-level)
             → onload trainer, T_N
             → offload trainer (sleep) or clear_memory
             → resume engine weights memory → U (CUDA-IPC) → resume kv_cache + cuda_graph
```

With `--colocate-memory-peak-device gpu` (`arguments.py:298-311`), the order changes so that both copies briefly coexist on the GPU instead of in host RAM: offload KV → onload trainer → offload engine weights (`train.py:117-120`). After training it runs clear_memory → onload engine weights → offload trainer (`train.py:152-157`). The timeline is the sync one from 2.1, with offload and onload stages between the phases.

### 2.3 `train_async.py` (one-step overlap)

The loop starts generating the next batch before it trains on the current one. When a sync is due it waits for that next batch to finish before touching the weights (`train_async.py:120-124`, comment: *"sync generate before update weights to prevent update weight in the middle of generation"*).

```
version:   v1                             v2                     v3
engines :  [G0 ~~~][G1 ~~~~~~~~~~~~]      [G2 ~~~~~~~~~~~]       [G3 ~~~
trainer :          [T0 on G0]  wait G1 [U→v2][T1 on G1] wait G2 [U→v3][T2 on G2]
```

1. `G_{N+1}` is launched with `eager_create_task` at `train_async.py:94`, before `T_N` at `:105`.
2. At `:120-125` the driver awaits `G_{N+1}`, then runs `U`. No generation is in flight during the sync, so `--pause-generation-mode` does not matter here.
3. At `:84-85` the next iteration uses the batch it already awaited.

With interval 1, `T_N` trains on data from the weights *before* `T_{N-1}`. The staleness is exactly one version for every token. A larger interval makes the lag vary within the interval: with interval 2, `G2` is generated at v1 but trained after `T1`, which is two optimizer steps later.

### 2.4 `train_async.py --fully-async`

`--fully-async` swaps the rollout function for `FullyAsyncRolloutFn` (`arguments.py:32-46`). It rejects `--colocate`, `--partial-rollout`, `--pause-generation-mode abort`, `--recompute-logprobs-via-prefill`, and multi-LoRA (`arguments.py:61-81`). The same fn also serves eval unless you pass `--eval-function-path` (`arguments.py:45`).

The first `get()` starts one persistent producer task on the rollout event loop (`miles/rollout/fully_async_rollout.py:84-91`). From then on it runs this loop (`fully_async_rollout.py:131-141`):

```
while True:
    wait until not paused-for-eval
    while scheduler.has_capacity(pending_groups, group_budget): submit one prompt group
    wait for progress (a group finished, or with sample granularity a sample finished)
    for each finished group: await buffer.put(group)      # blocks when the buffer is full
```

The in-flight budget is `rollout_batch_size` groups, or `--async-max-concurrent-samples // n_samples_per_prompt` groups when that flag is set (`fully_async_rollout.py:109-113`). A trainer step only *drains* `rollout_batch_size` finished groups from the buffer (`fully_async_rollout.py:179-216`).

```
version :  v1                   v2                   v3
producer:  ==G==G====G===G==|pause|===G==G===G====|pause|==G==...   (never idle except in U / eval)
buffer  :     ↑  ↑    ↑   ↑        ↑  ↑   ↑             ↑
trainer :  [D0][T0 on D0]  [U→v2][D1][T1 on D1]  [U→v3][D2][T2]
```

When a sync is due, the next drain is deferred until after `U` (`train_async.py:92`, `:126-127`). That drain is then called with `input.weight_version` = the new version, which the buffer's staleness filter compares against. When no sync is due (interval > 1), the next drain starts before training (`train_async.py:93-94`). `U` itself runs in the middle of generation: the engines pause, receive the weights, and resume.

`D` usually takes milliseconds when the buffer is non-empty. The wall time per step moves from `rollout + train` toward `max(rollout, train)` (`fully_async_rollout.py:1-7`).

### 2.5 Random async (`examples/infra_features/random_async/`)

This example is a stress test with random tokens and random rewards. It runs its own `threading.Thread` with a private asyncio loop and a `queue.Queue(maxsize=1000)` (`random_async_rollout.py:268-362`). The rollout function drains `rollout_batch_size` groups from that queue (`:386-424`). The launcher runs `train_async.py` **without** `--fully-async` (`run_random_async_3node.py:175`, and no `--fully-async` anywhere in that file), so the driver takes the one-step branch (2.3). It waits for the next drain before each `U`, while the thread keeps generating across `U`.

```
producer thread: =====================(continuous)==========|pause|=========...
trainer        : [D0][T0 ‖ D1 ][wait D1][U→v2][T1 ‖ D2][wait D2][U→v3]
```

Because generation always spans `U`, the example is useful for exercising `--pause-generation-mode in_place|retract` and the p2p/broadcast paths. It has no staleness filter, no buffer capacity other than the 1000-group queue, and no metrics from `DefaultDataBuffer`. Its submit loop also adds at most one group per second (`random_async_rollout.py:322-330`, a `break` plus `asyncio.sleep(1)`), so it ramps up slowly. The README titles it "Random fully-async example", but it is not `--fully-async`.

### 2.6 The submission scheduler (`--rollout-submission-granularity`)

`miles/rollout/submission_scheduler.py` decides when a finished unit frees a submission slot. Groups are always submitted whole.

- `group` (`submission_scheduler.py:16-28`): a slot frees when the whole group task returns. This is the default for `InferenceRolloutFn` (`inference_rollout_train.py:105`).
- `sample` (`submission_scheduler.py:31-63`): every finished sample decrements `samples_in_flight` through a done-callback that fires on success, exception, or cancel (`inference_rollout_common.py:163-165`). A new group goes out when `samples_in_flight + group_size <= budget * group_size`. This is the default under fully async (`fully_async_rollout.py:68`). In agentic runs it keeps the engines busy while straggler trajectories finish.

The scheduler only paces submission. It does not change which policy version generated anything.

## 3. Staleness: where it comes from and how it is bounded

### 3.1 Version stamping is per token span

Each generate call appends one `WeightVersionsPerCall` to `sample.weight_versions` (`miles/utils/types.py:88`, `:368`). It is built from SGLang's `meta_info` by `WeightVersionsPerCall.from_meta_info` (`types.py:33-57`):

- If `meta_info["weight_versions"]` exists, it holds a list of `{version, start, end}` over the output tokens. Each entry becomes an absolute-position `WeightVersionSpan`. This is how the engine reports that a weight update landed *inside* one request.
- Otherwise the single `meta_info["weight_version"]` is attributed to every output token of the call. SGLang fills that field when the response returns, so a request that straddled an update would be attributed entirely to the newer version, which under-counts span. This fallback only matters for engines without span support.
- The Miles-pinned engine always emits the span list on a finished or aborted request (verified against `sglang-miles-v0.5.19-final` at `c9660a7e11`, the default `SGLANG_BRANCH` in `docker/build.py:52`). When `/update_weight_version` changes the version, the scheduler records `(old_version, len(output_ids))` on every live request. That covers the running batch, the last batch, the waiting queue (where retracted requests sit) and the chunked request (`sglang/python/sglang/srt/managers/scheduler.py:4953-4972`, `sglang/python/sglang/srt/utils/weight_versions.py:39-50`). When the request finishes, the scheduler turns those events into contiguous spans (`sglang/python/sglang/srt/managers/scheduler_components/output_streamer.py:520-529`, `weight_versions.py:66-83`). The TokenizerManager then writes both `weight_versions` and `weight_version` into `meta_info` (`sglang/python/sglang/srt/managers/tokenizer_manager.py:2356-2365`, `weight_versions.py:90-105`). Aborted requests get the same spans through `_make_abort_req` (`scheduler.py:5645-5655`). On this build `generation_version_span` can be trusted. A request that is still held in the TokenizerManager when the version changes (see 3.3) has no tokens yet, so every one of its tokens is correctly attributed to the new version.
- Anchoring requires `return_logprob=True`, and the code asserts this (`types.py:37-45`).

`assert_samples_weight_version_sane` rejects any span stamped `"default"` (an engine that never received a sync) or any non-numeric span (`miles/utils/weight_version.py:33-48`, called at `rollout_executor.py:233`). `assert_weight_version_is_published` fails the run if the driver stops publishing versions (`weight_version.py:17-30`).

### 3.2 Can one trajectory span weight updates?

Yes, in four ways:

1. **Fully async / random async, mid-request.** `U` pauses in-flight requests and resumes them (see 3.3). Tokens sampled before the pause carry `v` and tokens sampled after carry `v+1`, even within one generate call, and a multi-turn trajectory can cross several updates. In `retract` mode the context the post-update tokens attend over is recomputed under `v+1`. In `in_place` mode it is not (3.3). The metrics `rollout/fully_async/avg_generation_version_span` and `rollout/weight_version/mixed_version_ratio` (`miles/ray/rollout/metrics.py:110-114`) measure this.
2. **Partial rollout (sync or one-step drivers).** With `--partial-rollout`, `abort()` collects the unfinished samples, with their partial `tokens`, `rollout_log_probs`, and `weight_versions`, and returns them to the data source (`inference_rollout_train.py:41-53`, `inference_rollout_common.py:221-228`). The data source serves buffered samples first (`miles/rollout/data_source.py:173-191`), and the next rollout appends to them under the new weights (`sglang_rollout.py:242-253`). `sample.metadata["start_rollout_id"]` records where the trajectory began. `--mask-offpolicy-in-partial-rollout` (loss-mask the old prefix) works only on the legacy path. The class-based path raises (`arguments.py:56-60`).
3. **Multi-turn agents under fully async.** Each turn is a separate call with its own version.
4. **External agents / session server.** Same stamping, via `session/samples/merge.py:132`.

`reset_for_retry` (`types.py:321-345`) clears `weight_versions` and `rollout_log_probs` when a prompt is recycled under `--async-unused-samples-handler retry`, so a retried group restarts from scratch at the current version. It also sets `status = ABORTED` (`types.py:339`) rather than `PENDING`, so a custom generate function must set its own final status. I did not trace every generate function to confirm they all do.

### 3.3 In-flight requests at weight update (`--pause-generation-mode`)

The engine-side behaviour below was verified against the Miles-pinned SGLang, `sglang-miles-v0.5.19-final` at `c9660a7e11`. SGLang paths are relative to `sglang/python/sglang/srt/`.

Rank 0 runs the session frame: pause, flush (unless `in_place`), `begin_weight_update`, the transfer, `end_weight_update`, `update_weight_version`, resume (`session.py:14-27`, `updater.py:106-140`). The transfer RPCs go out with `flush_cache=False` (`sglang_api_client.py:105`, `:350`, and neither `protocols/broadcast.py` nor `protocols/cuda_ipc.py` overrides it). SGLang's `begin_weight_update` and `end_weight_update` do not flush either (`managers/scheduler_components/weight_updater.py:395-432`). So the explicit `/flush_cache` in `pause_engines` is the only cache flush in a sync. Miles sends `/update_weight_version` with `abort_all_requests=False` (`sglang_api_client.py:392-396`). SGLang defaults that field to `True` (`managers/io_struct.py:1994`), so a custom caller that omits it aborts every request.

| Mode | What SGLang does | Cache flushed? | Consequence for tokens |
|---|---|---|---|
| `retract` (default) | The TokenizerManager sets `is_pause` and forwards the request to the scheduler (`managers/tokenizer_manager.py:2073-2078`). The scheduler sets `_engine_paused`, so its event loop stops scheduling batches (`managers/scheduler.py:5138`, `:1830`, `:1874`). It collects every unfinished request in the running batch, the last extend batch and the chunked prefill, and frees their KV **without** inserting it into the radix tree (`scheduler.py:5150-5209`, `managers/schedule_batch.py:2064`). They go back to the waiting queue with `output_ids` and output log-probs intact (`schedule_batch.py:1763-1809`). After resume they are re-prefilled over `origin_input_ids + output_ids` (`schedule_batch.py:1369-1380`) | Yes. The flush succeeds because a paused scheduler leaves the waiting queue out of its idle check (`scheduler.py:4747-4749`) | Already-sampled tokens keep the log-probs they were sampled with and are stamped `v`. The whole prefix, prompt and old tokens alike, is recomputed under `v+1`, so later tokens are clean `v+1` samples. No stale KV survives |
| `in_place` | Same pause flag, then the scheduler returns at once, leaving the running batch, the chunked prefill and their KV untouched (`scheduler.py:5140-5148`). After resume, decoding continues on the existing KV | No (`session.py:22-23`). SGLang would refuse anyway, because a flush needs an empty running batch (`scheduler.py:4747-4749`, `managers/io_struct.py:1753-1754`) | New tokens are stamped `v+1` but attend over KV computed with `v`. The radix cache is not flushed either, so **any later request whose prefix hits the cache** also reuses `v` KV. That includes group siblings, the next turn of a multi-turn agent, and waiting requests not yet prefilled. No version stamp records either effect |
| `abort` | Not forwarded to the scheduler. The TokenizerManager sets `is_pause` and keeps sending `abort_all` until no generate call holds `model_update_lock` (`tokenizer_manager.py:2079-2086`) | Yes | Requests return `finish_reason=abort` with their partial output and spans, which sets `Sample.Status.ABORTED` (`types.py:373-374`). The whole group is filtered. Fully async rejects this mode (`arguments.py:72-75`) |

**Requests sent during a pause wait. They are not rejected.** Every generate call waits on `is_pause_cond` before it tokenizes and dispatches (`tokenizer_manager.py:830-831`), and `continue_generation` clears the flag and wakes them (`tokenizer_manager.py:2088-2092`). This holds in all three modes. The fully-async producer is **not** paused during `U`. It keeps submitting up to its in-flight budget, and those requests wait in the TokenizerManager until resume. The producer is only paused for shared-engine eval (`fully_async_rollout.py:93-105`).

**The sync never waits for requests to drain.** Weight-update RPCs skip `model_update_lock` while the engine is paused (`managers/tokenizer_control_mixin.py:460-480`). An unpaused update takes the writer lock, which waits for every in-flight generate call to return, because each call holds the reader lock for its whole lifetime (`tokenizer_manager.py:833`). Miles always pauses first.

Engine-side caveats:

- The FSDP backend ignores `--pause-generation-mode`. It calls `pause_generation()` with the client default `retract` and always flushes (`miles/backends/fsdp_utils/update_weight_utils.py:81-86`).
- `disk-delta` combined with `in_place` and requests in flight. The protocol pauses and skips its own flush (`protocols/delta.py:398-403`), but `update_weights_from_disk` defaults to `flush_cache=True` (`managers/io_struct.py:1807`). The scheduler asserts that the flush after the load succeeds (`managers/scheduler_components/weight_updater.py:160-165`, `:182`), and with a non-empty running batch it cannot. This combination should therefore fail at the first sync that has requests in flight. That conclusion comes from reading the code, not from a run.
- PD disaggregation, prefill side. A retract pause keeps a live mid-chunk prefill request rather than retracting it, so its prefix KV stays on the old weights (`scheduler.py:5202-5209`, a TODO in the source).
- `--use-session-server --use-rollout-routing-replay` with `retract` logs a warning that retract-mode R3 has known SGLang issues (`arguments.py:2847-2853`).

Under the one-step and sync drivers no request is in flight at `U` (sections 2.1 and 2.3), so the mode only matters for fully async, random async, and anything else that drives the engines outside the driver's rhythm.

### 3.4 Bounds per mode

| Mode | Bound | Mechanism |
|---|---|---|
| Sync | 0, except partial-rollout carry-over (unbounded in principle: a sample aborted repeatedly keeps its oldest prefix) | Abort at the end of each batch |
| One-step | 1 version (interval 1) | The driver awaits `G_{N+1}` before `U` |
| Fully async | None by default | `--max-weight-staleness` (default `None`, `arguments.py:745-754`) |
| Random async | None | None |

With `--max-weight-staleness S`, `DefaultDataBuffer.get()` pops FIFO. It computes `current_version - oldest_span_version` for the group (`fully_async_data_buffer.py:163-170`) and hands any group with staleness greater than `S` to the unused handler: `drop` (the default) discards it, and `retry` recycles its prompts (`fully_async_rollout.py:69-73`, `:218-221`).

Three things bound staleness indirectly:

- **Buffer capacity.** `floor(--async-data-buffer-capacity-factor × rollout_batch_size)` finished groups, `2.0` by default (`fully_async_data_buffer.py:105-107`, `:125-133`). A full buffer blocks `put()`, which blocks the producer loop, so no new submissions go out.
- **The in-flight cap** (2.4).
- **FIFO consumption.** At `B` groups per step, a finished group waits at most about `capacity / B` steps.

None of these limits how long a single straggler can stay *in flight*. Its oldest span can be many versions old when it finishes, which is why the filter uses the *oldest* span.

### 3.5 What happens to unused work

| Where | What | Fate |
|---|---|---|
| Fully async `put` | Any sample `ABORTED` (for example an agent timeout, a sandbox crash, or a cancelled task, `fully_async_rollout.py:143-154`) | Unused handler (drop/retry). Counted in `aborted_groups_filtered` |
| Fully async `put` | Missing reward, or rejected by `--dynamic-sampling-filter-path` | Always dropped. Counted as `rollout/dynamic_filter/drop_<reason>` |
| Fully async `get` | Staleness greater than `S` | Unused handler. Counted in `stale_groups_filtered` |
| Sync or one-step end of batch | Over-sampled groups still in flight | Aborted. Recycled only with `--partial-rollout` |

## 4. Off-policy correction: what is the "old" policy?

### 4.1 Three log-prob sources

For each response token `t`:

- `logπ_rollout(t)` comes from `sample.rollout_log_probs`, which SGLang returned while sampling. It belongs to whichever version generated that span, so it is the behavior policy up to the kernel mismatch.
- `logπ_old(t)` is the trainer's forward-only pass at the *start* of the train step. It uses the `actor` weights, or the `old_actor` weights with `--keep-old-actor` (`miles/backends/megatron_utils/actor.py:613-630`). It runs unless `--use-rollout-logprobs` is set without `--get-mismatch-metrics`, or unless `--skip-actor-forward-only` is set.
- `logπ_θ(t)` comes from the training forward, with gradient.

`policy_loss_function` picks the ratio denominator (`miles/backends/training_utils/loss_hub/losses.py:96-141`):

```
old_log_probs = rollout_log_probs         if --use-rollout-logprobs
              = trainer log_probs         otherwise        (or detached logπ_θ under --skip-actor-forward-only)
ppo_kl        = old_log_probs - logπ_θ                                  (losses.py:186-188)
r_t           = exp(-ppo_kl)
pg_loss_t     = max(-r_t·A_t, -clip(r_t, 1-eps_clip, 1+eps_clip_high)·A_t)   (math_utils.py:254-277)
                [dual-clip with --eps-clip-c for A_t<0]
```

### 4.2 What the ratio anchors to, per mode

Let `k` be the trainer's current version and `k-s` the behavior version:

| Flags | Ratio denominator | Staleness corrected? | Train/inference mismatch corrected? |
|---|---|---|---|
| Defaults | `π_k` (trainer recompute) | **No.** The clip is centred on `π_k`, a policy that never generated the data | No |
| `--use-rollout-logprobs` | `π_rollout ≈ π_{k-s}` | Yes, inside the clip (the trust region is around the behavior policy) | Folded in, uncorrected |
| `--use-tis` | `π_k`, and the loss is multiplied by `w = clamp(π_k/π_rollout)` | Yes, as a truncated IS weight outside the clip (decoupled PPO) | Yes, the same weight |
| `--keep-old-actor` | `π_old_actor` (see 4.4) | Only where `old_actor` = the behavior weights | No |

`--use-rollout-logprobs` and `--use-tis` are mutually exclusive (`arguments.py:3056-3057`). In sync mode `s = 0`, so these flags only address the kernel mismatch. In one-step mode `s = 1`. In fully async `s` varies per token span, and the IS weight is per token, so mixed-version trajectories are handled token by token when using rollout log-probs.

`validate_async_off_policy_correction` (`arguments.py:3537-3555`, called at `train_async.py:26`) requires one of the three flags, but it **returns immediately when `--use-critic` is off** (`:3547-3548`). GRPO or REINFORCE runs under `train_async.py` therefore start with the default "anchor to `π_k`" row and no warning. The function's docstring describes exactly this hazard.

### 4.3 The correction functions

The built-in TIS is `vanilla_tis_function` (`loss_hub/corrections.py:7-32`), applied at `losses.py:233-276`:

```
ρ_t        = exp(logπ_old(t) - logπ_rollout(t))
w_t        = clamp(ρ_t, --tis-clip-low (default 0), --tis-clip (default 2.0))
pg_loss_t ← w_t · pg_loss_t
metrics    : tis = mean ρ_t,  tis_clipfrac = mean[w_t ≠ ρ_t],  tis_abs = mean |ρ_t − 1|,  ois = mean r_t
```

Here "old" is the trainer recompute, so `ρ` covers staleness plus kernel mismatch, and the PPO ratio `r = π_θ/π_old` covers the within-step drift over multiple minibatches.

- **`icepop_function`** (`corrections.py:35-62`) sets `w_t = ρ_t` when `ρ_t ∈ [low, high]` and `0` otherwise. Tokens outside the band are dropped instead of truncated. Select it with `--custom-tis-function-path miles.backends.training_utils.loss_hub.corrections.icepop_function`.
- **MIS** (`examples/infra_features/train_infer_mismatch_helper/mis.py:151-310`) adds token, sequence, or geometric levels and truncate, clip, or mask modes, rejection sampling (`rs_*`), a veto threshold, and batch normalization. It reads extra args (`tis_mode`, `tis_level`, `rs_*`) from its YAML. A custom function can return modified loss masks (rejection sampling), and on Megatron the denominators stay the pre-rejection `rollout_mask_sums` (`losses.py:266-276`); FSDP does not ship `rollout_mask_sums`, so there the rejected tokens also leave the denominator. See [Train-infer mismatch helper](/examples/infra-features/train-infer-mismatch-helper).
- **OPSM** (`--use-opsm`, `--opsm-delta`; `loss_hub/math_utils.py:183-221`) zeroes a whole sequence when `A < 0` and `mean_t(logπ_old − logπ_θ) > δ`. It uses whichever `old_log_probs` the flags selected.
- **ESS ratio** (`losses.py:289-300`, logged as `train/ess_ratio`) is computed from `exp(-ppo_kl)`.

`--get-mismatch-metrics` requires `--custom-tis-function-path` (`arguments.py:3059-3062`), even though the built-in TIS function exists.

The mismatch diagnostics `train_rollout_logprob_abs_diff` and `train_rollout_kl` (a k3 estimate of KL(rollout‖train)) are emitted whenever rollout log-probs are present (`losses.py:343-381`). They compare `π_old` (trainer recompute) with `π_rollout`, so **in async modes they measure staleness plus kernel mismatch**, not kernel mismatch alone. To separate the two, compare them with a sync baseline.

### 4.4 `--keep-old-actor`

This flag is Megatron-only. It keeps extra CPU copies of the weights (`actor.py:247-252`) and rotates them at every sync (`actor.py:839-848`):

- With interval 1: `old_actor ← rollout_actor` (the weights the engines had), then `rollout_actor ← actor` (the weights just sent).
- With interval > 1: `old_actor ← actor` (the weights just sent).

At the next train step, `compute_log_prob` runs on `old_actor` (`actor.py:613`). Tracing the one-step loop in 2.3 shows that this is exactly the behavior policy for **`train_async.py` without `--fully-async` and with interval 1**:

- `T_N` trains on `G_N`, which was generated by the weights synced before `T_{N-1}`.
- That is what `old_actor` holds after the rotation.

Elsewhere it is not exact:

- **`train.py`**: `G_N` comes from the *newest* weights, but `old_actor` is one sync behind them.
- **One-step with interval > 1**: the first batch after each sync was generated before it.
- **Fully async**: only groups with staleness 1 match.

`--keep-old-actor` is incompatible with `--skip-actor-forward-only` (`arguments.py:3481`) and with `--rematerialize-param-from-master-weight` (`arguments.py:2744`).

## 5. Weight-sync mechanics

### 5.1 Driver-side sequence

```
update_weights(actor_model, rollout_executor)                         placement_group.py:162
└─ RayTrainGroup.update_weights                                       ray/train/group.py:337-353
   ├─ inference_controller.start_update_weights()   @acquires_lock    inference_controller.py:158-178
   │     pause health checkers; wait until every cell is PendingWeights|Serving;
   │     snapshot (engines, gpu counts/offsets, cell→workers_hash)
   ├─ retry(_execute_first_alive("update_weights", info), 30 attempts)  group.py:35, 345-348
   │     └─ MegatronTrainRayActor.update_weights                      megatron_utils/actor.py:786-855
   │          reconnect if any cell's workers_hash changed (new/healed engines)   :804-815
   │          WeightUpdater.update_weights()                         updater.py:94-142
   │          [ci] compare engine version; rotate keep_old_actor copies
   ├─ inference_controller.end_update_weights()     @releases_lock    inference_controller.py:180-191
   │     PendingWeights cells → register with router → Serving
   └─ rollout_executor.set_weight_version(v)
```

The controller's context lock is held from `start_update_weights` to `end_update_weights`. That serializes weight syncs against offload, onload, eval preparation, cell reconciliation, and fault injection. The acquire and release are not wrapped in `try/finally` (`group.py:343-349`). If all 30 retries fail, the lock is never released, but the run is failing at that point anyway.

### 5.2 The session frame (`updater.py:94-142`)

```
begin_sync(v+1)            protocol hook; False → skip round (disk-delta baseline)
v += 1
rank 0: pause_generation(mode) [+ flush_cache unless in_place]   session.py:14-23
rank 0: register LoRA adapters; begin_weight_update(selector, sync_base)
gloo barrier
for bucket in hf_weight_iterator.iter_hf_weights(...):   every rank joins the TP/EP/PP gathers;
    if sender: protocol.send_bucket(bucket)              only senders materialize and send
protocol.after_base_weights(); gloo barrier
protocol.finalize(v)
rank 0: end_weight_update(checksums) → update_weight_version(v) → continue_generation
gloo barrier; protocol.after_engines_resumed()
```

The whole frame is synchronous on the driver. `train_async.py:125` awaits it, so **no training happens during a sync in any mode**. Only rollout (fully async) or the tail of the next batch (one-step, before `U`) overlaps with training.

Protocols that set `use_weight_update_session = False` skip the pause, begin, end, and version calls and run their own frame. Only `disk-delta` does this (`protocols/delta.py:78-81`).

**Bucketing.** `iter_hf_weights` converts Megatron shards to HF names, keeps "atomic update groups" together, and packs units greedily into buckets of at most `--update-weight-buffer-size` bytes (512 MiB by default; `hf_weight_iterator/__init__.py:78-104`, `bucketing.py:70-86`). A unit is never split. The Megatron gather batches use the same limit (`megatron_utils/update_weight/hf_weight_iterator_direct.py:185-190`). The FSDP backend has its own updater in `miles/backends/fsdp_utils/update_weight_utils.py`.

### 5.3 Protocols

`get_weight_transfer_protocol` (`weight_update/protocol.py:73-90`) picks CUDA-IPC whenever `--colocate` is set. Otherwise `--update-weight-transfer-mode` decides.

| Protocol | File | Sender(s) | Transport | Notes |
|---|---|---|---|---|
| `broadcast` (default) | `protocols/broadcast.py` | Replica rank 0 per PP shard (`:58-69`) | Per-shard NCCL group `miles-pp_{pp}` = sender + every engine GPU (`:85-130`) | Per bucket: an HTTP `update_weights_from_distributed` with names, dtypes, and shapes to every engine, one async `dist.broadcast` per tensor, then a wait for all engines (`:145-177`). With PP > 1, a world ticket lock serializes the PP stages' broadcasts to avoid an engine-side NCCL deadlock (`:33-40`, `:71-82`) |
| CUDA-IPC (colocate) | `protocols/cuda_ipc.py` | Every rank covered by an engine | Each rank serializes its bucket as a `FlattenedTensorBucket` of CUDA-IPC handles, `gather_object`s it to the engine group's source rank, which calls `update_weights_from_tensor` (`:196-259`) | `gather_pp=True`, so PP is gathered first. Engines beyond the actor GPUs in a hybrid layout get a NCCL broadcast tail (`:88-119`). Calls `torch.cuda.ipc_collect()` after resume (`:191-193`) |
| `p2p` | `protocols/p2p.py` | Ranks with `gathered_dp_rank < rollout_num_gpus` (`:144`) | Mooncake RDMA writes from a shared pinned-CPU replica shaped like each target shard (`:199-249`) | Tensors are staged until every shard of a fused SGLang parameter has arrived (`:251-305`). Writes to the last engine are fire-and-forget. `after_base_weights` waits for them (`:61-69`, `:93-119`). Bounded by `--p2p-transfer-timeout` and `--p2p-transfer-num-workers` |
| `disk-delta` | `protocols/delta.py` | Replica rank 0 | Shared FS: xor or overwrite byte deltas against a CPU snapshot, zstd, per-tensor checksum (`:127-155`, `:338-375`) | Engines `/pull_weights` **before** the pause, so the pull overlaps generation. Then pause, `update_weights_from_disk` (which stamps the version), and continue (`:377-419`). See [Disaggregated RL Rollout](/advanced/disaggregated-rollout) |

### 5.4 What each side does during a sync

- **Trainer**: all ranks sit in the gather collectives. Senders also move bytes. Rank 0 drives the engine RPCs. Nothing trains. With `--offload-train`, process groups are rebuilt and later destroyed around the sync (`actor.py:800-802`, `:852-853`). The weights come from the live GPU model, or from the CPU backup when colocated, asleep, or switched to another model tag (`actor.py:778-784`).
- **Engines**: paused. The retract, in-place, or abort semantics from 3.3 apply. New requests wait in each engine's TokenizerManager until resume (3.3). The weight RPCs themselves never flush the cache, because Miles sends them with `flush_cache=False` (3.3). When the session ends, SGLang re-finalizes the weights (`end_weight_update`, which covers quant unpacking and post-load hooks, `sglang/python/sglang/srt/managers/scheduler_components/weight_updater.py:415-432`), records the version stamp on live requests, and resumes.
- **Fault-tolerance health checks** are paused from `start_update_weights` until the next `prepare_rollout` or `prepare_eval` (`inference_controller.py:90-96`, `:161`). A check that was in flight when the pause began is discarded (`ft_utils/health_checker.py:195-201`).
- **Timing** is logged as `perf/update_weights_time` (the `@timer` on `actor.update_weights`), `perf/update_weights_implementation_time`, and `perf/finalize_and_resume_engines_time`. Timers accumulate in the trainer process and are flushed when the *next* `train()` logs (`miles/utils/train_metric_utils.py:21-28`), so a sync's time shows up one step late. The same applies to the protocol metrics `perf/update_weights_density` and `perf/update_weights_wire_bytes` from disk-delta (`delta.py:421-441`, drained at `actor.py:705`).

## 6. Fault tolerance and async

The current machinery is cells plus health checkers plus the mini FT controller. [Fault Tolerance](/advanced/fault-tolerance) describes an earlier stack: `rollout_manager.py`, `health_monitor.py`, and `recover_updatable_engines` no longer exist.

- **Detection.** `--use-fault-tolerance` with `rollout` in `--ft-components` gives each rollout cell a `SimpleHealthChecker` that calls `/health_generate` (`miles/ray/rollout/server_cell.py:281-296`). After `--rollout-health-check-failure-threshold` consecutive failures (each bounded by `--rollout-health-check-timeout`, spaced by `--rollout-health-check-interval`), the cell reports `Healthy=False` (`ft_utils/health_checker.py:217-225`).
- **Healing.** The mini FT controller polls `/api/v1/cells` and suspends and then resumes an unhealthy cell, with backoff (`ft_utils/mini_ft_controller.py:163-192`). A replacement cell goes Initializing → **PendingWeights** (`server_cell.py:167-195`). It joins the router only at the end of the next weight sync (`inference_controller.py:180-191`, `server_cell.py:197-200`), which guarantees it never serves version `"default"`. Under fully async a healed engine therefore sits idle until the next `U`, and the sync reconnects to it (NCCL group or P2P plan rebuilt) because its `workers_hash` changed (`actor.py:804-815`).
- **In-flight requests to a dying engine.** `post()` retries up to 60 times with 1 s sleeps (`miles/utils/http_utils.py:264-296`, `:411`). Each retry is a fresh request through the router, so that turn restarts on another engine. If every retry fails, the exception propagates out of `generate_and_rm_group` (`inference_rollout_common.py:168-176`) and then out of the fully-async worker (`fully_async_rollout.py:143-145`). The next `_next_group` then raises (`:169-171`) and the step fails. Only a `CancelledError` inside a sample is converted to an aborted group (`fully_async_rollout.py:147-154`). An agent integration that catches its own errors and sets `ABORTED` gets the softer path (3.5). In the sync or one-step `InferenceRolloutFn`, a failed group task is logged and skipped (`inference_rollout_train.py:124-128`).
- **Engine death during a sync.** The broadcast or IPC collective blocks until NCCL times out (`--distributed-timeout-minutes`, reset to 10 at `arguments.py:314`). `group.update_weights` retries on the next alive *trainer* cell with the same engine snapshot `info`. The code marks weight-update fault tolerance as work in progress (`inference_controller.py:228-241`, "TEMPORARY … reverted with the weight-update fault tolerance work"), so treat recovery from a mid-sync engine loss as unsupported.
- **P2P.** Per-transfer timeouts come from `--p2p-transfer-timeout`. A failed write raises (`p2p.py:339-341`). There is no fallback to broadcast.

## 7. Metrics and failure modes

These are the async-health signals, with the names as logged:

| Metric | Source | Read it as |
|---|---|---|
| `rollout/fully_async/queue_size` | `fully_async_data_buffer.py:195-236` | Pinned at 0: rollout-bound. Pinned at capacity: train-bound, and staleness grows |
| `rollout/fully_async/{avg,max}_staleness` | Same | Oldest-span lag of the groups that were trained on |
| `rollout/fully_async/{avg,max}_post_generation_staleness` | Same | Lag of the newest span. It measures time spent in the buffer after finishing |
| `rollout/fully_async/{avg,max}_generation_version_span` | Same | Number of updates that landed during a group's generation |
| `rollout/fully_async/token_weighted_staleness` | Same | The staleness most relevant to the IS weights |
| `rollout/fully_async/stale_groups_filtered`, `aborted_groups_filtered` | Same | Wasted generation |
| `rollout/fully_async/weight_version_sample_coverage` | Same | Below 1 means some samples carry no version (a custom generate function dropping `meta_info`) |
| `rollout/fully_async/buffer_{avg,max}_staleness` | Same | Age of what is still waiting |
| `rollout/weight_version/{mean,median,max,min}`, `rollout/weight_version/mixed_version_ratio` | `ray/rollout/metrics.py:110-114` | Every mode. The oldest version per sample and the fraction of samples with more than one version |
| `train/ppo_kl`, `train/pg_clipfrac`, `train/ess_ratio` | `losses.py:369-376` | Drift within a step against the chosen denominator |
| `train/train_rollout_logprob_abs_diff`, `train/train_rollout_kl` | `losses.py:343-381` | `π_old` vs `π_rollout`: staleness plus mismatch in async modes |
| `train/tis`, `train/tis_clipfrac`, `train/tis_abs`, `train/ois` | `corrections.py:26-30`, `losses.py:386-393` | A rising `tis_clipfrac` or `tis_abs` means the IS weights are saturating |
| `train/opsm_clipfrac` | `losses.py:395-396` | Masked negative-advantage sequences |
| `perf/train_wait_time`, `perf/wait_time_ratio` | `train_metric_utils.py:50-54` | Trainer idle between steps. This includes the sync, the drain, and, in one-step mode, the wait for `G_{N+1}` |
| `perf/rollout_time` | `ray/rollout/metrics.py:205-209` | Under fully async this is **drain** time, not generation time |
| `perf/update_weights_time` and sub-timers | 5.4 | Sync cost, logged one step late |

Failure modes to watch for:

1. **Silent anchoring to `π_k`.** Running GRPO under `train_async.py` without `--use-rollout-logprobs` or `--use-tis` (4.2). Every token is at least one version off-policy and nothing corrects for it. `train_rollout_kl` is large compared with a sync run.
2. **A staleness blow-up hidden by the filter default.** With `--max-weight-staleness` unset, a slow trainer lets `buffer_max_staleness` climb without limit. With it set and `drop`, look for `stale_groups_filtered` approaching `rollout_batch_size`: the trainer is starved and the drain logs `No completed rollout groups for 30.0s` (`fully_async_rollout.py:174`).
3. **Stragglers crossing many updates.** High `max_generation_version_span` means individual agent trajectories run across many syncs. Combined with `in_place`, the post-update tokens also attend over stale KV, and so does every later request that hits a radix-cache prefix computed before the update (3.3). Prefer `retract` if correctness matters more than recompute cost.
4. **The fully-async producer died.** An exception in a sample other than cancellation kills the run at the next drain (6). Check the rollout logs for the first traceback, not the drain error.
5. **Mixed-version eval on shared engines.** Eval points should show `eval/{ds}/weight_version/mixed_version_ratio == 0`. See [Fully Async RL: Metrics](/user-guide/fully-async#metrics).
6. **Frozen version.** `assert_weight_version_is_published` fires after `max(3, interval+1)` rollouts without a publish (`weight_version.py:14-30`). This usually means a custom driver skipped `update_weights`.

## 8. Comparison with open-instruct `grpo_fast`

| Aspect | open-instruct | Miles |
|---|---|---|
| Fully synchronous option | No. `async_steps >= 1`, and generation for `k+1` always overlaps training `k` | Yes, `train.py`, including colocated |
| Pipelining knob | `async_steps` (default 8): the prompt queue is prefilled `async_steps × num_unique_prompts_rollout` deep, and DataPrep refuses step `s` until `s − async_steps` is consumed | One-step: fixed at one batch. Fully async: in-flight cap (`--async-max-concurrent-samples`) plus buffer capacity (`--async-data-buffer-capacity-factor`) |
| Staleness bound | Groups older than `async_steps` are dropped | `--max-weight-staleness`, **off by default**, with drop or retry. One-step is 1 by construction |
| Version tag | `model_step` per trajectory at enqueue time. Updates inside a trajectory are invisible | Per-token-span `weight_versions` from SGLang `meta_info` |
| Weight update with requests in flight | `--inflight_updates`: pause without draining. KV kept unless `inflight_updates_recompute_kv_cache` | `--pause-generation-mode retract` (default, KV freed and recomputed, radix cache flushed) / `in_place` (KV and radix cache kept) / `abort` |
| Engine intake during sync | `actor_manager.set_should_stop(True)` stops dequeuing | The fully-async producer keeps submitting. New requests wait in each engine's TokenizerManager until resume and are not rejected |
| Old log-probs | `use_vllm_logprobs` (behavior) or recompute | Recompute (default), `--use-rollout-logprobs`, or `--keep-old-actor` |
| IS correction | TIS `truncated_importance_sampling_ratio_cap` (default 2.0), DPPO divergence mask | `--use-tis` (clamp `[--tis-clip-low, --tis-clip]`, default `[0, 2]`), icepop, MIS example, OPSM |
| Weight transport | NCCL from trainer rank 0 after a ZeRO-3 gather, in a driver `weight_sync_thread` | Bucketed HF conversion, then NCCL per PP shard, RDMA p2p, disk-delta, or CUDA-IPC (colocate). Inline on the driver loop |
| Staleness metrics | `model_step_{min,max,mean}` | `rollout/fully_async/*` and `rollout/weight_version/*` |
| Engine failure | A dead sync thread crashes the run (`health_check_fn`) | Health checker plus mini FT controller heal cells. Healed cells serve after the next sync |

## 9. CLI reference

All flags are defined in `miles/utils/arguments.py`:

| Flag | Default | Line | Role |
|---|---|---|---|
| `--fully-async` | off | 551 | Select `FullyAsyncRolloutFn`. Requires `train_async.py` |
| `--update-weights-interval` | 1 | 853 | Sync every k rollouts (`train_async.py` only) |
| `--pause-generation-mode` | `retract` | 859 | `abort`/`retract`/`in_place` |
| `--rollout-submission-granularity` | `sample` under fully async, `group` otherwise | 708 | Slot release unit |
| `--async-max-concurrent-samples` | `None` → `rollout_batch_size` groups | 756 | In-flight cap in samples |
| `--async-data-buffer-capacity-factor` | 2.0 | 767 | Finished-group buffer size |
| `--max-weight-staleness` | `None` | 745 | Get-time staleness filter |
| `--async-unused-samples-handler` | `drop` | 779 | `drop`/`retry` for aborted or stale groups |
| `--custom-async-data-buffer-path` | `None` | 792 | Replace `DefaultDataBuffer` |
| `--partial-rollout`, `--mask-offpolicy-in-partial-rollout` | off | 726, 736 | Recycle aborted partial samples. The mask flag is legacy-only |
| `--use-rollout-logprobs` | off | 1593 | Behavior log-probs as the ratio denominator |
| `--use-tis`, `--tis-clip`, `--tis-clip-low` | off, 2.0, 0 | 1614-1630 | Truncated IS |
| `--custom-tis-function-path` | `None` | 1632 | icepop, MIS, or your own |
| `--get-mismatch-metrics` | off | 1578 | Extra trainer forward pass for diagnostics |
| `--keep-old-actor` | off | 872 | CPU copy of the behavior weights (see 4.4) |
| `--skip-actor-forward-only` | off | 1602 | Skip the `π_old` pass |
| `--use-opsm`, `--opsm-delta` | off, 1e-4 | 1671, 1677 | Off-policy sequence masking |
| `--eps-clip`, `--eps-clip-high`, `--eps-clip-c` | 0.2, `None`, `None` | 1455-1458 | PPO clip |
| `--update-weight-transfer-mode` | `broadcast` | 910 | `broadcast`/`p2p`/`disk-delta` |
| `--update-weight-buffer-size` | 512 MiB | 844 | Bucket bytes |
| `--update-weight-disk-dir`, `--update-weight-local-checkpoint-dir` | — | 921, 931 | disk-delta paths |
| `--update-weight-delta-encoding`, `--update-weight-delta-checksum` | `xor`, `xxh3-128` | 944, 956 | disk-delta format |
| `--p2p-transfer-num-workers`, `--p2p-transfer-timeout` | 4, 30.0 | 977, 983 | P2P |
| `--colocate`, `--offload-train`, `--offload-rollout`, `--offload-rollout-level` | off, follows colocate, follows colocate, `kv_cache weight` | 172-224 | Colocated memory handoff |
| `--colocate-memory-peak-device` | `cpu` | 298 | Handoff order |
| `--check-weight-update-equal` | off | 2085 | Verify engine weights after sync |
| `--use-fault-tolerance`, `--ft-components`, `--rollout-health-check-{interval,timeout,first-wait,failure-threshold}` | off, `rollout`, 30, 30, 0, 1 | 992-1012 | FT |

## 10. Stale claims elsewhere, and surprises

These existing docs and help strings disagree with the code:

- **`docs/advanced/fault-tolerance.md`** describes a stack that has been removed (`rollout_manager.py`, `health_monitor.py`, `recover_updatable_engines`, and a p2p path under `megatron_utils/update_weight/update_weight_from_distributed/`). The page flags itself as outdated. The current flow is described in section 6.
- **`docs/advanced/p2p-weight-transfer.md`** places the shared pipeline in `miles/backends/megatron_utils/update_weight/` "via `common.py`". The protocols now live in `miles/backends/training_utils/weight_update/`, the Megatron directory only holds the HF iterators, and there is no `common.py`. It also says the engines "increment their weight version". In fact rank 0 sets the version by RPC (`session.py:73-79`).
- **`--fully-async` help** (`arguments.py:554-558`) says eval "keeps the standard rollout function". `resolve_rollout_function_paths` routes eval to `FullyAsyncRolloutFn` unless you pass `--eval-function-path` (`arguments.py:42-45`), and the module docstring agrees with the code.
- **`--max-weight-staleness` help** (`arguments.py:748-753`) says stale groups are "recycled back to the data buffer". They go to `--async-unused-samples-handler`, which defaults to `drop`.
- **The random-async README** calls itself "fully-async", but the launcher does not pass `--fully-async` (2.5).

These are behaviours worth knowing:

1. `train.py` ignores `--update-weights-interval` (2.1).
2. The async off-policy guard only applies to PPO with a critic (4.2).
3. `--keep-old-actor` matches the behavior policy only for one-step async with interval 1 (4.4).
4. Weight-sync timings and disk-delta metrics are logged one step late (5.4).
5. No mode overlaps training with the weight sync itself (5.2).
6. A healed rollout engine idles until the next sync (6).
7. `reset_for_retry` leaves `status = ABORTED` (3.2).
