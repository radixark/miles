# Fully Asynchronous Rollout Example

This example shows a simple way to make rollout generation **fully asynchronous**: a single global worker is created once and then keeps running in the background, continuously pulling prompts and launching generation tasks. Training only needs to fetch already finished results. This removes the per‑step wait that happens in the normal synchronous style.

The implementation lives in the core library at `miles/rollout/fully_async_rollout.py` (`FullyAsyncRolloutFn`, a class-based rollout function that owns a persistent background worker). It requires `MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1`.

## Files
* `run-qwen3-4b-fully_async.sh`: example launch script with Qwen3‑4B.
* `run-qwen3.5-4b-fully_async-eval.sh`: Qwen3.5‑4B with a dedicated eval fleet (fully-async eval).

## Prerequisite
First set up model & environment following the Qwen3-4B example.

## Quick Start
```bash
cd miles
bash examples/fully_async/run-qwen3-4b-fully_async.sh
```
You should see log lines like:
```
Started fully-async rollout worker
```

## How It Works (Very Short)
* First train call: the rollout fn starts a persistent worker task on the shared rollout event loop.
* The worker keeps up to `--rollout-batch-size` groups in flight using `generate_and_rm_group`.
* Completed groups are pushed into a queue; each step drains until it has `--rollout-batch-size` groups.
* Aborted or too-stale groups are recycled back into the data source.

## Evaluation
Without extra GPUs, eval shares the rollout engines (producer pauses during the blocking
eval). With `--eval-num-gpus`/`--eval-hf-dir`, eval runs on a dedicated fleet synced via
HF checkpoint snapshots; see `run-qwen3.5-4b-fully_async-eval.sh` and the fully-async docs.

## Limitations
* Ordering is best effort (sorted at the end by index).

## Config Differences (3 Key Points)
To enable the fully async pattern there are only three changes compared to a normal run:

1. Use the async training driver: `train_async.py` (not `train.py`).
2. Enable the class-based rollout API: `MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1`.
3. Set the rollout function path:
	```bash
	--rollout-function-path miles.rollout.fully_async_rollout.FullyAsyncRolloutFn
	```

Why is it still "fully" async although `train_async.py` itself schedules rollouts step‑by‑step?

Because the real generation work is done by a **persistent background worker** owned by `FullyAsyncRolloutFn`. Each call from `train_async.py` only drains already completed samples from the worker's output queue; the worker has been continuously generating since the first call. Thus rollout production (model inference) and training consume happen in parallel with minimal waiting.
