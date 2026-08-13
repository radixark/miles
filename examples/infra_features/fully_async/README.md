# Fully Asynchronous Rollout Example

<!-- docs:exclude:start -->
> **Read the docs:** [Fully Async RL](https://miles.radixark.com/docs/user-guide/fully-async)
> covers the schedule, the data buffer, the three evaluation modes, and every `--fully-async`
> argument.
<!-- docs:exclude:end -->

This example shows a simple way to make rollout generation **fully asynchronous**: a single global worker is created once and then keeps running in the background, continuously pulling prompts and launching generation tasks. Training only needs to fetch already finished results. This removes the per‑step wait that happens in the normal synchronous style.

The implementation lives in the core library at `miles/rollout/fully_async_rollout.py` (`FullyAsyncRolloutFn`, a class-based rollout function that owns a persistent background worker). It requires `MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1`.

## Files
* `run_qwen3_5_4b_fully_async_eval.py`: Qwen3.5‑4B with async checkpoint eval — `--eval-backend fleet` (dedicated eval fleet) or `--eval-backend external` (fn-launched sglang server, `examples.infra_features.fully_async.external_eval_fn.ExternalSglangEvalFn`).
* `run_qwen3_30b_a3b_fully_async.py`: the same pattern on a 30B MoE — `tp=8`, `ep=8`, one 8-GPU rollout engine.
* `external_eval_fn.py`: reference `CheckpointEvalFn` — launches/attaches an external sglang server and evals snapshots on it.

## Quick Start
Each launcher downloads its own checkpoint and converts it, then submits the job:
```bash
python examples/infra_features/fully_async/run_qwen3_5_4b_fully_async_eval.py
```
You should see log lines like:
```
Started fully-async rollout worker
```

## At a larger scale
[`examples/experimental/openenv/glm52_tbench2`](../../experimental/openenv/glm52_tbench2) runs
the same flag on a frontier-scale agentic workload: GLM-5.2 744B-A40B on terminal-bench-2,
16 GB300 nodes split 8 training / 8 inference, one Daytona sandbox per episode.
