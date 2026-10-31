---
title: Monitoring & Logging
description: wandb, structured logs, profiling, and what to look at when something looks off.
---
Miles emits per-rollout metrics to stdout and (optionally) Weights & Biases. SGLang and
Ray write their own logs to their default directories.


## What gets logged by default

Each rollout iteration emits a structured row to stdout (illustrative shape — exact
fields depend on backend and config):

```text
[trainer] iter 12/3000 | loss=0.412 reward=0.61 kl=0.018
                      | rollout=18.4s train=22.1s p2p=2.1s  (total 42.6s)
                      | grad_norm=0.93 lr=1.0e-06
```

When `--use-wandb` is set, metrics also go to wandb under the `train/`, `rollout/`,
and `perf/` namespaces (see `miles/utils/tracking_utils/wandb_utils.py`).

## Enabling wandb

```bash
ray job submit --address=auto -- \
  python3 train.py ... \
    --use-wandb \
    --wandb-project miles \
    --wandb-group qwen3-30b-grpo
```

Available flags: `--use-wandb`, `--wandb-project`, `--wandb-group`. `WANDB_API_KEY`
should be supplied via Ray's `env_vars` rather than baked into the launch script.

## What to watch

| Signal | Healthy pattern | Red flag |
|---|---|---|
| `loss` | Slow decay over hundreds of iterations | Spike → crash within an iteration |
| `raw_reward` | Trending up, with healthy variance | Saturates near a single value (collapse) |
| `kl_loss` | Bounded, drifts up over time | Sudden jump (policy diverged from ref) — only logged when `--use-kl-loss` |
| `train_rollout_logprob_abs_diff` | Stable and small (≪ 1.0) | Climbing without bound → train/inference precision drift |
| `entropy_loss` | Slowly decreasing | Falls to ~0 too fast (mode collapse) |
| `grad_norm` | < `clip_grad` (1.0 by default) | Repeatedly hitting clip threshold |
| `rollout_time` / `train_time` | Roughly balanced | One ≫ other → resource imbalance |
| `train/pg_clipfrac` | < 0.2 | > 0.5 means policy is moving fast → drop LR |

Panel names follow what `loss.py` and the rollout logger emit; Miles's wandb metrics
live under `train/`, `rollout/`, `perf/`, `multi_turn/`, `passrate/` namespaces.

## Custom loggers

Replace the default rollout logger with your own to push to internal systems:

```python
def my_log(rollout_id, args, samples, extra, rollout_time) -> bool:
    statsd.gauge("miles.reward", mean([s.reward for s in samples]))
    return False   # also keep default logging
```

```bash
--custom-rollout-log-function-path my_pkg.logging.my_log
```

## Profiling

| Tool | When |
|---|---|
| `nvidia-smi dmon -s u` | Quick sanity check on GPU utilization |
| `nsys profile` | Deep CUDA-level profiling |
| `py-spy dump --pid <ray worker>` | Find Python-side stalls |
| `ray timeline` | Inspect Ray task scheduling |

### Built-in PyTorch profiler

The PyTorch profiler is wired into Miles via `miles/utils/profile_utils.py`. It reads Megatron's profiling flags on every backend (FSDP and torchtitan mirror them):

```bash
--use-pytorch-profiler
--profile-step-start 1 --profile-step-end 2   # trace steps start .. end-1 (rollout steps for train_overall)
--profile-ranks 0 8                           # global ranks to trace; empty (the Miles images' default) traces every rank
--tensorboard-dir /data/tb-run-42             # one gzipped trace per traced rank
--pytorch-profiler-collect-callstack          # optional: Python call stacks (much larger traces)
--pytorch-profiler-collect-shapes             # optional: tensor shapes
```

`--profile-target train_overall` (the default) traces whole training steps on every backend. `train_actor` and `train_log_probs` trace the microbatch loop of each optimizer step, so their window counts microbatches and restarts at every step; the Megatron backend rejects them.

`--record-memory-history` records a memory snapshot instead (CUDA with the default `--memory-recorder torch`, host memory with `memray`; it needs `train_overall` in `--profile-target`). It is written under `--memory-snapshot-dir` once rollout id `--memory-snapshot-num-steps` - 1 has trained, and the torch recorder also writes one on a CUDA OOM.

Open the trace in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev/).

## Where the log files live

| Source | Path |
|---|---|
| Trainer stdout | wherever you redirected `ray job submit` (or Ray dashboard) |
| Ray workers | `~/.ray/session_latest/logs/` |
| wandb local cache | `wandb/run-<id>/files/` |
| Training profiler / memory snapshot | `--tensorboard-dir`, `--memory-snapshot-dir` |

## Router endpoints

The router exposes a small FastAPI surface used internally by Miles:

| Endpoint | Method | What |
|---|---|---|
| `/workers` | POST | Register an SGLang engine (`{"url": ...}`), as on sgl-router |
| `/workers` | GET | List registered workers with their ids |
| `/workers/{id}` | DELETE | Unregister a worker |
| `/add_worker`, `/remove_worker`, `/list_workers` | POST / POST / GET | The same, in sgl-router's pre-0.2.2 API |
| any other path | GET / POST / PUT / DELETE | Proxied to a selected SGLang worker — e.g. `/generate`, `/v1/chat/completions`, `/health`. |
