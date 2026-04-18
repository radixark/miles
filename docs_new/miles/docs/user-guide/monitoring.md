---
title: Monitoring & Logging
description: wandb, structured logs, profiling, and what to look at when something looks off.
---

# Monitoring & Logging

A long RL run is half training and half watching dials. Miles tries to make the dials
honest.

## What gets logged by default

Each rollout iteration emits a structured row to stdout:

```text
[trainer] iter 12/3000 | loss=0.412 reward=0.61 kl=0.018
                      | rollout=18.4s train=22.1s p2p=2.1s  (total 42.6s)
                      | grad_norm=0.93 lr=1.0e-06
                      | rollout/min_resp_len=128 rollout/max_resp_len=4096
```

Every column also goes into wandb if `--use-wandb` is on.

## Enabling wandb

```bash
ray job submit --address=auto -- \
  python3 train.py ... \
    --use-wandb \
    --wandb-project miles \
    --wandb-group qwen3-30b-grpo \
    --wandb-tags exp42,sweep
```

Sensitive runs? Add `WANDB_API_KEY` to Ray's `env_vars` instead of baking it into the
launch script.

## What to watch (and what they mean)

| Panel | Healthy pattern | Red flag |
|---|---|---|
| `loss` | Slow decay over hundreds of iterations | Spike → crash within an iteration |
| `reward` | Trending up, with healthy variance | Reward saturates near a single value (collapse) |
| `kl` | Bounded, drifts up over time | Sudden jump (policy diverged from ref) |
| `entropy` | Slowly decreasing | Falls to ~0 too fast (mode collapse) |
| `grad_norm` | < `clip_grad` (1.0 by default) | Repeatedly hitting clip threshold |
| `rollout_time / train_time` | Roughly balanced | One ≫ other → resource imbalance |
| `pg_clipfrac` | < 0.2 | > 0.5 means policy is moving fast → drop LR |
| `truncated_frac` | Low | High = many responses hit max-len |
| `rollout/empty_groups` | 0 | > 0 = filter is dropping everything |

## Per-source logging

When `--prompt-data` is multi-source, every numeric panel is broken out by name:

```text
reward/math   = 0.74
reward/coding = 0.32
reward/chat   = 0.91
```

Use this to diagnose curriculum imbalances.

## Custom loggers

Replace the default loggers with your own to push to internal systems:

```python
def my_log(rollout_id, args, samples, extra, rollout_time) -> bool:
    statsd.gauge("miles.reward", mean([s.reward for s in samples]))
    return False   # also keep default logging
```

```bash
--custom-rollout-log-function-path my_pkg.logging.my_log
```

→ See [Customization #14](customization.md#logging) for both train and eval log hooks.

## Profiling

| Tool | When |
|---|---|
| `nvidia-smi dmon -s u` | Quick sanity check on GPU utilisation |
| `nsys profile` | Deep CUDA-level profiling (PyTorch hooks built in) |
| `py-spy dump --pid <ray worker>` | Find Python-side stalls |
| `ray timeline` | Inspect Ray task scheduling |
| `--profile` flag | Built-in PyTorch profiler — writes Chrome traces |

To capture a profile of just iterations 100–110:

```bash
... --profile --profile-step-start 100 --profile-step-end 110 \
    --profile-output /data/profiles/run-42
```

Open the trace in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev/).

## Where the log files live

| Source | Path |
|---|---|
| Trainer stdout | wherever you redirected `ray job submit` (or Ray dashboard) |
| SGLang | `/tmp/sglang/*.log` (override with `--sglang-log-dir`) |
| Ray workers | `~/.ray/session_latest/logs/` |
| wandb local cache | `wandb/run-<id>/files/` |
| Profiler traces | `--profile-output` |

## Health-check endpoints

When `--miles-router` is on, the router exposes:

| Endpoint | What |
|---|---|
| `GET /healthz` | All engines responsive? |
| `GET /metrics` | Prometheus metrics for queue depth, throughput, errors |
| `GET /routing` | Per-engine routing stats (used by R3) |

Wire these into your usual monitoring stack — most teams point Grafana at `/metrics`.
