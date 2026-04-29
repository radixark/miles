---
title: Fault Tolerance
description: Heartbeats, rank-level recovery, and partial-rollout reuse.
---

# Fault Tolerance

Long-running RL jobs encounter transient failures: flaky NICs, ECC errors,
NCCL hangs, compiler-cache corruption. Miles ships rollout-side fault tolerance
to absorb most of those without restarting the run.

Enable with:

```bash
--use-fault-tolerance
```

The `--use-fault-tolerance` flag turns on the rollout health-check subsystem.
Training-side recovery uses the standard Megatron + Ray combination and is not
toggled by this flag.

## Rollout health checks

Miles periodically calls `/health_generate` on every SGLang server
(`miles/utils/health_monitor.py`, `miles/backends/sglang_utils/sglang_engine.py`).
If a heartbeat times out:

1. The unhealthy server is stopped.
2. The current rollout drains using the remaining engines.
3. After the rollout, the failed server is restarted from the latest weights.
4. Heartbeats resume.

| Flag | Default | Notes |
|---|---|---|
| `--rollout-health-check-first-wait` | `0` | Grace period before heartbeats start. Bump for first-run kernel compilation (DeepGEMM, large MoE). |
| `--rollout-health-check-interval` | `30` | Seconds between heartbeats. |
| `--rollout-health-check-timeout` | `30` | Heartbeat timeout. |

!!! tip "First-run kernel compilation"
    DeepGEMM and similar JIT'd kernels can take several minutes to compile on
    the first run. Without raising `--rollout-health-check-first-wait`, Miles
    can declare the engine dead and restart it (which then recompiles). 600
    seconds is a safe value for the largest models.

## Training-side recovery

Training-side recovery is provided by Megatron checkpointing plus Ray:

* `--save-interval` writes a checkpoint every N rollouts.
* On rank failure, Ray restarts the actor.
* The new actor reads from `--load` (typically equal to `--save`).
* The next rollout uses the recovered weights.

Restart cost depends on `--save-interval`; production runs often use 20 to
100.

## Partial rollout

If a rollout is partially complete when a worker fails, Miles can keep the
already-finished trajectories and only resample the missing prompts:

```bash
ROLLOUT_ARGS+=( --partial-rollout )
```

See [examples/fully-async](../examples/fully-async.md) for how partial-rollout
interacts with the async worker.

## What you still own

Miles cannot recover from these on its own:

* A flaky filesystem that loses checkpoint writes. Use a real shared FS and
  monitor `iostat`.
* A node with persistent ECC errors. Hand off to hardware operations.
* A bad model commit that diverges deterministically. Roll back to a
  known-good checkpoint.
* Out-of-disk on the checkpoint volume. Set up alerts.
