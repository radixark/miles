---
title: Fault Tolerance
description: Long jobs survive. Short jobs hope. The mechanisms that make multi-week training possible.
---

# Fault Tolerance

Run a job for an hour and everything's fine. Run it for two weeks and someone, somewhere,
crashes — flaky NIC, a bad ECC bit, a NCCL hang, a compiler cache JSON corruption.
Without fault tolerance you start over. With Miles you keep going.

Enable the whole subsystem with one flag:

```bash
--use-fault-tolerance
```

## What's in the box

| Layer | Mechanism |
|---|---|
| Rollout | Heartbeats + auto-restart of unhealthy SGLang servers |
| Training | Rank-level recovery from latest checkpoint |
| Data | Step-level replay of lost rollouts |
| Weight sync | Idempotent P2P transfer with retry |

## Rollout fault tolerance

Miles periodically sends a `/health_generate` heartbeat to every SGLang server. If a
heartbeat times out:

1. The unhealthy server is **stopped immediately**.
2. The current rollout finishes with the remaining engines (drained from the queue).
3. After the rollout, the failed server is **restarted from latest weights**.
4. Heartbeats resume.

Tunable knobs:

| Flag | Default | Notes |
|---|---|---|
| `--rollout-health-check-first-wait` | `0s` | Grace period for first compile. Bump to **300s+** when using DeepGEMM. |
| `--rollout-health-check-interval` | `30s` | Time between heartbeats. |
| `--rollout-health-check-timeout` | `30s` | When to declare an engine dead. |

!!! tip "DeepGEMM compile"
    DeepGEMM compiles kernels on first run — easily 5+ minutes. If you don't bump
    `--rollout-health-check-first-wait`, Miles will declare the engine dead and restart
    it (which then compiles again, ad infinitum). Set it to **600s** for the largest
    models.

## Training fault tolerance

For training, fault tolerance kicks in via the standard Megatron + Ray combination:

* `--save-interval` writes a checkpoint every N rollouts.
* On any rank failure, Ray restarts the actor.
* The new actor reads from `--load` (which equals `--save`).
* The next rollout uses the recovered weights.

Default cadence is `--save-interval 100`. Drop it to `20` for production runs where
restart cost dominates rollout cost.

## Step-level data replay

If a rollout is half-finished when a worker crashes, the partial samples are kept and
the trainer resamples just the missing prompts. Set:

```bash
ROLLOUT_ARGS+=( --partial-rollout )
```

See [examples/fully-async](../examples/fully-async.md) for how partial-rollout interacts
with the async worker.

## Weight sync retry

[P2P weight transfer](p2p-weight-transfer.md) is idempotent. If a NCCL connection drops
mid-sync:

* The trainer retries up to `--p2p-weight-sync-retries` times (default 3).
* Each retry uses a fresh NCCL group.
* If all retries fail, the trainer falls back to file-based sync (slower but reliable).

## Recommended starting config for production

```bash
TRAIN_ARGS+=(
   --use-fault-tolerance
   --rollout-health-check-first-wait 600
   --rollout-health-check-interval 60
   --save-interval 20
   --partial-rollout
   --p2p-weight-sync-retries 5
)
```

## Things you still have to do yourself

Miles can recover from most things. It can't fix:

* A flaky filesystem that loses checkpoint writes — use a real shared FS, monitor
  `iostat`.
* A node with persistent ECC errors — page in your hardware team.
* A bad model commit that diverges in 5 iterations every time. Roll the actor back to a
  known-good checkpoint.
* Out-of-disk on the checkpoint volume. Set up an alert.

Long RL is a marathon. Build the runway.
