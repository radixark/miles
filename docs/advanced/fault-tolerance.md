---
title: Fault Tolerance
description: Rollout-side health checks and engine recovery, gated by --use-fault-tolerance.
---
The `--use-fault-tolerance` flag enables Miles's rollout-side
fault-tolerance machinery. It gates two code paths:

1. A `RolloutHealthMonitor` thread per server group, started in
   `miles/ray/rollout.py`, which periodically heart-beats each SGLang
   engine.
2. A recovery hook in the trainer's weight-update step
   (`miles/backends/megatron_utils/actor.py`), which restarts engines
   that the health monitor has killed.

```bash
--use-fault-tolerance
```

The flag is `action="store_true"`, default `False`
(`miles/utils/arguments.py`).

## Health monitor

`RolloutHealthMonitor` (`miles/utils/health_monitor.py`) runs in a daemon
thread. Lifecycle: `start` (called once during init), `pause` and `resume`
(called when engines offload / onload), `stop` (called during dispose).
`pause` / `resume` are wired up in `miles/ray/rollout.py` and called
around offload / onload events.

Each loop iteration does:

1. After a `resume`, wait `--rollout-health-check-first-wait` seconds before
   the first check (intended to cover model compilation and initialization).
2. For every active engine in the group, call `engine.health_generate.remote(timeout=self._check_timeout)`.
3. If the call raises, run `_kill_engine`: `engine.shutdown.remote()`,
   `ray.kill(engine)`, and the engine slot is set to `None`
   (`miles/utils/health_monitor.py`).
4. Sleep `--rollout-health-check-interval` seconds, then repeat.

### Flags

| Flag | Default | Source help text |
|---|---|---|
| `--rollout-health-check-interval` | `30.0` | "Interval in seconds between rollout engine `/health_generate` checks during generate/eval." |
| `--rollout-health-check-timeout` | `30.0` | "Timeout in seconds to wait for a rollout engine `/health_generate` response before killing it." |
| `--rollout-health-check-first-wait` | `0` | "Initial grace period (in seconds) before starting health checks. This allows time for model compilation and initialization. Increase this value significantly when using deepgemm." |

## RPC overload and outcome ownership

Fault-tolerant trainer workers expose heartbeats and recovery operations through
the Miles RPC server. The server applies the following bounded-delivery
contract:

- **Ingress budgets are aggregate, not per request.** Data-plane and
  control-plane bodies use separate aggregate ingress byte and
  in-flight-request budgets, and each lane bounds its concurrent overload
  responses. A bounded byte array replaces per-chunk retention, and conversion
  to the downstream immutable body reserves both copies. There is no per-request
  byte cap and no per-request chunk-count cap. The server does not control how
  a body is fragmented into reads, so a fixed per-request cap would make the
  same legal upload fail or succeed depending on network conditions; the
  aggregate budgets bound the same resource without that dependence.
- **Admitted calls reserve their decoded arguments.** Matching bounded
  queued-request budgets mean many near-limit requests cannot retain multiple
  GiB of input or decoded arguments.
- **Outcome reservation is opt-in.** A method that declares a maximum
  serialized outcome reserves it before its executor starts, and that
  reservation is what the aggregate retained-outcome budget accounts for. A
  method that declares none reserves nothing and its retained outcome is
  therefore not bounded, matching the behaviour before reservations existed.
  New calls receive a retryable capacity response before worker execution when
  the active-call, queued-request, or reserved-outcome budget is full.
- **Heartbeats have their own reserve.** `get_heartbeat_status` uses a separate
  bounded control-plane reserve, so a full training queue does not make a
  healthy worker look dead while heartbeat traffic still has hard call,
  request, and outcome limits. The 65,536-entry control tombstone lane covers
  the 43,201 identities needed by the 12-hour horizon at the minimum supported
  one-second health-check interval; the default trainer heartbeat interval is
  10 seconds and needs 4,321 entries.
- **A call ID and its fixed request digest identify one execution.** The client
  ACKs only after decoding the terminal result or copying the remote error. ACK
  drops the full outcome but keeps the digest tombstone for the 12-hour
  resolution horizon.
- **Duplicate submissions fail loudly.** Reusing a call ID is refused with
  `409`, whether the original is still running, already finished, or already
  acknowledged, and whether or not the payload matches. A resubmission never
  re-executes the call. Submit retries are safe because the client retries a
  submit only when the request provably never reached the server.
- **ACK is pinned to the boot that returned the outcome** and has a sub-second
  retry budget. ACK transport failure never replaces an already decoded result
  or copied business error.
- **There is no server-side shutdown sequence.** The server owns no teardown
  path: a worker process serves until it exits, and the process exit is what
  releases its threads, sockets and memory. Nothing refuses a call because the
  server is stopping, and retained outcomes expire lazily on the next admission
  rather than on a timer. Python also cannot interrupt a synchronous function
  that is already running in a thread, so terminating the owning worker process
  remains the only way to fence that side effect.

Health, in-flight inspection, outcome polling, and ACK remain available while
new data-plane admission is saturated. Capacity does not evict live outcomes or
unexpired tombstones.

## Engine recovery

When `--use-fault-tolerance` is on, `MegatronActor.update_weights` calls
`inference_controller.recover_updatable_engines` before each weight
update (`miles/backends/megatron_utils/actor.py`).

`recover_updatable_engines` (`miles/ray/rollout/inference_controller.py`):

1. Pauses health monitoring.
2. Calls `srv.recover()` on the updatable server.

`srv.recover()` (`miles/ray/rollout.py`):

1. Finds engine slots set to `None` (killed by the health monitor).
2. Calls `start_engines` for each affected group.
3. Releases memory occupation on the new engines.

After `recover_updatable_engines` returns, the weight updater connects to
the new engines and the next weight transfer proceeds normally.

## P2P weight transfer timeouts

When `--update-weight-transfer-mode p2p` is on, every P2P transfer is
bounded by `--p2p-transfer-timeout` (default `30.0`s, defined in
`miles/utils/arguments.py`; consumed at
`miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py`).
On timeout the failed transfer is logged (`[P2P] Transfer future failed: ...`)
in `p2p_transfer_utils.py`. There is no automatic retry or automatic
broadcast-mode fallback in the source today.

## Dumper-mode interaction

In dumper mode (`miles/utils/arguments.py`), Miles forces
`use_fault_tolerance = False` and `rollout_health_check_interval = 1e18`
to keep heartbeats from firing.
