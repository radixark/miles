---
title: Fault Tolerance
description: Per-cell health checking, the rollout cell state machine, and cell suspend/resume over the api server.
---
Fault tolerance is enabled by `--use-fault-tolerance` and scoped by
`--ft-components` (`rollout`, `train`; defaults to `rollout` when the flag is on
and the list is omitted). Resolution lives in `_resolve_ft_components`
(`miles/utils/arguments.py`).

```bash
--use-fault-tolerance --ft-components rollout train
```

The machinery has three independent pieces:

1. A `SimpleHealthChecker` per cell, which decides whether the cell is healthy.
2. A cell state machine, which decides whether a cell may serve traffic.
3. An HTTP api server, which exposes cells and lets an external controller
   suspend, resume, or fault-inject them.

## Health checker

`SimpleHealthChecker` (`miles/utils/ft_utils/health_checker.py`) runs one asyncio
task per cell. It has no `pause` / `resume` API: probing is *pulled* from a
`get_activeness` predicate, read once at the top of every loop iteration.

Each iteration:

1. Read `get_activeness()`. On a false → true edge, re-arm `first_wait`; on a
   true → false edge, reset the status to `UNKNOWN`.
2. If inactive, sleep and repeat — no probe is sent.
3. If active and `first_wait` is armed, sleep `first_wait` seconds first
   (covers engine compilation and initialization).
4. Call `check_fn` under `timeout`. A success resets the failure counter and
   reports `TRUE`; `failure_threshold` consecutive failures report `FALSE`.

The checker only reports status. It never kills anything — recycling a cell is
the external controller's decision, taken over the api server.

Cells attach it in `create_rollout_cell_health_checker`
(`miles/ray/rollout/server_cell.py`); `check_fn` is `health_generate` against the
cell's SGLang server. When `rollout` is not in `--ft-components`, a
`NoopHealthChecker` is attached instead, so the wiring is unconditional.

### Rollout flags

Registered by `SimpleHealthCheckerConfig.add_arguments(prefix="rollout-health-check")`
(`miles/utils/arguments.py`).

| Flag | Default |
|---|---|
| `--rollout-health-check-interval` | `30.0` |
| `--rollout-health-check-timeout` | `30.0` |
| `--rollout-health-check-first-wait` | `0.0` |
| `--rollout-health-check-failure-threshold` | `3` |

The trainer side registers the same four options under
`--trainer-heartbeat-checker-*` with the config defaults (`10.0` / `10.0` /
`300.0` / `3`).

## Rollout cell state machine

A `ServerCell` (`miles/ray/rollout/server_cell.py`) moves through five states
(`miles/ray/rollout/cell_state.py`):

| State | Meaning |
|---|---|
| `StateUninitialized` | The process exists but is held at its launch gate. |
| `StateInitializing` | Released; waiting for the SGLang server to answer. |
| `StatePendingWeights` | Server is up but still holds stale weights. |
| `StateServing` | Registered with the router and serving. |
| `StateDisposed` | Unregistered and torn down. |

Transitions:

- `init()` calls `activate_launch_gate` against the cell's out-of-band gate port
  and moves `Uninitialized → Initializing`.
- `tick()` is called periodically by `InferenceController._tick_cells`. While
  `Initializing`, it probes the server once with a short timeout; once healthy it
  releases and re-resumes weight memory (colocate only) and moves to
  `PendingWeights`.
- `mark_weights_ready()` registers the cell with the router and moves
  `PendingWeights → Serving`. It is driven by `end_update_weights`, so a cell
  never serves with stale weights. Cells that take no weights (frozen model, or
  `--debug-rollout-only`) skip straight to `Serving` from `tick()`.
- `dispose()` is legal from any state; it stops the health checker, unregisters
  from the router when serving, and moves to `Disposed`.

Health probing is gated on the same machine:
`_health_checker_activeness` is true only in `PendingWeights` or `Serving`, and
only while the controller's global activeness is on. This is what keeps a
gated-but-not-yet-launched cell from being reported unhealthy.

### Gated launch

Engines are launched with `--gated-launch-port` (SGLang side). The process starts,
initializes torch distributed, and then blocks before allocating its memory pools
until someone POSTs to its gate. Under colocate this is what lets a replacement
engine be created at any time without fighting Megatron for memory: the cell sits
in `StateUninitialized` costing only context plus NCCL, and
`InferenceController._ensure_cells_ready` activates it inside the weight-update
window, where the trainer has already offloaded.

## Weight-update window

`start_update_weights` / `end_update_weights`
(`miles/ray/rollout/inference_controller.py`) bracket every weight update:

1. `start_update_weights` pauses health probing globally, initializes any
   `Uninitialized` cell (colocate), and waits until no cell is still
   `Uninitialized` or `Initializing`, releasing the controller lock while it
   waits. It returns the updatable engines plus a snapshot of every cell's
   `workers_hash`.
2. `end_update_weights` marks weights ready only for cells present in that
   snapshot whose hash is unchanged, so a cell that appeared or was replaced
   mid-update is not falsely marked as fresh.

## Api server

`start_api_server` (`miles/utils/ft_utils/api_server/server.py`) exposes a
Kubernetes-shaped read/write view of cells, one `_CellHandler` per enabled
component. It is off by default; `--api-server-port` (`0` disables) turns it on.

Each handler enumerates cells from `RayWorkerManager` and renders status from the
controller:

- Not reported by the manager as alive → `phase="Suspended"`.
- Reported by the manager but not yet observed by the controller, or observed in
  `Uninitialized` / `Initializing` → `phase="Pending"` **without** a `Healthy`
  condition. Cells in this shape are neither healed nor counted as live.
- `PendingWeights` / `Serving` → `phase="Running"` with `Healthy` taken from the
  health checker.

Writes map straight onto the worker manager: `suspend` → `stop_cells`, `resume` →
`start_cells`, plus a fault-injection endpoint used by the soak tests. Resume only
restarts the process; the cell comes back gated and rejoins through the next
weight-update window.

`--mini-ft-controller-enable` starts the built-in controller that polls those
cells and heals unhealthy ones (`--mini-ft-controller-poll-interval`,
`--mini-ft-controller-resume-delay`).

## P2P weight transfer timeouts

When `--update-weight-transfer-mode p2p` is on, every P2P transfer is bounded by
`--p2p-transfer-timeout` (default `30.0`s, defined in `miles/utils/arguments.py`;
consumed at
`miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py`).
On timeout the failed transfer is logged (`[P2P] Transfer future failed: ...`) in
`p2p_transfer_utils.py`. There is no automatic retry or automatic broadcast-mode
fallback in the source today.

## Dumper-mode interaction

In dumper mode (`miles/utils/arguments.py`), Miles forces
`use_fault_tolerance = False`, so no health checker probes and no cell is healed.
