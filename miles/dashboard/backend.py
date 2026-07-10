"""Dashboard tracking-backend internals and Ray lifecycle glue.

Call matrix of ``init_dashboard`` (design doc §6.1), reached through the
``MilesDashboardBackend`` adapter in ``tracking_utils``:

- ``train.py`` driver (``primary=True``): builds the config, creates the
  named ``DashboardCollector`` actor pinned to the driver node (same pattern
  as ``prometheus_utils``), starts its flush loop, and spawns one
  ``GpuSampler`` actor per GPU node.
- train actor main rank (``primary=False``): resolves the named actor so
  ``dashboard_log`` works there.
- rollout manager (``primary=False, router_addr=...``): additionally attaches
  a Timer phase sink for ``rollout`` / ``eval_rollout`` events. The router
  itself is registered later through ``hooks.register_router`` — at
  init_tracking time the router has not started yet.

All pushes are fire-and-forget; when the collector cannot be resolved the
module warns once and every later call is a no-op — the dashboard must never
take a training run down with it.
"""

from __future__ import annotations

import logging
import time

from miles.dashboard import hooks
from miles.dashboard.args import collector_config_from_args
from miles.dashboard.collector import COLLECTOR_ACTOR_NAME
from miles.dashboard.logging_utils import RateLimitedWarner
from miles.dashboard.store import MetricsRecord, Role

logger = logging.getLogger(__name__)

GET_ACTOR_TIMEOUT_SECONDS = 60.0
GET_ACTOR_INTERVAL_SECONDS = 2.0

_handle = None
_is_primary = False
_resolution_failed = False
_samplers: list = []
_warner = RateLimitedWarner(logger)


def init_dashboard(args, *, primary: bool = True, router_addr: str | None = None, **kwargs) -> None:
    global _handle, _is_primary
    import ray

    if primary:
        _is_primary = True
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        from miles.dashboard.collector import DashboardCollector

        config = collector_config_from_args(args, start_ts=time.time())
        _handle = (
            ray.remote(DashboardCollector)
            .options(
                name=COLLECTOR_ACTOR_NAME,
                # driver node, hard-pinned: the dump volume and the run's other
                # observability live there, and driver death ends the job anyway
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(), soft=False
                ),
            )
            .remote(config, prometheus_handle_factory=_prometheus_factory if config.forward_prometheus else None)
        )
        ray.get(_handle.ping.remote())
        _handle.start.remote()
        _samplers.extend(spawn_node_samplers(_handle, interval=args.dashboard_gpu_sample_interval))
        logger.info("dashboard collector created; %d node sampler(s) running", len(_samplers))
        return

    if resolve_collector() is None:
        return
    if router_addr is not None:
        # the kwarg marks the rollout-manager process. It cannot be used for
        # scraping: init_tracking runs before start_rollout_servers, so the
        # address is still "http://None:None" here — the real registration
        # happens via hooks.register_router once the router is up.
        hooks.attach_phase_sink(_handle, Role.ROLLOUT_MANAGER)


def resolve_collector():
    """This process's collector handle, resolving the named actor on first
    use (train ranks never run init_dashboard themselves). Returns None —
    permanently, after one warning — when the actor cannot be found."""
    global _handle, _resolution_failed
    if _handle is not None or _resolution_failed:
        return _handle
    import ray

    deadline = time.monotonic() + GET_ACTOR_TIMEOUT_SECONDS
    while True:
        try:
            _handle = ray.get_actor(COLLECTOR_ACTOR_NAME)
            return _handle
        except ValueError:
            if time.monotonic() >= deadline:
                logger.warning(
                    "dashboard collector actor not found after %.0fs; telemetry from this process is disabled",
                    GET_ACTOR_TIMEOUT_SECONDS,
                )
                _resolution_failed = True
                return None
            time.sleep(GET_ACTOR_INTERVAL_SECONDS)


def current_collector():
    """The already-resolved handle, or None. Never blocks (hot paths)."""
    return _handle


def dashboard_log(metrics: dict, *, step: int | None = None, step_key: str | None = None) -> None:
    if _handle is None:
        return
    try:
        _handle.push_metrics.remote(
            MetricsRecord(ts=time.time(), step_key=step_key, step=step, metrics=_scalars_only(metrics))
        )
    except Exception:
        _warner.warn("dashboard metric push failed; dropping this log call")


def finish_dashboard() -> None:
    global _handle, _is_primary, _resolution_failed
    hooks.detach_and_flush()
    if _handle is not None and _is_primary:
        import ray

        for sampler in _samplers:
            try:
                ray.get(sampler.stop.remote(), timeout=10)
            except Exception:
                logger.warning("a dashboard gpu sampler did not stop cleanly", exc_info=True)
            ray.kill(sampler)
        _samplers.clear()
        try:
            # synchronous: the final flush must land before the driver exits
            ray.get(_handle.shutdown.remote(), timeout=30)
        except Exception:
            logger.warning("dashboard collector shutdown incomplete", exc_info=True)
        ray.kill(_handle)
    _handle = None
    _is_primary = False
    _resolution_failed = False


# ------------------------------ ray helpers ---------------------------------


class _RemoteGpuPush:
    """Serializable push callable handed to sampler actors on other nodes."""

    def __init__(self, handle):
        self._handle = handle

    def __call__(self, node: str, batch: list) -> None:
        self._handle.push_gpu_samples.remote(node, batch)


def spawn_node_samplers(collector_handle, *, interval: float) -> list:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from miles.dashboard.gpu_sampler import GpuSampler

    samplers = []
    for node in ray.nodes():
        if not node.get("Alive") or node.get("Resources", {}).get("GPU", 0) <= 0:
            continue
        sampler = (
            ray.remote(GpuSampler)
            .options(
                num_cpus=0,
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False),
            )
            .remote(_RemoteGpuPush(collector_handle), node=node["NodeManagerAddress"], interval=interval)
        )
        if ray.get(sampler.start.remote()):
            samplers.append(sampler)
        else:  # NVML unavailable there; the sampler already warned
            ray.kill(sampler)
    return samplers


def _prometheus_factory():
    """Resolves the miles prometheus collector from inside the dashboard
    collector actor (its module-global handle is per-process)."""
    import ray

    from miles.utils.tracking_utils.prometheus_utils import _COLLECTOR_ACTOR_NAME

    try:
        return ray.get_actor(_COLLECTOR_ACTOR_NAME)
    except ValueError:
        return None


def _scalars_only(metrics: dict) -> dict:
    """The JSONL store persists plain scalars; tracking payloads can carry
    tensors or numpy values, which are converted or dropped here."""
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)):
            out[key] = value
        elif hasattr(type(value), "__float__"):  # numpy/torch zero-dim scalars
            out[key] = float(value)
    return out
