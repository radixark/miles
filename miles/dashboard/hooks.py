"""Process-side hooks feeding the dashboard collector.

These are the functions the (tiny) core wiring calls: a Timer event sink on
every training rank, and engine-topology registration from the rollout
manager. Every entry point is a no-op when the dashboard is disabled or the
collector cannot be reached, and every sink swallows its own exceptions with
rate-limited warnings — a deliberate exception to fail-loud (design doc
§6.3): observability must never be able to kill a training step.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from miles.dashboard.logging_utils import RateLimitedWarner
from miles.dashboard.store import EngineInfo, PhaseEvent, Role, TopologySnapshot
from miles.utils.timer import Timer

logger = logging.getLogger(__name__)

BATCH_MAX_EVENTS = 64
BATCH_MAX_SECONDS = 2.0


@dataclass
class _Identity:
    node: str
    gpus: list[int]
    rank: int


def _default_resolve_identity() -> _Identity:
    import ray
    import torch.distributed as dist

    from miles.utils.misc import get_current_node_ip

    return _Identity(
        node=get_current_node_ip(),
        gpus=[int(gpu) for gpu in ray.get_gpu_ids()],
        rank=dist.get_rank() if dist.is_initialized() else -1,
    )


def _default_ray_get(refs: list):
    import ray

    return ray.get(refs)


# test seams; production always uses the defaults
_resolve_identity = _default_resolve_identity
_ray_get = _default_ray_get


class PhaseSink:
    """Timer event sink: stamps process identity onto intervals and pushes
    them to the collector in batches (fire-and-forget)."""

    def __init__(self, handle, role: str):
        self._handle = handle
        self.role = role
        self._buffer: list[PhaseEvent] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        self._identity: _Identity | None = None
        self._warner = RateLimitedWarner(logger)

    def __call__(self, name: str, t0: float, t1: float) -> None:
        try:
            with self._lock:
                # lazy: torch.distributed is usually not initialized yet when
                # the sink attaches; re-resolve until a real rank appears
                if self._identity is None or (self._identity.rank < 0 and self.role == Role.TRAIN):
                    self._identity = _resolve_identity()
                identity = self._identity
                self._buffer.append(
                    PhaseEvent(
                        name=name,
                        t0=t0,
                        t1=t1,
                        node=identity.node,
                        gpus=identity.gpus,
                        rank=identity.rank,
                        role=self.role,
                    )
                )
                batch = self._take_batch_if_due()
            if batch:
                self._handle.push_phases.remote(batch)
        except Exception:
            self._warner.warn("dashboard phase sink failed; dropping events")

    def _take_batch_if_due(self) -> list[PhaseEvent] | None:
        if len(self._buffer) < BATCH_MAX_EVENTS and time.monotonic() - self._last_flush < BATCH_MAX_SECONDS:
            return None
        batch, self._buffer = self._buffer, []
        self._last_flush = time.monotonic()
        return batch

    def flush(self) -> None:
        try:
            with self._lock:
                batch, self._buffer = self._buffer, []
            if batch:
                self._handle.push_phases.remote(batch)
        except Exception:
            self._warner.warn("dashboard phase sink flush failed; dropping events")


_phase_sink: PhaseSink | None = None
_engines_fingerprint: tuple | None = None
_warner = RateLimitedWarner(logger)


def attach_phase_sink(handle, role: str) -> None:
    global _phase_sink
    if _phase_sink is not None:
        return  # one sink per process
    _phase_sink = PhaseSink(handle, role)
    Timer().event_sinks.append(_phase_sink)


def detach_and_flush() -> None:
    global _phase_sink, _engines_fingerprint
    if _phase_sink is not None:
        if _phase_sink in Timer().event_sinks:
            Timer().event_sinks.remove(_phase_sink)
        _phase_sink.flush()
    _phase_sink = None
    _engines_fingerprint = None


def register_train_actor(args) -> None:
    """Called by TrainActor.init on EVERY rank — init_tracking only runs on
    the megatron main rank, but per-GPU phase lanes need every rank's Timer."""
    if not args.use_miles_dashboard:
        return
    from miles.dashboard import backend

    handle = backend.resolve_collector()
    if handle is None:
        return
    attach_phase_sink(handle, Role.TRAIN)


def register_router(args) -> None:
    """Called by the rollout manager AFTER start_rollout_servers: only then are
    ``args.sglang_router_ip/port`` filled in. init_tracking runs earlier in
    __init__, so the backend cannot register the router at init time."""
    from miles.dashboard import backend

    handle = backend.current_collector()
    if handle is None:
        return
    # a None ip here is a wiring-order bug, not runtime flakiness: fail loud
    assert args.sglang_router_ip is not None, "register_router must run after start_rollout_servers"
    try:
        handle.set_router.remote(
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
            use_miles_router=args.use_miles_router,
        )
    except Exception:
        _warner.warn("dashboard router registration failed; engine metrics will be missing")


def register_engines(servers) -> None:
    """Called at the top of every RolloutManager.generate(): pushes an engine
    topology snapshot whenever the set of engine actors changed (startup,
    fault-tolerance recovery). Steady state costs one local tuple compare."""
    global _engines_fingerprint
    from miles.dashboard import backend

    handle = backend.current_collector()
    if handle is None:
        return
    try:
        chunks = _alive_engine_chunks(servers)
        fingerprint = tuple(id(engine.actor_handle) for chunk in chunks for engine in chunk)
        if fingerprint == _engines_fingerprint:
            return
        infos = _ray_get([engine.actor_handle.get_topology_info.remote() for chunk in chunks for engine in chunk])
        handle.update_topology.remote(TopologySnapshot(ts=time.time(), engines=_group_engines(chunks, infos)))
        _engines_fingerprint = fingerprint
    except Exception:
        _warner.warn("dashboard engine registration failed; topology may be stale")


def _alive_engine_chunks(servers) -> list[list]:
    """Multi-node engines occupy ``nodes_per_engine`` consecutive entries of
    ``group.all_engines``; only the first (master) owns the router-visible
    URL. Chunks with any dead member are skipped until recovery completes."""
    chunks = []
    for server in servers.values():
        for group in server.server_groups:
            stride = group.nodes_per_engine
            engines = group.all_engines
            for i in range(0, len(engines), stride):
                chunk = engines[i : i + stride]
                if all(engine.is_allocated and engine.is_alive for engine in chunk):
                    chunks.append(chunk)
    return chunks


def _group_engines(chunks: list[list], infos: list[dict]) -> list[EngineInfo]:
    engines = []
    index = 0
    for engine_rank, chunk in enumerate(chunks):
        chunk_infos = infos[index : index + len(chunk)]
        index += len(chunk)
        master = chunk_infos[0]
        engines.append(
            EngineInfo(
                addr=master["url"],
                worker_type=master["worker_type"],
                engine_rank=engine_rank,
                gpus=[[info["node_ip"], gpu] for info in chunk_infos for gpu in info["gpu_ids"]],
                gpu_uuids=[uuid for info in chunk_infos for uuid in info["gpu_uuids"]],
            )
        )
    return engines
