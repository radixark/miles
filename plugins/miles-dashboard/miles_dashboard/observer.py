"""The dashboard as a run observer: what miles reports goes to the dashboard collector.

``MILES_DASHBOARD=1`` turns it on. When the run starts, the driver creates
the collector, a named Ray actor on the driver node; every other process of
the run finds it by name and pushes to it fire-and-forget, so nothing in the
training path waits on the dashboard. Per-step samples and per-token values
are not forwarded: the viewer reads them from the ``--dump-details`` dumps.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from miles.utils.logging_utils import RateLimitedWarner
from miles.utils.observers import (
    EngineInfo,
    Observer,
    ProcessStart,
    RunRefusedError,
    RunStart,
    SandboxReport,
    TrajectoryEvent,
)
from miles.utils.tracking_utils.prometheus_utils import _COLLECTOR_ACTOR_NAME as _PROMETHEUS_ACTOR_NAME
from miles_dashboard import sandbox_health, store
from miles_dashboard.args import ENABLE_ENV, collector_config_from_args, env_switch, validate_dashboard_args
from miles_dashboard.collector import COLLECTOR_ACTOR_NAME, DashboardCollector

logger = logging.getLogger(__name__)

BATCH_MAX_EVENTS = 64
BATCH_MAX_SECONDS = 2.0
SHUTDOWN_TIMEOUT_SECONDS = 30.0
# an estimate: the collector takes a push in milliseconds; kept below miles' END_PROCESS_TIMEOUT_S
END_PROCESS_TIMEOUT_SECONDS = 10.0

# the processes the timeline has lanes for
_PHASE_ROLES = {"train": store.Role.TRAIN, "rollout_manager": store.Role.ROLLOUT_MANAGER}


class DashboardObserver(Observer):
    def __init__(self):
        self._collector = None
        self._owns_collector = False
        self._process: ProcessStart | None = None
        self._phases = _Batcher(lambda batch: self._collector.push_phases.remote(batch))
        self._trajectories = _Batcher(lambda batch: self._collector.push_trajectories.remote(batch))

    def enabled(self) -> bool:
        return env_switch(ENABLE_ENV)

    def start_run(self, run: RunStart) -> dict[str, str]:
        validate_dashboard_args(run["args"])
        try:
            config = collector_config_from_args(run["args"], start_ts=run["start_ts"])
        except ValueError as e:
            raise RunRefusedError(f"unreadable dashboard setting: {e}") from e
        self._collector = (
            ray.remote(DashboardCollector)
            .options(
                name=COLLECTOR_ACTOR_NAME,
                # driver node, hard-pinned: the dump volume and the run's other
                # observability live there, and driver death ends the job anyway
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(), soft=False
                ),
            )
            .remote(
                config=config, prometheus_handle_factory=_prometheus_collector if config.forward_prometheus else None
            )
        )
        ray.get(self._collector.ping.remote())
        self._collector.start.remote()
        self._owns_collector = True
        logger.info(
            "miles dashboard: telemetry -> %s | view live: python -m miles_dashboard.serve "
            "--dump-details %s --follow --port 7788",
            config.dashboard_dir,
            run["args"]["dump_details"],
        )
        return {ENABLE_ENV: "1"}

    def start_process(self, process: ProcessStart) -> None:
        self._process = process
        if self._collector is None:
            self._collector = _find_collector()

    def phase_start(self, name: str, t0: float) -> None:
        # pushed at once: an open phase shows the process is still in it
        if (phase := self._phase(name, t0, store.PhaseEvent.OPEN_T1)) is not None:
            self._collector.push_phases.remote([phase])

    def phase_end(self, name: str, t0: float, t1: float) -> None:
        if (phase := self._phase(name, t0, t1)) is not None:
            self._phases.add(phase)

    def trajectory_event(self, event: TrajectoryEvent) -> None:
        if self._collector is not None:
            self._trajectories.add(store.TrajectoryEvent(**event))

    def engine_topology(self, engines: list[EngineInfo]) -> None:
        if self._collector is not None:
            self._collector.update_topology.remote(
                store.TopologySnapshot(ts=time.time(), engines=[store.EngineInfo(**engine) for engine in engines])
            )

    def data_buffer(self, length: int) -> None:
        if self._collector is not None:
            self._collector.push_data_buffer.remote(store.DataBufferSample(ts=time.time(), length=length))

    def sandbox_report(self, report: SandboxReport) -> None:
        if self._collector is not None:
            self._collector.push_sandbox_reports.remote([sandbox_health.SandboxReport(**report)])

    def metrics(self, metrics: dict[str, Any], *, step_key: str, step: int | None) -> None:
        if self._collector is not None:
            self._collector.push_metrics.remote(
                store.MetricsRecord(ts=time.time(), step_key=step_key, step=step, metrics=_scalars_only(metrics))
            )

    def end_process(self) -> None:
        self._phases.flush()
        self._trajectories.flush()
        if self._collector is not None:
            # the collector runs each caller's calls in order, so the ping answers after this process's pushes
            ray.get(self._collector.ping.remote(), timeout=END_PROCESS_TIMEOUT_SECONDS)

    def end_run(self, *, succeeded: bool) -> None:
        if not self._owns_collector:
            return
        try:
            # synchronous: the final flush must land before the driver exits
            ray.get(self._collector.shutdown.remote(), timeout=SHUTDOWN_TIMEOUT_SECONDS)
        except Exception:
            logger.warning("dashboard collector shutdown incomplete", exc_info=True)
        ray.kill(self._collector)

    def _phase(self, name: str, t0: float, t1: float) -> store.PhaseEvent | None:
        if self._collector is None or self._process is None:
            return None
        if (role := _PHASE_ROLES.get(self._process["role"])) is None:
            return None
        process = self._process
        return store.PhaseEvent(
            name=name, t0=t0, t1=t1, node=process["node"], gpus=process["gpus"], rank=process["rank"], role=role
        )


class _Batcher:
    """Pushes records in batches, by count or by age. A background thread also
    pushes what a process that went quiet still holds: the last phases of a
    train rank idling through an eval would otherwise show as still open.
    Batches are pushed under the lock, so ``flush`` returns only after every
    batch taken before it has been pushed."""

    def __init__(self, push: Callable[[list], Any]):
        self._push = push
        self._records: list = []
        self._lock = threading.Lock()
        self._last_push = time.monotonic()
        self._pusher: threading.Thread | None = None
        self._warner = RateLimitedWarner(logger)

    def add(self, record) -> None:
        with self._lock:
            self._records.append(record)
            if len(self._records) >= BATCH_MAX_EVENTS or time.monotonic() - self._last_push >= BATCH_MAX_SECONDS:
                self._push(self._take())
            if self._pusher is None:
                self._pusher = threading.Thread(target=self._push_periodically, daemon=True, name="dashboard-push")
                self._pusher.start()

    def flush(self) -> None:
        with self._lock:
            if self._records:
                self._push(self._take())

    def _take(self) -> list:
        batch, self._records = self._records, []
        self._last_push = time.monotonic()
        return batch

    def _push_periodically(self) -> None:
        while True:
            time.sleep(BATCH_MAX_SECONDS)
            try:
                self.flush()
            except Exception:
                self._warner.warn("dashboard push failed; dropping a batch")


def _find_collector():
    """The run's collector, which the driver created before it started this process."""
    try:
        return ray.get_actor(COLLECTOR_ACTOR_NAME)
    except ValueError:
        logger.warning("dashboard collector actor not found; telemetry from this process is disabled")
        return None


def _prometheus_collector():
    """Resolves the miles prometheus collector from inside the dashboard
    collector actor (its module-global handle is per-process)."""
    try:
        return ray.get_actor(_PROMETHEUS_ACTOR_NAME)
    except ValueError:
        return None


def _scalars_only(metrics: dict) -> dict:
    """The JSONL store persists plain scalars; logged metrics can carry numpy
    values or non-scalars, which are converted or dropped here."""
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)):
            out[key] = value
        elif hasattr(
            type(value), "__float__"
        ):  # numpy zero-dim scalars  # config-access-exempt: detect numeric scalar conversion without importing array libraries
            out[key] = float(value)
    return out
