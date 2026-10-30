"""The dashboard ingest hub.

Producers (the dashboard observer in every process of the run, per-node GPU
samplers) push records here; the collector buffers them, appends to the
JSONL streams under ``{dump_details}/dashboard/`` on a flush cadence, runs
the sglang scraper thread against the engines of the latest topology, and
optionally forwards a latest-value snapshot to the existing Prometheus
collector for external Grafana.

This class is deliberately Ray-free: the observer (``observer.py``) wraps it
in a named Ray actor pinned to the driver node, and the per-node sampler
actors are spawned through module-level seams — the collector itself only
ever sees plain method calls, so every behavior here is unit-testable. Producers call in fire-and-forget
style; nothing in the training path ever waits on this class.

Failure policy: if the disk write fails (disk full, NFS hiccup) the error is
logged LOUDLY on every flush attempt — never masked — while ingestion keeps
running with bounded buffers (oldest records dropped past the cap) so a disk
problem cannot OOM the driver.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from miles.utils.logging_utils import RateLimitedWarner
from miles_dashboard.sandbox_health import SNAPSHOT_FILENAME, SandboxHealth, SandboxReport
from miles_dashboard.sglang_scraper import DEFAULT_METRIC_WHITELIST, SglangScraper
from miles_dashboard.store import (
    DataBufferSample,
    EngineSample,
    GpuProcessSample,
    GpuSample,
    Meta,
    MetricsRecord,
    MetricStore,
    PhaseEvent,
    Record,
    TopologySnapshot,
    TrajectoryEvent,
)

logger = logging.getLogger(__name__)

COLLECTOR_ACTOR_NAME = "miles_dashboard_collector"


class _SelfGpuPush:
    """GpuSampler sink pushing back into this collector actor (the sampler
    runs on its own node; the handle crosses the process boundary)."""

    def __init__(self, handle):
        self._handle = handle

    def __call__(self, node: str, batch: list[GpuSample]) -> None:
        self._handle.push_gpu_samples.remote(node, batch)


class _SelfGpuProcessPush:
    """Same as ``_SelfGpuPush`` for the per-process memory breakdown stream."""

    def __init__(self, handle):
        self._handle = handle

    def __call__(self, node: str, batch: list[GpuProcessSample]) -> None:
        self._handle.push_gpu_processes.remote(node, batch)


def _default_list_gpu_nodes() -> list[tuple[str, str]]:
    # no initialized ray in this process = not running as the collector actor
    # (unit tests, tooling): nothing to reconcile, and never pay the import
    ray = sys.modules.get("ray")
    if ray is None or not ray.is_initialized():
        return []
    return [
        (node["NodeID"], node["NodeManagerAddress"])
        for node in ray.nodes()
        if node.get("Alive") and node.get("Resources", {}).get("GPU", 0) > 0
    ]


def _default_spawn_sampler(node_id: str, node_ip: str, interval: float):
    """Spawn + start a GpuSampler pinned to one node; None when NVML is
    unavailable there (recorded so the node is not retried every tick)."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from miles_dashboard.gpu_sampler import GpuSampler

    handle = (
        ray.remote(GpuSampler)
        .options(
            num_cpus=0,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
        )
        .remote(
            push=_SelfGpuPush(ray.get_runtime_context().current_actor),
            node=node_ip,
            interval=interval,
            push_processes=_SelfGpuProcessPush(ray.get_runtime_context().current_actor),
        )
    )
    if ray.get(handle.start.remote()):
        return handle
    ray.kill(handle)  # NVML unavailable; the sampler already warned
    return None


def _default_kill_sampler(handle) -> None:
    import ray

    ray.kill(handle)


# test seams; production always uses the defaults
_list_gpu_nodes = _default_list_gpu_nodes
_spawn_sampler = _default_spawn_sampler
_kill_sampler = _default_kill_sampler


@dataclass
class CollectorConfig:
    dashboard_dir: str  # {dump_details}/dashboard
    run_name: str
    start_ts: float
    args_snapshot: dict[str, Any] = field(default_factory=dict)
    flush_interval_seconds: float = 5.0
    gpu_sample_interval_seconds: float = 1.0
    scrape_interval_seconds: float = 2.0
    metric_whitelist: tuple[str, ...] = DEFAULT_METRIC_WHITELIST
    forward_prometheus: bool = False


class DashboardCollector:
    # bounded ingest buffers: past this many buffered records per stream the
    # oldest are dropped (only reachable when flushing to disk keeps failing)
    MAX_BUFFERED_PER_STREAM: ClassVar[int] = 500_000

    def __init__(
        self,
        *,
        config: CollectorConfig,
        prometheus_handle_factory=None,  # () -> handle with .update.remote(dict), or None
        scraper_http_get=None,  # test hook, forwarded to SglangScraper
    ):
        self.config = config
        self._store = MetricStore(config.dashboard_dir)
        self._store.write_meta(Meta(run_name=config.run_name, start_ts=config.start_ts, args=config.args_snapshot))
        self._lock = threading.Lock()
        self._sandboxes = SandboxHealth(Path(config.dashboard_dir) / SNAPSHOT_FILENAME)
        self._warner = RateLimitedWarner(logger)
        self._dropped_since_flush = 0
        self._last_topology: TopologySnapshot | None = None
        self._scraper: SglangScraper | None = None
        self._scraper_http_get = scraper_http_get
        self._prometheus_handle_factory = prometheus_handle_factory
        # latest-value caches for the Prometheus forwarding snapshot; kept
        # separately because store buffers empty out on every flush
        self._latest_gpu: dict[tuple[str, int], GpuSample] = {}
        self._latest_running_reqs: dict[str, tuple[float, float]] = {}
        self._latest_phase_seconds: dict[str, float] = {}
        self._stop_event = threading.Event()
        self._flush_thread: threading.Thread | None = None
        # per-node GPU samplers, keyed by ray NodeID and reconciled on every
        # flush tick: nodes joining AFTER startup (late rollout nodes, node
        # restarts) get a sampler too — a one-shot spawn at init cannot.
        # None marks a node where NVML is unavailable (never retried).
        self._samplers: dict[str, Any] = {}

    # ------------------------------ lifecycle -------------------------------

    def ping(self) -> bool:
        return True

    def start(self) -> None:
        assert self._flush_thread is None, "collector already started"
        kwargs = {} if self._scraper_http_get is None else {"http_get": self._scraper_http_get}
        self._scraper = SglangScraper(
            self._append,
            engine_addrs=self._current_engine_addrs,
            interval=self.config.scrape_interval_seconds,
            whitelist=self.config.metric_whitelist,
            **kwargs,
        )
        self._scraper.start()
        self._flush_thread = threading.Thread(target=self._run_flush_loop, name="dashboard-flush", daemon=True)
        self._flush_thread.start()

    def shutdown(self) -> None:
        if self._scraper is not None:
            self._scraper.stop()
        self._stop_event.set()
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=self.config.flush_interval_seconds + 1)
        for handle in self._samplers.values():
            if handle is not None:
                _kill_sampler(handle)
        self._samplers.clear()
        self.flush()

    # ------------------------------- ingestion ------------------------------

    def push_metrics(self, record: MetricsRecord) -> None:
        self._append(record)

    def push_phases(self, batch: list[PhaseEvent]) -> None:
        for event in batch:
            self._append(event)

    def push_trajectories(self, batch: list[TrajectoryEvent]) -> None:
        for event in batch:
            self._append(event)

    def push_sandbox_reports(self, batch: list[SandboxReport]) -> None:
        with self._lock:
            for report in batch:
                self._sandboxes.update(report)

    def push_gpu_samples(self, node: str, batch: list[GpuSample]) -> None:
        for sample in batch:
            self._append(sample)

    def push_gpu_processes(self, node: str, batch: list[GpuProcessSample]) -> None:
        for sample in batch:
            self._append(sample)

    def push_data_buffer(self, sample: DataBufferSample) -> None:
        self._append(sample)

    def update_topology(self, snapshot: TopologySnapshot) -> None:
        with self._lock:
            if self._last_topology is not None and self._last_topology.engines == snapshot.engines:
                return  # steady-state re-registration; only changes are recorded
            self._last_topology = snapshot
        self._append(snapshot)

    def _current_engine_addrs(self) -> list[str]:
        with self._lock:
            if self._last_topology is None:
                return []
            return [engine.addr for engine in self._last_topology.engines]

    def _append(self, record: Record) -> None:
        with self._lock:
            if self._store.buffered_count(record.stream) >= self.MAX_BUFFERED_PER_STREAM:
                self._dropped_since_flush += self._store.drop_oldest_buffered(record.stream)
            self._store.append(record)
            self._update_latest(record)

    def _update_latest(self, record: Record) -> None:
        if isinstance(record, GpuSample):
            self._latest_gpu[(record.node, record.gpu)] = record
        elif isinstance(record, EngineSample):
            if record.metric == "sglang_num_running_reqs":
                ts, total = self._latest_running_reqs.get(record.addr, (None, 0.0))
                self._latest_running_reqs[record.addr] = (
                    record.ts,
                    (total if ts == record.ts else 0.0) + record.value,
                )
        elif isinstance(record, PhaseEvent):
            self._latest_phase_seconds[record.name] = record.t1 - record.t0

    # -------------------------------- flushing ------------------------------

    def _run_flush_loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(self.config.flush_interval_seconds)
            try:
                self._reconcile_samplers()
            except Exception:
                logger.exception("dashboard sampler reconcile failed; will retry next tick")
            try:
                self.flush()
            except Exception:
                self._warner.warn("dashboard flush failed; will retry next tick")

    def _reconcile_samplers(self) -> None:
        """Diff alive GPU nodes against owned samplers; spawn the missing.

        Runs on the flush cadence (a ray.nodes() call is cheap), so late-
        joining nodes start reporting util within one flush interval. A node
        restart changes its NodeID, which reads as gone + new -> respawn."""
        alive = _list_gpu_nodes()
        alive_ids = {node_id for node_id, _ in alive}
        for node_id in list(self._samplers):
            if node_id not in alive_ids:
                del self._samplers[node_id]  # node gone; actor died with it
        for node_id, node_ip in alive:
            if node_id not in self._samplers:
                self._samplers[node_id] = _spawn_sampler(node_id, node_ip, self.config.gpu_sample_interval_seconds)

    def flush(self) -> None:
        with self._lock:
            if self._dropped_since_flush:
                logger.error(
                    "dashboard collector dropped %d records since the last flush "
                    "(buffers past the cap — is the disk full?)",
                    self._dropped_since_flush,
                )
                self._dropped_since_flush = 0
            try:
                self._sandboxes.write_snapshot(now=time.time())
            except OSError:
                self._warner.warn("dashboard sandbox snapshot write failed; will retry next flush")
            dropped_before = self._store.unserializable_dropped
            try:
                self._store.flush()
            except OSError:
                # deliberately loud on EVERY failed flush: a disk problem must
                # surface, not be masked; buffers stay bounded via _append
                logger.exception("dashboard flush to %s failed; records stay buffered", self.config.dashboard_dir)
                return
            finally:
                if self._store.unserializable_dropped > dropped_before:
                    self._warner.warn("dashboard flush dropped records that cannot be encoded as JSON")
        if self.config.forward_prometheus and self._prometheus_handle_factory is not None:
            handle = self._prometheus_handle_factory()
            if handle is not None:
                handle.update.remote(self._prometheus_snapshot())

    # -------------------------- prometheus forwarding -----------------------

    def _prometheus_snapshot(self) -> dict[str, float]:
        """Latest-value gauges for external Grafana. Keys avoid characters the
        prometheus collector's sanitizer does not handle ('.', ':')."""

        def safe(text: str) -> str:
            return text.replace("http://", "").replace(".", "_").replace(":", "_")

        with self._lock:
            snapshot = {
                f"dashboard/gpu_{safe(node)}_{gpu}_util": float(sample.util)
                for (node, gpu), sample in self._latest_gpu.items()
            }
            snapshot |= {
                f"dashboard/gpu_{safe(node)}_{gpu}_mem_mb": float(sample.mem_mb)
                for (node, gpu), sample in self._latest_gpu.items()
            }
            snapshot |= {
                f"dashboard/engine_{safe(addr)}_running_reqs": value
                for addr, (_, value) in self._latest_running_reqs.items()
            }
            snapshot |= {
                f"dashboard/phase_{name}_seconds": seconds for name, seconds in self._latest_phase_seconds.items()
            }
        return snapshot
