"""The dashboard ingest hub.

Producers (Timer sinks on every rank, the rollout manager's hooks, per-node
GPU samplers) push records here; the collector buffers them, appends to the
JSONL streams under ``{dump_details}/dashboard/`` on a flush cadence, runs
the sglang scraper thread once a router is registered, and optionally
forwards a latest-value snapshot to the existing Prometheus collector for
external Grafana.

This class is deliberately Ray-free: the backend glue (``backend.py``) wraps
it in a named Ray actor pinned to the driver node and spawns the per-node
sampler actors — the collector itself only ever sees plain method calls, so
every behavior here is unit-testable. Producers call in fire-and-forget
style; nothing in the training path ever waits on this class.

Failure policy: if the disk write fails (disk full, NFS hiccup) the error is
logged LOUDLY on every flush attempt — never masked — while ingestion keeps
running with bounded buffers (oldest records dropped past the cap) so a disk
problem cannot OOM the driver.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, ClassVar

from miles.dashboard.sglang_scraper import DEFAULT_METRIC_WHITELIST, ScrapeMode, SglangScraper
from miles.dashboard.store import (
    EngineSample,
    GpuSample,
    Meta,
    MetricsRecord,
    MetricStore,
    PhaseEvent,
    Record,
    TopologySnapshot,
)

logger = logging.getLogger(__name__)

COLLECTOR_ACTOR_NAME = "miles_dashboard_collector"


@dataclass
class CollectorConfig:
    dashboard_dir: str  # {dump_details}/dashboard
    run_name: str
    start_ts: float
    args_snapshot: dict[str, Any] = field(default_factory=dict)
    flush_interval_seconds: float = 5.0
    scrape_interval_seconds: float = 2.0
    scrape_mode: str = "auto"  # "auto" or a ScrapeMode value; auto resolves at set_router()
    metric_whitelist: tuple[str, ...] = DEFAULT_METRIC_WHITELIST
    forward_prometheus: bool = False


class DashboardCollector:
    # bounded ingest buffers: past this many buffered records per stream the
    # oldest are dropped (only reachable when flushing to disk keeps failing)
    MAX_BUFFERED_PER_STREAM: ClassVar[int] = 500_000

    def __init__(
        self,
        config: CollectorConfig,
        *,
        prometheus_handle_factory=None,  # () -> handle with .update.remote(dict), or None
        scraper_http_get=None,  # test hook, forwarded to SglangScraper
    ):
        self.config = config
        self._store = MetricStore(config.dashboard_dir)
        self._store.write_meta(Meta(run_name=config.run_name, start_ts=config.start_ts, args=config.args_snapshot))
        self._lock = threading.Lock()
        self._dropped_since_flush = 0
        self._last_topology: TopologySnapshot | None = None
        self._scraper: SglangScraper | None = None
        self._scraper_http_get = scraper_http_get
        self._prometheus_handle_factory = prometheus_handle_factory
        # latest-value caches for the Prometheus forwarding snapshot; kept
        # separately because store buffers empty out on every flush
        self._latest_gpu: dict[tuple[str, int], GpuSample] = {}
        self._latest_running_reqs: dict[str, float] = {}
        self._latest_phase_seconds: dict[str, float] = {}
        self._stop_event = threading.Event()
        self._flush_thread: threading.Thread | None = None

    # ------------------------------ lifecycle -------------------------------

    def ping(self) -> bool:
        return True

    def start(self) -> None:
        assert self._flush_thread is None, "collector already started"
        self._flush_thread = threading.Thread(target=self._run_flush_loop, name="dashboard-flush", daemon=True)
        self._flush_thread.start()

    def shutdown(self) -> None:
        if self._scraper is not None:
            self._scraper.stop()
        self._stop_event.set()
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=self.config.flush_interval_seconds + 1)
        self.flush()

    # ------------------------------- ingestion ------------------------------

    def push_metrics(self, record: MetricsRecord) -> None:
        self._append(record)

    def push_phases(self, batch: list[PhaseEvent]) -> None:
        for event in batch:
            self._append(event)

    def push_gpu_samples(self, node: str, batch: list[GpuSample]) -> None:
        for sample in batch:
            self._append(sample)

    def update_topology(self, snapshot: TopologySnapshot) -> None:
        with self._lock:
            if self._last_topology is not None and self._last_topology.engines == snapshot.engines:
                return  # steady-state re-registration; only changes are recorded
            self._last_topology = snapshot
        self._append(snapshot)

    def set_router(self, router_addr: str, *, use_miles_router: bool) -> None:
        """Register the sglang router and start (or re-point) the scraper."""
        if self.config.scrape_mode == "auto":
            mode = ScrapeMode.DIRECT if use_miles_router else ScrapeMode.ROUTER
        else:
            mode = ScrapeMode(self.config.scrape_mode)
        # never hold the lock while stopping a scraper: its thread may be
        # blocked on the same lock inside the _append sink (deadlock)
        with self._lock:
            previous = self._scraper
            if previous is not None and previous.router_addr == router_addr and previous.mode == mode:
                return
            self._scraper = None
        if previous is not None:
            previous.stop()
        kwargs = dict(
            mode=mode,
            router_addr=router_addr,
            engine_addrs=self._current_engine_addrs,
            interval=self.config.scrape_interval_seconds,
            whitelist=self.config.metric_whitelist,
        )
        if self._scraper_http_get is not None:
            kwargs["http_get"] = self._scraper_http_get
        scraper = SglangScraper(self._append, **kwargs)
        with self._lock:
            self._scraper = scraper
        scraper.start()
        logger.info("dashboard scraper started in %s mode against %s", mode, router_addr)

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
                self._latest_running_reqs[record.addr] = record.value
        elif isinstance(record, PhaseEvent):
            self._latest_phase_seconds[record.name] = record.t1 - record.t0

    # -------------------------------- flushing ------------------------------

    def _run_flush_loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(self.config.flush_interval_seconds)
            self.flush()

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
                self._store.flush()
            except OSError:
                # deliberately loud on EVERY failed flush: a disk problem must
                # surface, not be masked; buffers stay bounded via _append
                logger.exception("dashboard flush to %s failed; records stay buffered", self.config.dashboard_dir)
                return
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
                for addr, value in self._latest_running_reqs.items()
            }
            snapshot |= {
                f"dashboard/phase_{name}_seconds": seconds for name, seconds in self._latest_phase_seconds.items()
            }
        return snapshot
