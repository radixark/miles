from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from miles.utils.ft.agents.types import MetricSample
from miles.utils.ft.controller.metrics.mini_prometheus.eviction import RetentionEvictor
from miles.utils.ft.controller.metrics.mini_prometheus.in_memory_store import InMemoryMetricStore
from miles.utils.ft.controller.metrics.mini_prometheus.scrape_loop import ScrapeLoop
from miles.utils.ft.controller.types import TimeSeriesStoreProtocol, ScrapeTargetManagerProtocol


@dataclass
class MiniPrometheusConfig:
    scrape_interval: timedelta = field(default_factory=lambda: timedelta(seconds=10))
    retention: timedelta = field(default_factory=lambda: timedelta(minutes=60))


class MiniPrometheus(InMemoryMetricStore, TimeSeriesStoreProtocol, ScrapeTargetManagerProtocol):
    def __init__(self, config: MiniPrometheusConfig | None = None) -> None:
        super().__init__()
        self._config = config or MiniPrometheusConfig()
        self._evictor = RetentionEvictor(store=self, retention=self._config.retention)

        self._scrape_loop = ScrapeLoop(
            store=self,
            scrape_interval_seconds=self._config.scrape_interval.total_seconds(),
        )

    # -------------------------------------------------------------------
    # Scrape target management (delegated to ScrapeLoop)
    # -------------------------------------------------------------------

    def add_scrape_target(self, target_id: str, address: str) -> None:
        self._scrape_loop.add_target(target_id=target_id, address=address)

    def remove_scrape_target(self, target_id: str) -> None:
        self._scrape_loop.remove_target(target_id)

    # -------------------------------------------------------------------
    # Scrape lifecycle (delegated to ScrapeLoop)
    # -------------------------------------------------------------------

    async def scrape_once(self) -> None:
        await self._scrape_loop.scrape_once()

    async def start(self) -> None:
        await self._scrape_loop.start()

    async def stop(self) -> None:
        await self._scrape_loop.stop()

    # -------------------------------------------------------------------
    # Data ingestion (extends base with eviction)
    # -------------------------------------------------------------------

    def ingest_samples(
        self,
        target_id: str,
        samples: list[MetricSample],
        timestamp: datetime | None = None,
    ) -> None:
        super().ingest_samples(target_id, samples, timestamp)
        self._evictor.maybe_evict()

    @property
    def _scrape_targets(self) -> dict[str, str]:
        return self._scrape_loop.targets
