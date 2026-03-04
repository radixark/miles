from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import polars as pl

from miles.utils.ft.controller.mini_prometheus.query import (
    SeriesStore,
    TimeSeriesSample,
    _SeriesKey,
    instant_query,
    range_query,
)
from miles.utils.ft.controller.mini_prometheus.scrape_loop import ScrapeLoop
from miles.utils.ft.models import MetricSample


@dataclass
class MiniPrometheusConfig:
    scrape_interval: timedelta = field(default_factory=lambda: timedelta(seconds=10))
    retention: timedelta = field(default_factory=lambda: timedelta(minutes=60))


class MiniPrometheus:
    def __init__(self, config: MiniPrometheusConfig | None = None) -> None:
        self._config = config or MiniPrometheusConfig()
        self._store = SeriesStore()
        self._last_eviction_time: datetime | None = None

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

    @property
    def _scrape_targets(self) -> dict[str, str]:
        return self._scrape_loop.targets

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
    # Data ingestion
    # -------------------------------------------------------------------

    def ingest_samples(
        self,
        target_id: str,
        samples: list[MetricSample],
        timestamp: datetime | None = None,
    ) -> None:
        ts = timestamp or datetime.now(timezone.utc)
        for sample in samples:
            labels = dict(sample.labels)
            labels["node_id"] = target_id
            key: _SeriesKey = (sample.name, frozenset(labels.items()))

            if key not in self._store.series:
                self._store.series[key] = deque()
                self._store.label_maps[key] = labels
                self._store.name_index.setdefault(sample.name, set()).add(key)

            self._store.series[key].append(TimeSeriesSample(timestamp=ts, value=sample.value))

        self._maybe_evict()

    # -------------------------------------------------------------------
    # Query API (MetricStoreProtocol)
    # -------------------------------------------------------------------

    def instant_query(self, query: str) -> pl.DataFrame:
        return instant_query(self._store, query)

    def range_query(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: timedelta,
    ) -> pl.DataFrame:
        return range_query(
            self._store, query,
            start=start, end=end, step=step,
        )

    # -------------------------------------------------------------------
    # Internal: eviction
    # -------------------------------------------------------------------

    def _maybe_evict(self) -> None:
        now = datetime.now(timezone.utc)
        evict_interval = self._config.retention / 10
        if (
            self._last_eviction_time is not None
            and now - self._last_eviction_time < evict_interval
        ):
            return
        self._last_eviction_time = now
        self._evict_expired()

    def _evict_expired(self) -> None:
        cutoff = datetime.now(timezone.utc) - self._config.retention
        empty_keys: list[_SeriesKey] = []

        for key, samples in self._store.series.items():
            while samples and samples[0].timestamp < cutoff:
                samples.popleft()
            if not samples:
                empty_keys.append(key)

        for key in empty_keys:
            metric_name, _ = key
            del self._store.series[key]
            self._store.label_maps.pop(key, None)
            index_set = self._store.name_index.get(metric_name)
            if index_set is not None:
                index_set.discard(key)
                if not index_set:
                    del self._store.name_index[metric_name]
