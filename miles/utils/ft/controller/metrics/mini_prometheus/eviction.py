from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from miles.utils.ft.controller.metrics.mini_prometheus.query import SeriesKey

if TYPE_CHECKING:
    from miles.utils.ft.controller.metrics.mini_prometheus.in_memory_store import InMemoryMetricStore


class RetentionEvictor:
    def __init__(self, store: InMemoryMetricStore, retention: timedelta) -> None:
        self._store = store
        self._retention = retention
        self._last_eviction_time: datetime | None = None

    def maybe_evict(self) -> None:
        now = datetime.now(timezone.utc)
        evict_interval = self._retention / 10
        if self._last_eviction_time is not None and now - self._last_eviction_time < evict_interval:
            return
        self._last_eviction_time = now
        self._evict_expired()

    def _evict_expired(self) -> None:
        cutoff = datetime.now(timezone.utc) - self._retention
        empty_keys: list[SeriesKey] = []

        for key, samples in list(self._store._series.items()):
            while samples and samples[0].timestamp < cutoff:
                samples.popleft()
            if not samples:
                empty_keys.append(key)

        for key in empty_keys:
            metric_name, _ = key
            del self._store._series[key]
            self._store._label_maps.pop(key, None)
            index_set = self._store._name_index.get(metric_name)
            if index_set is not None:
                index_set.discard(key)
                if not index_set:
                    del self._store._name_index[metric_name]
