from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta, timezone

import polars as pl

from miles.utils.ft.agents.types import GaugeSample, MetricSample
from miles.utils.ft.controller.metrics.aggregation_mixin import RangeAggregationMixin
from miles.utils.ft.controller.metrics.mini_prometheus.query import SeriesKey, TimeSeriesSample
from miles.utils.ft.controller.metrics.mini_prometheus.query import query_latest as _query_latest
from miles.utils.ft.controller.metrics.mini_prometheus.query import query_range as _query_range
from miles.utils.ft.controller.metrics.mini_prometheus.query import query_single_latest as _query_single_latest
from miles.utils.ft.controller.metrics.mini_prometheus.query import range_aggregate as _range_aggregate

logger = logging.getLogger(__name__)


class OutOfOrderSampleError(ValueError):
    """A sample was ingested with a timestamp older than the latest in that series."""


class InMemoryMetricStore(RangeAggregationMixin):
    """In-memory metric store with typed query API. No scraping, no eviction."""

    def __init__(self) -> None:
        self._series: dict[SeriesKey, deque[TimeSeriesSample]] = {}
        self._label_maps: dict[SeriesKey, dict[str, str]] = {}
        self._name_index: dict[str, set[SeriesKey]] = {}

    def ingest_samples(
        self,
        target_id: str,
        samples: list[MetricSample],
        timestamp: datetime | None = None,
    ) -> None:
        ts = timestamp or datetime.now(timezone.utc)
        logger.debug("mini_prom: ingest target_id=%s, samples=%d", target_id, len(samples))
        for sample in samples:
            labels = dict(sample.labels)
            labels.setdefault("node_id", target_id)
            key: SeriesKey = (sample.name, frozenset(labels.items()))

            if key not in self._series:
                self._series[key] = deque()
                self._label_maps[key] = labels
                self._name_index.setdefault(sample.name, set()).add(key)

            existing = self._series[key]
            if existing and ts < existing[-1].timestamp:
                raise OutOfOrderSampleError(
                    f"out-of-order sample for metric={sample.name!r} target={target_id!r}: "
                    f"new_ts={ts} < latest_ts={existing[-1].timestamp}"
                )

            raw_value = sample.value if isinstance(sample, GaugeSample) else sample.delta
            existing.append(TimeSeriesSample(timestamp=ts, value=raw_value))

    def query_single_latest(
        self,
        metric_name: str,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        return _query_single_latest(self._series, self._label_maps, self._name_index, metric_name, label_filters)

    def query_latest(
        self,
        metric_name: str,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        return _query_latest(self._series, self._label_maps, self._name_index, metric_name, label_filters)

    def query_range(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        return _query_range(self._series, self._label_maps, self._name_index, metric_name, window, label_filters)

    def _dispatch_range_function(
        self,
        func_name: str,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None,
    ) -> pl.DataFrame:
        return _range_aggregate(
            self._series,
            self._label_maps,
            self._name_index,
            func_name,
            metric_name,
            window,
            label_filters,
        )
