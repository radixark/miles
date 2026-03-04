from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import polars as pl

from typing import Iterator

from miles.utils.ft.controller.mini_prometheus.promql import (
    CompareExpr,
    CompareOp,
    MetricSelector,
    PromQLExpr,
    RangeFunction,
    RangeFunctionCompare,
    _compare_col,
    _match_labels,
    parse_promql,
)
from miles.utils.ft.controller.mini_prometheus.scraper import parse_prometheus_text
from miles.utils.ft.models import MetricSample

logger = logging.getLogger(__name__)

_SeriesKey = tuple[str, frozenset[tuple[str, str]]]


@dataclass
class _TimeSeriesSample:
    timestamp: datetime
    value: float


@dataclass
class MiniPrometheusConfig:
    scrape_interval: timedelta = field(default_factory=lambda: timedelta(seconds=10))
    retention: timedelta = field(default_factory=lambda: timedelta(minutes=60))


class MiniPrometheus:
    def __init__(self, config: MiniPrometheusConfig | None = None) -> None:
        self._config = config or MiniPrometheusConfig()
        self._series: dict[_SeriesKey, deque[_TimeSeriesSample]] = {}
        # Cached label dicts to avoid reconstructing from frozenset on every query
        self._label_maps: dict[_SeriesKey, dict[str, str]] = {}
        self._name_index: dict[str, list[_SeriesKey]] = {}
        self._scrape_targets: dict[str, str] = {}
        self._running = False
        self._last_eviction_time: datetime | None = None

    # -------------------------------------------------------------------
    # Scrape target management
    # -------------------------------------------------------------------

    def add_scrape_target(self, target_id: str, address: str) -> None:
        self._scrape_targets[target_id] = address

    def remove_scrape_target(self, target_id: str) -> None:
        self._scrape_targets.pop(target_id, None)

    # -------------------------------------------------------------------
    # Data ingestion
    # -------------------------------------------------------------------

    def ingest_samples(
        self,
        target_id: str,
        samples: list[MetricSample],
        timestamp: datetime | None = None,
    ) -> None:
        ts = timestamp or datetime.utcnow()
        for sample in samples:
            labels = dict(sample.labels)
            labels["node_id"] = target_id
            key: _SeriesKey = (sample.name, frozenset(labels.items()))

            if key not in self._series:
                self._series[key] = deque()
                self._label_maps[key] = labels
                self._name_index.setdefault(sample.name, []).append(key)

            self._series[key].append(_TimeSeriesSample(timestamp=ts, value=sample.value))

        self._maybe_evict()

    # -------------------------------------------------------------------
    # HTTP scraping
    # -------------------------------------------------------------------

    async def scrape_once(self) -> None:
        import httpx  # optional heavy dependency

        targets = list(self._scrape_targets.items())
        if not targets:
            return

        async with httpx.AsyncClient(timeout=10.0) as client:

            async def _scrape_target(target_id: str, address: str) -> None:
                try:
                    response = await client.get(f"{address}/metrics")
                    response.raise_for_status()
                    samples = parse_prometheus_text(response.text)
                    self.ingest_samples(target_id=target_id, samples=samples)
                except Exception:
                    logger.warning(
                        "Failed to scrape target %s at %s",
                        target_id,
                        address,
                        exc_info=True,
                    )

            await asyncio.gather(
                *(_scrape_target(tid, addr) for tid, addr in targets)
            )

    async def start(self) -> None:
        self._running = True
        while self._running:
            await self.scrape_once()
            await asyncio.sleep(self._config.scrape_interval.total_seconds())

    async def stop(self) -> None:
        self._running = False

    # -------------------------------------------------------------------
    # Query API (MetricStoreProtocol)
    # -------------------------------------------------------------------

    def instant_query(self, query: str) -> pl.DataFrame:
        expr = parse_promql(query)
        return self._evaluate_instant(expr)

    def range_query(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: timedelta,
    ) -> pl.DataFrame:
        expr = parse_promql(query)
        return self._evaluate_range(expr, start=start, end=end, step=step)

    # -------------------------------------------------------------------
    # Internal: shared query helpers
    # -------------------------------------------------------------------

    def _iter_matching_series(
        self, selector: MetricSelector,
    ) -> Iterator[tuple[dict[str, str], deque[_TimeSeriesSample]]]:
        for key in self._name_index.get(selector.name, []):
            samples = self._series.get(key)
            if not samples:
                continue

            labels = self._label_maps[key]
            if not _match_labels(labels, selector.matchers):
                continue

            yield labels, samples

    @staticmethod
    def _filter_by_compare(
        df: pl.DataFrame, op: CompareOp, threshold: float,
    ) -> pl.DataFrame:
        if df.is_empty():
            return df
        return df.filter(_compare_col(pl.col("value"), op, threshold))

    # -------------------------------------------------------------------
    # Internal: instant evaluation
    # -------------------------------------------------------------------

    def _evaluate_instant(self, expr: PromQLExpr) -> pl.DataFrame:
        if isinstance(expr, MetricSelector):
            return self._instant_selector(expr)

        if isinstance(expr, CompareExpr):
            df = self._instant_selector(expr.selector)
            return self._filter_by_compare(df, expr.op, expr.threshold)

        if isinstance(expr, RangeFunction):
            return self._instant_range_function(expr)

        if isinstance(expr, RangeFunctionCompare):
            df = self._instant_range_function(expr.func)
            return self._filter_by_compare(df, expr.op, expr.threshold)

        raise ValueError(f"Unsupported expression type: {type(expr)}")

    def _instant_selector(self, selector: MetricSelector) -> pl.DataFrame:
        rows: list[dict] = []
        for labels, samples in self._iter_matching_series(selector):
            latest = samples[-1]
            row: dict = {"__name__": selector.name, "value": latest.value}
            row.update(labels)
            rows.append(row)

        if not rows:
            return pl.DataFrame({"__name__": [], "value": []})
        return pl.DataFrame(rows)

    def _instant_range_function(self, func: RangeFunction) -> pl.DataFrame:
        now = datetime.utcnow()
        window_start = now - func.duration
        rows: list[dict] = []

        for labels, samples in self._iter_matching_series(func.selector):
            window_samples = [s for s in samples if s.timestamp >= window_start]
            if not window_samples:
                continue

            value = _apply_range_function(func.func_name, window_samples)
            row: dict = {"__name__": func.selector.name, "value": value}
            row.update(labels)
            rows.append(row)

        if not rows:
            return pl.DataFrame({"__name__": [], "value": []})
        return pl.DataFrame(rows)

    # -------------------------------------------------------------------
    # Internal: range evaluation
    # -------------------------------------------------------------------

    def _evaluate_range(
        self,
        expr: PromQLExpr,
        start: datetime,
        end: datetime,
        step: timedelta,
    ) -> pl.DataFrame:
        if isinstance(expr, MetricSelector):
            return self._range_selector(expr, start=start, end=end, step=step)

        if isinstance(expr, CompareExpr):
            df = self._range_selector(expr.selector, start=start, end=end, step=step)
            return self._filter_by_compare(df, expr.op, expr.threshold)

        raise ValueError(
            f"range_query not yet supported for expression type: {type(expr)}"
        )

    def _range_selector(
        self,
        selector: MetricSelector,
        start: datetime,
        end: datetime,
        step: timedelta,
    ) -> pl.DataFrame:
        rows: list[dict] = []
        for labels, samples in self._iter_matching_series(selector):
            for sample in samples:
                if sample.timestamp > end:
                    break
                if sample.timestamp >= start:
                    row: dict = {
                        "__name__": selector.name,
                        "timestamp": sample.timestamp,
                        "value": sample.value,
                    }
                    row.update(labels)
                    rows.append(row)

        if not rows:
            return pl.DataFrame({"__name__": [], "timestamp": [], "value": []})
        return pl.DataFrame(rows)

    # -------------------------------------------------------------------
    # Internal: eviction
    # -------------------------------------------------------------------

    def _maybe_evict(self) -> None:
        now = datetime.utcnow()
        evict_interval = self._config.retention / 10
        if (
            self._last_eviction_time is not None
            and now - self._last_eviction_time < evict_interval
        ):
            return
        self._last_eviction_time = now
        self._evict_expired()

    def _evict_expired(self) -> None:
        cutoff = datetime.utcnow() - self._config.retention
        empty_keys: list[_SeriesKey] = []

        for key, samples in self._series.items():
            while samples and samples[0].timestamp < cutoff:
                samples.popleft()
            if not samples:
                empty_keys.append(key)

        for key in empty_keys:
            metric_name, _ = key
            del self._series[key]
            self._label_maps.pop(key, None)
            index_list = self._name_index.get(metric_name)
            if index_list is not None:
                try:
                    index_list.remove(key)
                except ValueError:
                    pass
                if not index_list:
                    del self._name_index[metric_name]


# ---------------------------------------------------------------------------
# Range function evaluation
# ---------------------------------------------------------------------------


def _apply_range_function(
    func_name: str,
    samples: list[_TimeSeriesSample],
) -> float:
    if func_name == "count_over_time":
        return float(len(samples))

    if func_name == "changes":
        if len(samples) < 2:
            return 0.0
        changes = sum(
            1
            for i in range(1, len(samples))
            if samples[i].value != samples[i - 1].value
        )
        return float(changes)

    if func_name == "min_over_time":
        return min(s.value for s in samples)

    if func_name == "max_over_time":
        return max(s.value for s in samples)

    if func_name == "avg_over_time":
        return sum(s.value for s in samples) / len(samples)

    raise ValueError(f"Unknown range function: {func_name}")
