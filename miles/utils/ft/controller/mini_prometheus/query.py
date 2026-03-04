from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterator

import polars as pl

_SeriesKey = tuple[str, frozenset[tuple[str, str]]]

_EMPTY_INSTANT = pl.DataFrame({"__name__": [], "value": []})
_EMPTY_RANGE = pl.DataFrame({"__name__": [], "timestamp": [], "value": []})


@dataclass
class TimeSeriesSample:
    timestamp: datetime
    value: float


# ---------------------------------------------------------------------------
# Public query functions (typed API — no PromQL parsing)
# ---------------------------------------------------------------------------


def query_latest(
    series: dict[_SeriesKey, deque[TimeSeriesSample]],
    label_maps: dict[_SeriesKey, dict[str, str]],
    name_index: dict[str, set[_SeriesKey]],
    metric_name: str,
    label_filters: dict[str, str] | None = None,
) -> pl.DataFrame:
    rows: list[dict] = []
    for labels, samples in _iter_matching(series, label_maps, name_index, metric_name, label_filters):
        latest = samples[-1]
        row: dict = {"__name__": metric_name, "value": latest.value}
        row.update(labels)
        rows.append(row)

    if not rows:
        return _EMPTY_INSTANT
    return pl.DataFrame(rows)


def query_range(
    series: dict[_SeriesKey, deque[TimeSeriesSample]],
    label_maps: dict[_SeriesKey, dict[str, str]],
    name_index: dict[str, set[_SeriesKey]],
    metric_name: str,
    window: timedelta,
    label_filters: dict[str, str] | None = None,
) -> pl.DataFrame:
    now = datetime.now(timezone.utc)
    start = now - window
    rows: list[dict] = []

    for labels, samples in _iter_matching(series, label_maps, name_index, metric_name, label_filters):
        for sample in samples:
            if sample.timestamp > now:
                break
            if sample.timestamp >= start:
                row: dict = {
                    "__name__": metric_name,
                    "timestamp": sample.timestamp,
                    "value": sample.value,
                }
                row.update(labels)
                rows.append(row)

    if not rows:
        return _EMPTY_RANGE
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _labels_match(labels: dict[str, str], filters: dict[str, str]) -> bool:
    return all(labels.get(k) == v for k, v in filters.items())


def _iter_matching(
    series: dict[_SeriesKey, deque[TimeSeriesSample]],
    label_maps: dict[_SeriesKey, dict[str, str]],
    name_index: dict[str, set[_SeriesKey]],
    metric_name: str,
    label_filters: dict[str, str] | None,
) -> Iterator[tuple[dict[str, str], deque[TimeSeriesSample]]]:
    for key in name_index.get(metric_name, []):
        samples = series.get(key)
        if not samples:
            continue

        labels = label_maps[key]
        if label_filters and not _labels_match(labels, label_filters):
            continue

        yield labels, samples


def range_aggregate(
    series: dict[_SeriesKey, deque[TimeSeriesSample]],
    label_maps: dict[_SeriesKey, dict[str, str]],
    name_index: dict[str, set[_SeriesKey]],
    func_name: str,
    metric_name: str,
    window: timedelta,
    label_filters: dict[str, str] | None,
) -> pl.DataFrame:
    now = datetime.now(timezone.utc)
    window_start = now - window
    rows: list[dict] = []

    for labels, samples in _iter_matching(series, label_maps, name_index, metric_name, label_filters):
        window_samples = [s for s in samples if s.timestamp >= window_start]
        if not window_samples:
            continue

        value = _compute_aggregate(func_name, window_samples)
        row: dict = {"__name__": metric_name, "value": value}
        row.update(labels)
        rows.append(row)

    if not rows:
        return _EMPTY_INSTANT
    return pl.DataFrame(rows)


def _compute_aggregate(func_name: str, samples: list[TimeSeriesSample]) -> float:
    if func_name == "count_over_time":
        return float(len(samples))

    if func_name == "changes":
        if len(samples) < 2:
            return 0.0
        return float(sum(
            1
            for i in range(1, len(samples))
            if samples[i].value != samples[i - 1].value
        ))

    if func_name == "min_over_time":
        return min(s.value for s in samples)

    if func_name == "max_over_time":
        return max(s.value for s in samples)

    if func_name == "avg_over_time":
        return sum(s.value for s in samples) / len(samples)

    raise ValueError(f"Unknown range function: {func_name}")
