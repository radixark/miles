from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import polars as pl

logger = logging.getLogger(__name__)

SeriesKey = tuple[str, frozenset[tuple[str, str]]]

EMPTY_INSTANT = pl.DataFrame({"__name__": pl.Series([], dtype=pl.Utf8), "value": pl.Series([], dtype=pl.Float64)})
EMPTY_RANGE = pl.DataFrame(
    {
        "__name__": pl.Series([], dtype=pl.Utf8),
        "timestamp": pl.Series([], dtype=pl.Datetime("us", "UTC")),
        "value": pl.Series([], dtype=pl.Float64),
    }
)


@dataclass
class TimeSeriesSample:
    timestamp: datetime
    value: float


class AmbiguousSeriesError(Exception):
    """Raised when query_single_latest matches more than one series."""

    def __init__(self, metric_name: str, label_filters: dict[str, str] | None, matched_count: int) -> None:
        self.metric_name = metric_name
        self.label_filters = label_filters
        self.matched_count = matched_count
        super().__init__(
            f"query_single_latest matched {matched_count} series "
            f"for metric={metric_name!r} label_filters={label_filters!r}"
        )


# ---------------------------------------------------------------------------
# Public query functions (typed API — no PromQL parsing)
# ---------------------------------------------------------------------------


def query_single_latest(
    series: dict[SeriesKey, deque[TimeSeriesSample]],
    label_maps: dict[SeriesKey, dict[str, str]],
    name_index: dict[str, set[SeriesKey]],
    metric_name: str,
    label_filters: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Like query_latest, but enforces that at most one series matches.

    Returns EMPTY_INSTANT if zero series match.
    Raises AmbiguousSeriesError if more than one series matches.
    """
    df = _instant_query(
        series,
        label_maps,
        name_index,
        metric_name,
        label_filters,
        value_fn=lambda samples: samples[-1].value,
    )
    if len(df) > 1:
        logger.error(
            "mini_prom: query_single_latest ambiguous metric=%s, filters=%s, matched=%d",
            metric_name,
            label_filters,
            len(df),
        )
        raise AmbiguousSeriesError(
            metric_name=metric_name,
            label_filters=label_filters,
            matched_count=len(df),
        )
    return df


def query_latest(
    series: dict[SeriesKey, deque[TimeSeriesSample]],
    label_maps: dict[SeriesKey, dict[str, str]],
    name_index: dict[str, set[SeriesKey]],
    metric_name: str,
    label_filters: dict[str, str] | None = None,
) -> pl.DataFrame:
    return _instant_query(
        series,
        label_maps,
        name_index,
        metric_name,
        label_filters,
        value_fn=lambda samples: samples[-1].value,
    )


def query_range(
    series: dict[SeriesKey, deque[TimeSeriesSample]],
    label_maps: dict[SeriesKey, dict[str, str]],
    name_index: dict[str, set[SeriesKey]],
    metric_name: str,
    window: timedelta,
    label_filters: dict[str, str] | None = None,
) -> pl.DataFrame:
    now = datetime.now(timezone.utc)
    start = now - window
    rows: list[dict] = []

    for labels, samples in _iter_matching(series, label_maps, name_index, metric_name, label_filters):
        rows.extend(_collect_range_rows_for_series(metric_name, labels, samples, now=now, start=start))

    if not rows:
        return EMPTY_RANGE
    return pl.DataFrame(rows)


def range_aggregate(
    series: dict[SeriesKey, deque[TimeSeriesSample]],
    label_maps: dict[SeriesKey, dict[str, str]],
    name_index: dict[str, set[SeriesKey]],
    func_name: str,
    metric_name: str,
    window: timedelta,
    label_filters: dict[str, str] | None,
) -> pl.DataFrame:
    now = datetime.now(timezone.utc)
    window_start = now - window

    def _extract(samples: deque[TimeSeriesSample]) -> float | None:
        window_samples: list[TimeSeriesSample] = []
        for s in reversed(samples):
            if s.timestamp > now:
                continue
            if s.timestamp < window_start:
                break
            window_samples.append(s)

        if not window_samples:
            return None

        window_samples.reverse()
        return _compute_aggregate(func_name, window_samples)

    return _instant_query(
        series,
        label_maps,
        name_index,
        metric_name,
        label_filters,
        value_fn=_extract,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_range_rows_for_series(
    metric_name: str,
    labels: dict[str, str],
    samples: deque[TimeSeriesSample],
    *,
    now: datetime,
    start: datetime,
) -> list[dict]:
    rows: list[dict] = []
    for sample in reversed(samples):
        if sample.timestamp > now:
            continue
        if sample.timestamp < start:
            break

        row: dict = {
            "__name__": metric_name,
            "timestamp": sample.timestamp,
            "value": sample.value,
        }
        row.update(labels)
        rows.append(row)
    return rows


def _instant_query(
    series: dict[SeriesKey, deque[TimeSeriesSample]],
    label_maps: dict[SeriesKey, dict[str, str]],
    name_index: dict[str, set[SeriesKey]],
    metric_name: str,
    label_filters: dict[str, str] | None,
    value_fn: Callable[[deque[TimeSeriesSample]], float | None],
) -> pl.DataFrame:
    rows: list[dict] = []
    for labels, samples in _iter_matching(series, label_maps, name_index, metric_name, label_filters):
        value = value_fn(samples)
        if value is None:
            continue

        row: dict = {"__name__": metric_name, "value": value}
        row.update(labels)
        rows.append(row)

    if not rows:
        return EMPTY_INSTANT
    return pl.DataFrame(rows)


def _iter_matching(
    series: dict[SeriesKey, deque[TimeSeriesSample]],
    label_maps: dict[SeriesKey, dict[str, str]],
    name_index: dict[str, set[SeriesKey]],
    metric_name: str,
    label_filters: dict[str, str] | None,
) -> Iterator[tuple[dict[str, str], deque[TimeSeriesSample]]]:
    for key in list(name_index.get(metric_name, [])):
        samples = series.get(key)
        if not samples:
            continue

        labels = label_maps[key]
        if label_filters and not _labels_match(labels, label_filters):
            continue

        yield labels, deque(samples)


def _labels_match(labels: dict[str, str], filters: dict[str, str]) -> bool:
    return all(labels.get(k) == v for k, v in filters.items())


def _compute_aggregate(func_name: str, samples: list[TimeSeriesSample]) -> float:
    if func_name == "count_over_time":
        return float(len(samples))

    if func_name == "changes":
        if len(samples) < 2:
            return 0.0
        return float(sum(1 for i in range(1, len(samples)) if samples[i].value != samples[i - 1].value))

    if func_name == "avg_over_time":
        return sum(s.value for s in samples) / len(samples)

    raise ValueError(f"Unknown range function: {func_name}")
