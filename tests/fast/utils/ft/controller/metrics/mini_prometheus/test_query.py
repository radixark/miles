"""Direct unit tests for mini_prometheus/query.py functions."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from miles.utils.ft.controller.metrics.mini_prometheus.query import (
    EMPTY_INSTANT,
    EMPTY_RANGE,
    SeriesKey,
    TimeSeriesSample,
    _compute_aggregate,
    _labels_match,
    query_latest,
    query_range,
    range_aggregate,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _key(name: str, **labels: str) -> SeriesKey:
    return (name, frozenset(labels.items()))


def _build_series(
    entries: list[tuple[SeriesKey, list[tuple[float, float]]]],
) -> tuple[
    dict[SeriesKey, deque[TimeSeriesSample]],
    dict[SeriesKey, dict[str, str]],
    dict[str, set[SeriesKey]],
]:
    """Build the three data structures from (key, [(offset_seconds, value)]) pairs.

    offset_seconds is relative to now (negative = in the past).
    """
    now = _now()
    series: dict[SeriesKey, deque[TimeSeriesSample]] = {}
    label_maps: dict[SeriesKey, dict[str, str]] = {}
    name_index: dict[str, set[SeriesKey]] = {}

    for key, samples_raw in entries:
        name, label_set = key
        labels = dict(label_set)
        label_maps[key] = labels
        name_index.setdefault(name, set()).add(key)

        dq: deque[TimeSeriesSample] = deque()
        for offset_seconds, value in sorted(samples_raw, key=lambda x: x[0]):
            dq.append(
                TimeSeriesSample(
                    timestamp=now + timedelta(seconds=offset_seconds),
                    value=value,
                )
            )
        series[key] = dq

    return series, label_maps, name_index


# ===================================================================
# _labels_match
# ===================================================================


class TestLabelsMatch:
    def test_exact_match(self) -> None:
        assert _labels_match({"a": "1", "b": "2"}, {"a": "1", "b": "2"})

    def test_subset_filter(self) -> None:
        assert _labels_match({"a": "1", "b": "2"}, {"a": "1"})

    def test_empty_filter_always_matches(self) -> None:
        assert _labels_match({"a": "1"}, {})

    def test_mismatch(self) -> None:
        assert not _labels_match({"a": "1"}, {"a": "2"})

    def test_missing_key(self) -> None:
        assert not _labels_match({"a": "1"}, {"b": "1"})


# ===================================================================
# query_latest
# ===================================================================


class TestQueryLatest:
    def test_returns_latest_value(self) -> None:
        key = _key("metric_a", node="n1")
        series, lm, ni = _build_series(
            [
                (key, [(-10, 1.0), (-5, 2.0), (-1, 3.0)]),
            ]
        )

        df = query_latest(series, lm, ni, metric_name="metric_a")

        assert len(df) == 1
        assert df["value"][0] == 3.0

    def test_no_matching_metric_returns_empty(self) -> None:
        series, lm, ni = _build_series([])

        df = query_latest(series, lm, ni, metric_name="nonexistent")

        assert df.shape == EMPTY_INSTANT.shape

    def test_label_filter_narrows_results(self) -> None:
        k1 = _key("m", node="n1")
        k2 = _key("m", node="n2")
        series, lm, ni = _build_series(
            [
                (k1, [(-1, 10.0)]),
                (k2, [(-1, 20.0)]),
            ]
        )

        df = query_latest(series, lm, ni, metric_name="m", label_filters={"node": "n2"})

        assert len(df) == 1
        assert df["value"][0] == 20.0

    def test_empty_deque_is_skipped(self) -> None:
        """A key exists in name_index but its deque is empty."""
        key = _key("m", node="n1")
        series, lm, ni = _build_series([(key, [(-1, 1.0)])])
        series[key].clear()

        df = query_latest(series, lm, ni, metric_name="m")

        assert df.shape == EMPTY_INSTANT.shape


# ===================================================================
# query_range
# ===================================================================


class TestQueryRange:
    def test_returns_samples_within_window(self) -> None:
        key = _key("m", node="n1")
        series, lm, ni = _build_series(
            [
                (key, [(-120, 0.5), (-30, 1.0), (-5, 2.0)]),
            ]
        )

        df = query_range(series, lm, ni, metric_name="m", window=timedelta(minutes=1))

        assert len(df) == 2
        values = sorted(df["value"].to_list())
        assert values == [1.0, 2.0]

    def test_future_timestamp_samples_are_skipped(self) -> None:
        key = _key("m", node="n1")
        series, lm, ni = _build_series(
            [
                (key, [(-5, 1.0), (60, 99.0)]),
            ]
        )

        df = query_range(series, lm, ni, metric_name="m", window=timedelta(minutes=5))

        assert len(df) == 1
        assert df["value"][0] == 1.0

    def test_empty_window_returns_empty(self) -> None:
        key = _key("m", node="n1")
        series, lm, ni = _build_series(
            [
                (key, [(-600, 1.0)]),
            ]
        )

        df = query_range(series, lm, ni, metric_name="m", window=timedelta(seconds=1))

        assert df.shape == EMPTY_RANGE.shape


# ===================================================================
# _compute_aggregate
# ===================================================================


class TestComputeAggregate:
    def test_count_over_time(self) -> None:
        samples = [TimeSeriesSample(timestamp=_now(), value=v) for v in [1, 2, 3]]
        assert _compute_aggregate("count_over_time", samples) == 3.0

    def test_changes_with_multiple_values(self) -> None:
        samples = [TimeSeriesSample(timestamp=_now(), value=v) for v in [1.0, 1.0, 2.0, 2.0, 3.0]]
        assert _compute_aggregate("changes", samples) == 2.0

    def test_changes_with_single_sample_returns_zero(self) -> None:
        samples = [TimeSeriesSample(timestamp=_now(), value=1.0)]
        assert _compute_aggregate("changes", samples) == 0.0

    def test_avg_over_time(self) -> None:
        samples = [TimeSeriesSample(timestamp=_now(), value=v) for v in [2.0, 4.0, 6.0]]
        assert _compute_aggregate("avg_over_time", samples) == 4.0

    def test_unknown_function_raises_value_error(self) -> None:
        samples = [TimeSeriesSample(timestamp=_now(), value=1.0)]
        with pytest.raises(ValueError, match="Unknown range function"):
            _compute_aggregate("bogus_func", samples)


# ===================================================================
# range_aggregate
# ===================================================================


class TestRangeAggregate:
    def test_avg_over_time_via_range_aggregate(self) -> None:
        key = _key("m", node="n1")
        series, lm, ni = _build_series(
            [
                (key, [(-10, 2.0), (-5, 4.0), (-1, 6.0)]),
            ]
        )

        df = range_aggregate(
            series,
            lm,
            ni,
            func_name="avg_over_time",
            metric_name="m",
            window=timedelta(minutes=1),
            label_filters=None,
        )

        assert len(df) == 1
        assert df["value"][0] == 4.0

    def test_no_samples_in_window_returns_empty(self) -> None:
        key = _key("m", node="n1")
        series, lm, ni = _build_series(
            [
                (key, [(-600, 1.0)]),
            ]
        )

        df = range_aggregate(
            series,
            lm,
            ni,
            func_name="count_over_time",
            metric_name="m",
            window=timedelta(seconds=1),
            label_filters=None,
        )

        assert df.shape == EMPTY_INSTANT.shape


# ===================================================================
# EMPTY_RANGE / query_range timestamp dtype consistency (M-1)
# ===================================================================


class TestTimestampDtypeConsistency:
    """M-1: EMPTY_RANGE used Float64 for timestamp while non-empty results
    used Datetime. Both must use Datetime("us", "UTC") for consistency."""

    def test_empty_range_has_datetime_dtype(self) -> None:
        assert EMPTY_RANGE["timestamp"].dtype == pl.Datetime("us", "UTC")

    def test_non_empty_query_range_has_datetime_dtype(self) -> None:
        key = _key("m", node="n1")
        series, lm, ni = _build_series(
            [
                (key, [(-5, 1.0), (-1, 2.0)]),
            ]
        )
        df = query_range(series, lm, ni, metric_name="m", window=timedelta(minutes=1))
        assert len(df) > 0
        assert df["timestamp"].dtype == pl.Datetime("us", "UTC")
