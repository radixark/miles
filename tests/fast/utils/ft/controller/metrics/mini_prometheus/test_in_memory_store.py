"""Tests for InMemoryMetricStore: ingest, query_latest, query_range, aggregations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from miles.utils.ft.agents.types import CounterSample, GaugeSample
from miles.utils.ft.controller.metrics.mini_prometheus.in_memory_store import (
    InMemoryMetricStore,
    OutOfOrderSampleError,
)


def _ts(seconds: int) -> datetime:
    return datetime(2026, 1, 1, 0, 0, seconds, tzinfo=timezone.utc)


class TestIngestSamples:
    def test_gauge_sample_stores_value(self) -> None:
        store = InMemoryMetricStore()
        store.ingest_samples(
            target_id="node-0",
            samples=[GaugeSample(name="temp", labels={"gpu": "0"}, value=75.0)],
            timestamp=_ts(10),
        )
        df = store.query_latest("temp")
        assert len(df) == 1
        assert df["value"][0] == 75.0

    def test_counter_sample_stores_delta(self) -> None:
        store = InMemoryMetricStore()
        store.ingest_samples(
            target_id="node-0",
            samples=[CounterSample(name="errors", labels={"gpu": "0"}, delta=3.0)],
            timestamp=_ts(10),
        )
        df = store.query_latest("errors")
        assert len(df) == 1
        assert df["value"][0] == 3.0

    def test_node_id_label_injected_by_default(self) -> None:
        store = InMemoryMetricStore()
        store.ingest_samples(
            target_id="node-42",
            samples=[GaugeSample(name="temp", labels={}, value=1.0)],
            timestamp=_ts(10),
        )
        df = store.query_latest("temp")
        assert df["node_id"][0] == "node-42"

    def test_existing_node_id_label_not_overwritten(self) -> None:
        store = InMemoryMetricStore()
        store.ingest_samples(
            target_id="node-42",
            samples=[GaugeSample(name="temp", labels={"node_id": "custom"}, value=1.0)],
            timestamp=_ts(10),
        )
        df = store.query_latest("temp")
        assert df["node_id"][0] == "custom"

    def test_multiple_series_distinguished_by_labels(self) -> None:
        store = InMemoryMetricStore()
        store.ingest_samples(
            target_id="node-0",
            samples=[
                GaugeSample(name="temp", labels={"gpu": "0"}, value=70.0),
                GaugeSample(name="temp", labels={"gpu": "1"}, value=80.0),
            ],
            timestamp=_ts(10),
        )
        df = store.query_latest("temp")
        assert len(df) == 2
        values = sorted(df["value"].to_list())
        assert values == [70.0, 80.0]


class TestOutOfOrderSampleRejection:
    """ingest_samples() used to silently accept out-of-order timestamps.
    Query functions assumed monotonic append order (e.g. samples[-1] is latest,
    reversed() iteration breaks on old timestamps), so out-of-order writes would
    silently corrupt query results. Now ingest rejects them with OutOfOrderSampleError."""

    def test_in_order_writes_succeed(self) -> None:
        store = InMemoryMetricStore()
        store.ingest_samples(
            target_id="n",
            samples=[GaugeSample(name="m", labels={}, value=1.0)],
            timestamp=_ts(10),
        )
        store.ingest_samples(
            target_id="n",
            samples=[GaugeSample(name="m", labels={}, value=2.0)],
            timestamp=_ts(20),
        )
        df = store.query_latest("m")
        assert df["value"][0] == 2.0

    def test_equal_timestamp_allowed(self) -> None:
        store = InMemoryMetricStore()
        store.ingest_samples(
            target_id="n",
            samples=[GaugeSample(name="m", labels={}, value=1.0)],
            timestamp=_ts(10),
        )
        store.ingest_samples(
            target_id="n",
            samples=[GaugeSample(name="m", labels={}, value=2.0)],
            timestamp=_ts(10),
        )
        df = store.query_latest("m")
        assert df["value"][0] == 2.0

    def test_out_of_order_raises_error(self) -> None:
        store = InMemoryMetricStore()
        store.ingest_samples(
            target_id="n",
            samples=[GaugeSample(name="m", labels={}, value=1.0)],
            timestamp=_ts(20),
        )

        with pytest.raises(OutOfOrderSampleError, match="out-of-order"):
            store.ingest_samples(
                target_id="n",
                samples=[GaugeSample(name="m", labels={}, value=2.0)],
                timestamp=_ts(10),
            )

    def test_out_of_order_does_not_corrupt_existing_data(self) -> None:
        store = InMemoryMetricStore()
        store.ingest_samples(
            target_id="n",
            samples=[GaugeSample(name="m", labels={}, value=1.0)],
            timestamp=_ts(10),
        )
        store.ingest_samples(
            target_id="n",
            samples=[GaugeSample(name="m", labels={}, value=2.0)],
            timestamp=_ts(20),
        )

        with pytest.raises(OutOfOrderSampleError):
            store.ingest_samples(
                target_id="n",
                samples=[GaugeSample(name="m", labels={}, value=999.0)],
                timestamp=_ts(15),
            )

        df = store.query_latest("m")
        assert df["value"][0] == 2.0

    def test_different_series_independent(self) -> None:
        """Out-of-order check is per-series, not global."""
        store = InMemoryMetricStore()
        store.ingest_samples(
            target_id="n",
            samples=[GaugeSample(name="m", labels={"gpu": "0"}, value=1.0)],
            timestamp=_ts(20),
        )
        store.ingest_samples(
            target_id="n",
            samples=[GaugeSample(name="m", labels={"gpu": "1"}, value=2.0)],
            timestamp=_ts(10),
        )
        df = store.query_latest("m")
        assert len(df) == 2

    def test_first_sample_in_series_always_accepted(self) -> None:
        store = InMemoryMetricStore()
        store.ingest_samples(
            target_id="n",
            samples=[GaugeSample(name="m", labels={}, value=1.0)],
            timestamp=_ts(1),
        )
        df = store.query_latest("m")
        assert df["value"][0] == 1.0


class TestQueryLatest:
    def test_returns_most_recent_value(self) -> None:
        store = InMemoryMetricStore()
        store.ingest_samples(
            target_id="n",
            samples=[GaugeSample(name="m", labels={}, value=1.0)],
            timestamp=_ts(1),
        )
        store.ingest_samples(
            target_id="n",
            samples=[GaugeSample(name="m", labels={}, value=9.0)],
            timestamp=_ts(2),
        )
        df = store.query_latest("m")
        assert df["value"][0] == 9.0

    def test_nonexistent_metric_returns_empty(self) -> None:
        store = InMemoryMetricStore()
        df = store.query_latest("does_not_exist")
        assert len(df) == 0

    def test_label_filter_narrows_results(self) -> None:
        store = InMemoryMetricStore()
        store.ingest_samples(
            target_id="n",
            samples=[
                GaugeSample(name="m", labels={"gpu": "0"}, value=1.0),
                GaugeSample(name="m", labels={"gpu": "1"}, value=2.0),
            ],
            timestamp=_ts(10),
        )
        df = store.query_latest("m", label_filters={"gpu": "1"})
        assert len(df) == 1
        assert df["value"][0] == 2.0


class TestQueryRange:
    def test_returns_samples_within_window(self) -> None:
        store = InMemoryMetricStore()
        now = datetime.now(timezone.utc)
        store.ingest_samples(
            target_id="n",
            samples=[GaugeSample(name="m", labels={}, value=10.0)],
            timestamp=now - timedelta(seconds=5),
        )
        store.ingest_samples(
            target_id="n",
            samples=[GaugeSample(name="m", labels={}, value=20.0)],
            timestamp=now - timedelta(seconds=1),
        )
        df = store.query_range("m", window=timedelta(seconds=30))
        assert len(df) == 2

    def test_empty_store_returns_empty_dataframe(self) -> None:
        store = InMemoryMetricStore()
        df = store.query_range("m", window=timedelta(seconds=30))
        assert len(df) == 0


class TestRangeAggregations:
    def _make_store_with_series(self) -> InMemoryMetricStore:
        store = InMemoryMetricStore()
        now = datetime.now(timezone.utc)
        for i, v in enumerate([1.0, 2.0, 2.0, 5.0]):
            store.ingest_samples(
                target_id="n",
                samples=[GaugeSample(name="m", labels={}, value=v)],
                timestamp=now - timedelta(seconds=10 - i),
            )
        return store

    def test_count_over_time(self) -> None:
        store = self._make_store_with_series()
        df = store.count_over_time("m", window=timedelta(seconds=30))
        assert len(df) == 1
        assert df["value"][0] == 4.0

    def test_avg_over_time(self) -> None:
        store = self._make_store_with_series()
        df = store.avg_over_time("m", window=timedelta(seconds=30))
        assert len(df) == 1
        assert df["value"][0] == pytest.approx(2.5)

    def test_changes(self) -> None:
        store = self._make_store_with_series()
        df = store.changes("m", window=timedelta(seconds=30))
        assert len(df) == 1
        assert df["value"][0] == 2.0  # 1->2 and 2->5 are changes; 2->2 is not


class TestQueryIterationSafetyAgainstConcurrentMutation:
    """_iter_matching used to iterate the original name_index set and yield original
    deque references. If eviction ran concurrently (across an asyncio await point),
    index_set.discard() or del _series[key] would crash iteration with RuntimeError.
    Fix: _iter_matching snapshots the index set with list() and yields deque copies."""

    def test_iter_matching_yields_deque_snapshot_not_reference(self) -> None:
        store = InMemoryMetricStore()
        now = datetime.now(timezone.utc)
        store.ingest_samples(
            target_id="n",
            samples=[GaugeSample(name="m", labels={}, value=1.0)],
            timestamp=now,
        )

        from miles.utils.ft.controller.metrics.mini_prometheus.query import _iter_matching

        for _labels, samples in _iter_matching(store._series, store._label_maps, store._name_index, "m", None):
            key = list(store._series.keys())[0]
            original = store._series[key]
            assert samples is not original
            assert list(samples) == list(original)

    def test_evict_during_query_does_not_corrupt_results(self) -> None:
        """Simulate: take query snapshot, then evict all data, then continue using snapshot."""
        store = InMemoryMetricStore()
        now = datetime.now(timezone.utc)
        store.ingest_samples(
            target_id="n",
            samples=[GaugeSample(name="m", labels={}, value=42.0)],
            timestamp=now,
        )

        df_before = store.query_latest("m")
        assert df_before["value"][0] == 42.0

        from miles.utils.ft.controller.metrics.mini_prometheus.eviction import RetentionEvictor

        evictor = RetentionEvictor(store=store, retention=timedelta(seconds=0))
        evictor._evict_expired()

        assert len(store._series) == 0
        df_after = store.query_latest("m")
        assert len(df_after) == 0

    def test_concurrent_ingest_evict_query_stress(self) -> None:
        """Stress test: interleave ingest (with eviction) and query_latest for many
        rounds. Before the fix, this could raise RuntimeError on set/dict mutation."""
        store = InMemoryMetricStore()
        from miles.utils.ft.controller.metrics.mini_prometheus.eviction import RetentionEvictor

        evictor = RetentionEvictor(store=store, retention=timedelta(seconds=1))

        for i in range(500):
            ts = datetime.now(timezone.utc) - timedelta(seconds=max(0, 5 - i * 0.01))
            store.ingest_samples(
                target_id="n",
                samples=[GaugeSample(name="m", labels={"i": str(i % 10)}, value=float(i))],
                timestamp=ts,
            )
            evictor.maybe_evict()
            store.query_latest("m")
            store.query_range("m", window=timedelta(seconds=30))
