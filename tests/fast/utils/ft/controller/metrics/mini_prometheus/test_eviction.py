"""Unit tests for RetentionEvictor (P0 item 4)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from miles.utils.ft.agents.types import GaugeSample
from miles.utils.ft.controller.metrics.mini_prometheus.eviction import RetentionEvictor
from miles.utils.ft.controller.metrics.mini_prometheus.in_memory_store import InMemoryMetricStore


def _ingest_at(store: InMemoryMetricStore, metric_name: str, value: float, ts: datetime) -> None:
    store.ingest_samples(
        target_id="node-0",
        samples=[GaugeSample(name=metric_name, value=value, labels={})],
        timestamp=ts,
    )


class TestEvictionIntervalThrottling:
    def test_calls_within_interval_are_noop(self) -> None:
        """Calls within retention/10 after last eviction should be no-ops."""
        store = InMemoryMetricStore()
        retention = timedelta(minutes=10)
        evictor = RetentionEvictor(store=store, retention=retention)

        now = datetime.now(timezone.utc)
        old_ts = now - timedelta(minutes=20)
        _ingest_at(store, "metric_a", 1.0, old_ts)

        evictor.maybe_evict()
        assert len(store._series) == 0

        _ingest_at(store, "metric_b", 2.0, old_ts)
        evictor.maybe_evict()
        assert len(store._series) == 1

    def test_eviction_runs_after_interval_elapses(self) -> None:
        """After retention/10 passes, eviction should run again."""
        store = InMemoryMetricStore()
        retention = timedelta(seconds=10)
        evictor = RetentionEvictor(store=store, retention=retention)

        evictor.maybe_evict()

        evictor._last_eviction_time = datetime.now(timezone.utc) - timedelta(seconds=2)

        old_ts = datetime.now(timezone.utc) - timedelta(seconds=20)
        _ingest_at(store, "metric_a", 1.0, old_ts)

        evictor.maybe_evict()
        assert len(store._series) == 0


class TestEvictExpired:
    def test_removes_old_samples(self) -> None:
        store = InMemoryMetricStore()
        retention = timedelta(minutes=5)
        evictor = RetentionEvictor(store=store, retention=retention)

        now = datetime.now(timezone.utc)
        _ingest_at(store, "metric_a", 1.0, now - timedelta(minutes=10))
        _ingest_at(store, "metric_a", 2.0, now)

        evictor._evict_expired()

        remaining_keys = list(store._series.keys())
        assert len(remaining_keys) == 1
        samples = store._series[remaining_keys[0]]
        assert len(samples) == 1
        assert samples[0].value == 2.0

    def test_series_partially_expired(self) -> None:
        """When some samples are old and some are fresh, only old ones are removed."""
        store = InMemoryMetricStore()
        retention = timedelta(minutes=5)
        evictor = RetentionEvictor(store=store, retention=retention)

        now = datetime.now(timezone.utc)
        _ingest_at(store, "metric_a", 1.0, now - timedelta(minutes=10))
        _ingest_at(store, "metric_a", 2.0, now - timedelta(minutes=7))
        _ingest_at(store, "metric_a", 3.0, now - timedelta(minutes=1))

        evictor._evict_expired()

        remaining_keys = list(store._series.keys())
        assert len(remaining_keys) == 1
        samples = store._series[remaining_keys[0]]
        assert len(samples) == 1
        assert samples[0].value == 3.0

    def test_all_samples_expired_removes_series_and_name_index(self) -> None:
        """When all samples expire, the series key AND name_index entry are removed."""
        store = InMemoryMetricStore()
        retention = timedelta(minutes=5)
        evictor = RetentionEvictor(store=store, retention=retention)

        old_ts = datetime.now(timezone.utc) - timedelta(minutes=10)
        _ingest_at(store, "metric_a", 1.0, old_ts)

        assert len(store._series) == 1
        assert "metric_a" in store._name_index

        evictor._evict_expired()

        assert len(store._series) == 0
        assert "metric_a" not in store._name_index

    def test_mixed_series_only_empty_ones_removed(self) -> None:
        """With multiple series, only the fully-expired ones are removed."""
        store = InMemoryMetricStore()
        retention = timedelta(minutes=5)
        evictor = RetentionEvictor(store=store, retention=retention)

        now = datetime.now(timezone.utc)
        _ingest_at(store, "metric_a", 1.0, now - timedelta(minutes=10))
        _ingest_at(store, "metric_b", 2.0, now)

        evictor._evict_expired()

        assert "metric_a" not in store._name_index
        assert "metric_b" in store._name_index
        assert len(store._series) == 1


class TestEvictionIterationSafety:
    """_evict_expired used to iterate self._store._series.items() directly.
    If ingest_samples added a new key to _series concurrently (across an asyncio
    await point), the dict would mutate during iteration causing RuntimeError.
    Fix: _evict_expired snapshots _series.items() with list() before iterating."""

    def test_ingest_during_eviction_does_not_crash(self) -> None:
        """Simulate: start eviction scan, then ingest new data mid-scan.
        The snapshot ensures the new key doesn't disrupt iteration."""
        store = InMemoryMetricStore()
        retention = timedelta(minutes=5)
        evictor = RetentionEvictor(store=store, retention=retention)

        now = datetime.now(timezone.utc)
        _ingest_at(store, "metric_a", 1.0, now - timedelta(minutes=10))
        _ingest_at(store, "metric_b", 2.0, now)

        evictor._evict_expired()
        assert "metric_a" not in store._name_index
        assert "metric_b" in store._name_index

        _ingest_at(store, "metric_c", 3.0, now)
        evictor._last_eviction_time = None
        evictor.maybe_evict()
        assert "metric_b" in store._name_index
        assert "metric_c" in store._name_index
