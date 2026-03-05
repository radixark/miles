from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from miles.utils.ft.controller.metrics.mini_prometheus.storage import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.models import MetricSample


class TestMiniPrometheusQueryLatest:
    def _make_store(self) -> MiniPrometheus:
        return MiniPrometheus(
            config=MiniPrometheusConfig(
                retention=timedelta(minutes=60),
            )
        )

    def test_simple_metric_query(self) -> None:
        store = self._make_store()
        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=75.0)],
        )

        df = store.query_latest("gpu_temp")
        assert len(df) == 1
        assert df["value"][0] == 75.0

    def test_latest_value_returned(self) -> None:
        store = self._make_store()
        t1 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 0, 0, 10, tzinfo=timezone.utc)

        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=70.0)],
            timestamp=t1,
        )
        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=80.0)],
            timestamp=t2,
        )

        df = store.query_latest("gpu_temp")
        assert len(df) == 1
        assert df["value"][0] == 80.0

    def test_filter_by_value_with_polars(self) -> None:
        store = self._make_store()
        store.ingest_samples(
            target_id="node-0",
            samples=[
                MetricSample(name="gpu_available", labels={"gpu": "0"}, value=1.0),
                MetricSample(name="gpu_available", labels={"gpu": "1"}, value=0.0),
            ],
        )

        df = store.query_latest("gpu_available").filter(pl.col("value") == 0.0)
        assert len(df) == 1
        assert df["gpu"][0] == "1"

    def test_label_filter(self) -> None:
        store = self._make_store()
        store.ingest_samples(
            target_id="node-0",
            samples=[
                MetricSample(name="xid_code_recent", labels={"xid": "48"}, value=1.0),
                MetricSample(name="xid_code_recent", labels={"xid": "31"}, value=1.0),
            ],
        )

        df = store.query_latest("xid_code_recent", label_filters={"xid": "48"})
        assert len(df) == 1
        assert df["xid"][0] == "48"

    def test_label_filter_by_node_id(self) -> None:
        store = self._make_store()
        store.ingest_samples(
            target_id="node-0",
            samples=[
                MetricSample(name="gpu_available", labels={"gpu": "0"}, value=1.0),
            ],
        )
        store.ingest_samples(
            target_id="node-1",
            samples=[
                MetricSample(name="gpu_available", labels={"gpu": "0"}, value=0.0),
            ],
        )

        df = store.query_latest("gpu_available", label_filters={"node_id": "node-1"})
        assert len(df) == 1
        assert df["node_id"][0] == "node-1"

    def test_multiple_targets(self) -> None:
        store = self._make_store()
        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=70.0)],
        )
        store.ingest_samples(
            target_id="node-1",
            samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=80.0)],
        )

        df = store.query_latest("gpu_temp")
        assert len(df) == 2

    def test_empty_result(self) -> None:
        store = self._make_store()
        df = store.query_latest("nonexistent_metric")
        assert df.is_empty()


class TestMiniPrometheusRangeFunctions:
    def _make_store_with_samples(self) -> MiniPrometheus:
        store = MiniPrometheus(
            config=MiniPrometheusConfig(
                retention=timedelta(minutes=60),
            )
        )
        now = datetime.now(timezone.utc)

        for i in range(5):
            store.ingest_samples(
                target_id="node-0",
                samples=[MetricSample(name="nic_alert", labels={}, value=1.0)],
                timestamp=now - timedelta(minutes=4 - i),
            )

        return store

    def test_count_over_time(self) -> None:
        store = self._make_store_with_samples()
        df = store.count_over_time("nic_alert", window=timedelta(minutes=5))
        assert len(df) == 1
        assert df["value"][0] == 5.0

    def test_count_over_time_shorter_window(self) -> None:
        store = self._make_store_with_samples()
        df = store.count_over_time("nic_alert", window=timedelta(minutes=2))
        assert len(df) == 1
        assert df["value"][0] <= 3.0

    def test_changes_no_change(self) -> None:
        store = MiniPrometheus(
            config=MiniPrometheusConfig(
                retention=timedelta(minutes=60),
            )
        )
        now = datetime.now(timezone.utc)

        for i in range(3):
            store.ingest_samples(
                target_id="node-0",
                samples=[
                    MetricSample(
                        name="miles_ft_training_iteration",
                        labels={"rank": "0"},
                        value=100.0,
                    )
                ],
                timestamp=now - timedelta(minutes=2 - i),
            )

        df = store.changes("miles_ft_training_iteration", window=timedelta(minutes=5))
        assert len(df) == 1
        assert df["value"][0] == 0.0

    def test_changes_with_actual_changes(self) -> None:
        store = MiniPrometheus(
            config=MiniPrometheusConfig(
                retention=timedelta(minutes=60),
            )
        )
        now = datetime.now(timezone.utc)

        values = [100.0, 101.0, 102.0, 102.0, 103.0]
        for i, val in enumerate(values):
            store.ingest_samples(
                target_id="node-0",
                samples=[
                    MetricSample(
                        name="miles_ft_training_iteration",
                        labels={"rank": "0"},
                        value=val,
                    )
                ],
                timestamp=now - timedelta(minutes=4 - i),
            )

        df = store.changes("miles_ft_training_iteration", window=timedelta(minutes=5))
        assert df["value"][0] == 3.0

    def test_count_over_time_with_polars_filter(self) -> None:
        store = self._make_store_with_samples()
        df = store.count_over_time("nic_alert", window=timedelta(minutes=5))
        filtered = df.filter(pl.col("value") >= 2.0)
        assert len(filtered) == 1
        assert filtered["value"][0] >= 2.0

    def test_min_over_time(self) -> None:
        store = MiniPrometheus(
            config=MiniPrometheusConfig(
                retention=timedelta(minutes=60),
            )
        )
        now = datetime.now(timezone.utc)

        for i, val in enumerate([75.0, 80.0, 70.0, 85.0]):
            store.ingest_samples(
                target_id="node-0",
                samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=val)],
                timestamp=now - timedelta(minutes=3 - i),
            )

        df = store.min_over_time("gpu_temp", window=timedelta(minutes=5))
        assert df["value"][0] == 70.0

    def test_avg_over_time(self) -> None:
        store = MiniPrometheus(
            config=MiniPrometheusConfig(
                retention=timedelta(minutes=60),
            )
        )
        now = datetime.now(timezone.utc)

        for i, val in enumerate([10.0, 20.0, 30.0]):
            store.ingest_samples(
                target_id="node-0",
                samples=[MetricSample(name="metric_a", labels={}, value=val)],
                timestamp=now - timedelta(minutes=2 - i),
            )

        df = store.avg_over_time("metric_a", window=timedelta(minutes=5))
        assert df["value"][0] == pytest.approx(20.0)

    def test_max_over_time(self) -> None:
        store = MiniPrometheus(
            config=MiniPrometheusConfig(
                retention=timedelta(minutes=60),
            )
        )
        now = datetime.now(timezone.utc)

        for i, val in enumerate([75.0, 80.0, 70.0, 85.0]):
            store.ingest_samples(
                target_id="node-0",
                samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=val)],
                timestamp=now - timedelta(minutes=3 - i),
            )

        df = store.max_over_time("gpu_temp", window=timedelta(minutes=5))
        assert df["value"][0] == 85.0

    def test_label_filter_with_polars_compare(self) -> None:
        store = MiniPrometheus(
            config=MiniPrometheusConfig(
                retention=timedelta(minutes=60),
            )
        )
        store.ingest_samples(
            target_id="node-0",
            samples=[
                MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=95.0),
                MetricSample(name="gpu_temp", labels={"gpu": "1"}, value=70.0),
            ],
        )

        df = store.query_latest("gpu_temp", label_filters={"gpu": "0"})
        filtered = df.filter(pl.col("value") > 90.0)
        assert len(filtered) == 1
        assert filtered["value"][0] == 95.0
        assert filtered["gpu"][0] == "0"


class TestMiniPrometheusQueryRange:
    def test_query_range_returns_time_series(self) -> None:
        store = MiniPrometheus(
            config=MiniPrometheusConfig(
                retention=timedelta(minutes=60),
            )
        )
        now = datetime.now(timezone.utc)
        t1 = now - timedelta(minutes=10)
        t2 = now - timedelta(minutes=5)
        t3 = now

        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=70.0)],
            timestamp=t1,
        )
        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=75.0)],
            timestamp=t2,
        )
        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=80.0)],
            timestamp=t3,
        )

        df = store.query_range("gpu_temp", window=timedelta(minutes=15))
        assert len(df) == 3
        values = sorted(df["value"].to_list())
        assert values == [70.0, 75.0, 80.0]

    def test_query_range_with_polars_filter(self) -> None:
        store = MiniPrometheus(
            config=MiniPrometheusConfig(
                retention=timedelta(minutes=60),
            )
        )
        now = datetime.now(timezone.utc)

        for i, val in enumerate([70.0, 75.0, 80.0, 85.0]):
            store.ingest_samples(
                target_id="node-0",
                samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=val)],
                timestamp=now - timedelta(minutes=3 - i),
            )

        df = store.query_range("gpu_temp", window=timedelta(minutes=5))
        filtered = df.filter(pl.col("value") > 72.0)
        assert len(filtered) == 3
        values = sorted(filtered["value"].to_list())
        assert values == [75.0, 80.0, 85.0]

    def test_empty_range_query(self) -> None:
        store = MiniPrometheus(
            config=MiniPrometheusConfig(
                retention=timedelta(minutes=60),
            )
        )

        df = store.query_range("nonexistent", window=timedelta(minutes=10))
        assert df.is_empty()
        assert "timestamp" in df.columns
        assert "value" in df.columns


class TestMiniPrometheusRetention:
    def test_expired_data_evicted(self) -> None:
        store = MiniPrometheus(
            config=MiniPrometheusConfig(
                retention=timedelta(minutes=5),
            )
        )
        now = datetime.now(timezone.utc)

        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=70.0)],
            timestamp=now - timedelta(minutes=10),
        )
        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=80.0)],
            timestamp=now,
        )

        df = store.query_latest("gpu_temp")
        assert len(df) == 1
        assert df["value"][0] == 80.0

    def test_eviction_cleans_internal_indexes(self) -> None:
        store = MiniPrometheus(
            config=MiniPrometheusConfig(
                retention=timedelta(minutes=5),
            )
        )
        now = datetime.now(timezone.utc)

        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="old_metric", labels={}, value=1.0)],
            timestamp=now - timedelta(minutes=10),
        )
        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="new_metric", labels={}, value=2.0)],
            timestamp=now,
        )

        assert "old_metric" not in store._name_index
        assert "new_metric" in store._name_index
        assert len(store._series) == 1


class TestMiniPrometheusIngestSamples:
    def test_ingest_adds_node_id_label(self) -> None:
        store = MiniPrometheus()
        store.ingest_samples(
            target_id="node-42",
            samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=75.0)],
        )

        df = store.query_latest("gpu_temp")
        assert "node_id" in df.columns
        assert df["node_id"][0] == "node-42"

    def test_ingest_preserves_existing_node_id(self) -> None:
        """When a metric already carries a node_id label (e.g. from MegatronAgent),
        ingest_samples must NOT overwrite it with the scrape target_id."""
        store = MiniPrometheus()
        store.ingest_samples(
            target_id="rank-0",
            samples=[
                MetricSample(
                    name="training_iter",
                    labels={"rank": "0", "node_id": "real-node-hostname"},
                    value=100.0,
                )
            ],
        )

        df = store.query_latest("training_iter")
        assert df["node_id"][0] == "real-node-hostname"

    def test_ingest_fills_node_id_when_absent(self) -> None:
        """When a metric has no node_id label, target_id should be used."""
        store = MiniPrometheus()
        store.ingest_samples(
            target_id="scrape-target-0",
            samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=75.0)],
        )

        df = store.query_latest("gpu_temp")
        assert df["node_id"][0] == "scrape-target-0"

    def test_ingest_empty_samples_is_noop(self) -> None:
        store = MiniPrometheus()
        store.ingest_samples(target_id="node-0", samples=[])

        df = store.query_latest("any_metric")
        assert df.is_empty()

    def test_ingest_multiple_targets_separate(self) -> None:
        store = MiniPrometheus()
        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="gpu_available", labels={"gpu": "0"}, value=1.0)],
        )
        store.ingest_samples(
            target_id="node-1",
            samples=[MetricSample(name="gpu_available", labels={"gpu": "0"}, value=0.0)],
        )

        df = store.query_latest("gpu_available").filter(pl.col("value") == 0.0)
        assert len(df) == 1
        assert df["node_id"][0] == "node-1"


class TestMiniPrometheusScrapeTargets:
    def test_remove_scrape_target(self) -> None:
        store = MiniPrometheus()
        store.add_scrape_target(target_id="node-0", address="http://localhost:9090")
        store.add_scrape_target(target_id="node-1", address="http://localhost:9091")

        store.remove_scrape_target("node-0")

        assert "node-0" not in store._scrape_targets
        assert "node-1" in store._scrape_targets

    def test_remove_nonexistent_target_is_noop(self) -> None:
        store = MiniPrometheus()
        store.remove_scrape_target("doesnt-exist")  # should not raise


class TestMiniPrometheusRangeFunctionEdgeCases:
    def test_single_sample_count_over_time(self) -> None:
        store = MiniPrometheus(
            config=MiniPrometheusConfig(
                retention=timedelta(minutes=60),
            )
        )
        now = datetime.now(timezone.utc)

        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="metric", labels={}, value=42.0)],
            timestamp=now,
        )

        df = store.count_over_time("metric", window=timedelta(minutes=5))
        assert len(df) == 1
        assert df["value"][0] == 1.0

    def test_single_sample_changes_returns_zero(self) -> None:
        store = MiniPrometheus(
            config=MiniPrometheusConfig(
                retention=timedelta(minutes=60),
            )
        )
        now = datetime.now(timezone.utc)

        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="metric", labels={}, value=42.0)],
            timestamp=now,
        )

        df = store.changes("metric", window=timedelta(minutes=5))
        assert len(df) == 1
        assert df["value"][0] == 0.0

    def test_changes_with_polars_filter(self) -> None:
        store = MiniPrometheus(
            config=MiniPrometheusConfig(
                retention=timedelta(minutes=60),
            )
        )
        now = datetime.now(timezone.utc)

        for i in range(3):
            store.ingest_samples(
                target_id="node-0",
                samples=[MetricSample(name="iter", labels={"rank": "0"}, value=100.0)],
                timestamp=now - timedelta(minutes=2 - i),
            )

        df = store.changes("iter", window=timedelta(minutes=5))
        filtered = df.filter(pl.col("value") == 0.0)
        assert len(filtered) == 1
        assert filtered["value"][0] == 0.0

    def test_count_over_time_with_polars_filter(self) -> None:
        store = MiniPrometheus(
            config=MiniPrometheusConfig(
                retention=timedelta(minutes=60),
            )
        )
        now = datetime.now(timezone.utc)

        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="alert", labels={}, value=1.0)],
            timestamp=now,
        )

        df = store.count_over_time("alert", window=timedelta(minutes=5))
        df_pass = df.filter(pl.col("value") >= 1.0)
        assert len(df_pass) == 1

        df_fail = df.filter(pl.col("value") >= 10.0)
        assert len(df_fail) == 0
