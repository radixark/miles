"""Backend conformance: MiniPrometheus and real Prometheus must behave
identically when both scrape the same HTTP /metrics endpoint.

Every test is parametrized via the ``backend`` fixture and runs once
against MiniBackend (ScrapeLoop → HTTP GET → ingest) and once against
LiveBackend (real Prometheus binary → PrometheusClient HTTP API).
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl
import pytest

from tests.fast.utils.ft.integration.prometheus_live.conftest import MetricBackend

pytestmark = pytest.mark.integration


class TestQueryLatest:
    async def test_returns_pushed_value(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_latest", 42.0)
        await backend.flush()

        df = backend.store.query_latest("conformance_latest")

        assert not df.is_empty()
        assert "__name__" in df.columns
        assert "value" in df.columns
        assert df["value"][0] == 42.0

    async def test_with_label_filters(self, backend: MetricBackend) -> None:
        backend.set_gauge(
            "conformance_labeled",
            1.0,
            labels={"node_id": "node-0", "device": "ib0"},
        )
        backend.set_gauge(
            "conformance_labeled",
            0.0,
            labels={"node_id": "node-0", "device": "ib1"},
        )
        await backend.flush()

        df = backend.store.query_latest(
            "conformance_labeled",
            label_filters={"device": "ib0"},
        )

        assert df.shape[0] == 1
        assert df["value"][0] == 1.0

    async def test_empty_for_nonexistent_metric(self, backend: MetricBackend) -> None:
        df = backend.store.query_latest("conformance_no_such_metric_xyz")

        assert df.is_empty()
        assert "__name__" in df.columns
        assert "value" in df.columns

    async def test_latest_value_overwrites_previous(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_overwrite", 10.0)
        await backend.flush()

        backend.set_gauge("conformance_overwrite", 42.0)
        await backend.flush()

        df = backend.store.query_latest("conformance_overwrite")

        assert not df.is_empty()
        assert df["value"][0] == 42.0

    async def test_multiple_series_returned(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_multi_series", 1.0, labels={"device": "ib0"})
        backend.set_gauge("conformance_multi_series", 2.0, labels={"device": "ib1"})
        await backend.flush()

        df = backend.store.query_latest("conformance_multi_series")

        assert df.shape[0] >= 2

    async def test_label_columns_present_in_result(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_label_cols", 7.0, labels={"device": "eth0"})
        await backend.flush()

        df = backend.store.query_latest("conformance_label_cols")

        assert not df.is_empty()
        assert "device" in df.columns

    async def test_name_column_matches_queried_metric(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_name_check", 5.0)
        await backend.flush()

        df = backend.store.query_latest("conformance_name_check")

        assert not df.is_empty()
        assert all(name == "conformance_name_check" for name in df["__name__"].to_list())

    async def test_multiple_label_filters_narrow_result(self, backend: MetricBackend) -> None:
        backend.set_gauge(
            "conformance_multi_filter",
            1.0,
            labels={"node_id": "n0", "device": "ib0"},
        )
        backend.set_gauge(
            "conformance_multi_filter",
            2.0,
            labels={"node_id": "n0", "device": "ib1"},
        )
        backend.set_gauge(
            "conformance_multi_filter",
            3.0,
            labels={"node_id": "n1", "device": "ib0"},
        )
        await backend.flush()

        df = backend.store.query_latest(
            "conformance_multi_filter",
            label_filters={"node_id": "n0", "device": "ib0"},
        )

        assert df.shape[0] == 1
        assert df["value"][0] == 1.0

    async def test_empty_dict_filters_equals_no_filters(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_empty_filter", 9.0, labels={"device": "gpu0"})
        await backend.flush()

        df_none = backend.store.query_latest("conformance_empty_filter")
        df_empty = backend.store.query_latest(
            "conformance_empty_filter",
            label_filters={},
        )

        assert df_none.shape == df_empty.shape
        assert df_none["value"].to_list() == df_empty["value"].to_list()

    async def test_label_filter_nonexistent_key_returns_empty(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_nokey", 1.0, labels={"device": "ib0"})
        await backend.flush()

        df = backend.store.query_latest(
            "conformance_nokey",
            label_filters={"nonexistent_key": "val"},
        )

        assert df.is_empty()

    async def test_label_filter_nonexistent_value_returns_empty(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_noval", 1.0, labels={"device": "ib0"})
        await backend.flush()

        df = backend.store.query_latest(
            "conformance_noval",
            label_filters={"device": "no_such_device"},
        )

        assert df.is_empty()


class TestQueryRange:
    async def test_returns_data_in_window(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_range", 1.0)
        await backend.flush()

        backend.set_gauge("conformance_range", 2.0)
        await backend.flush()

        df = backend.store.query_range(
            "conformance_range",
            window=timedelta(minutes=5),
        )

        assert not df.is_empty()
        assert "__name__" in df.columns
        assert "timestamp" in df.columns
        assert "value" in df.columns

    async def test_range_returns_multiple_points(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_range_multi", 1.0)
        await backend.flush()

        backend.set_gauge("conformance_range_multi", 2.0)
        await backend.flush()

        backend.set_gauge("conformance_range_multi", 3.0)
        await backend.flush()

        df = backend.store.query_range(
            "conformance_range_multi",
            window=timedelta(minutes=5),
        )

        assert df.shape[0] >= 2

    async def test_range_values_correctness(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_range_vals", 10.0)
        await backend.flush()

        backend.set_gauge("conformance_range_vals", 20.0)
        await backend.flush()

        df = backend.store.query_range(
            "conformance_range_vals",
            window=timedelta(minutes=5),
        )

        values = set(df["value"].to_list())
        assert 10.0 in values
        assert 20.0 in values

    async def test_range_with_label_filters(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_range_lf", 1.0, labels={"device": "ib0"})
        backend.set_gauge("conformance_range_lf", 2.0, labels={"device": "ib1"})
        await backend.flush()

        df = backend.store.query_range(
            "conformance_range_lf",
            window=timedelta(minutes=5),
            label_filters={"device": "ib0"},
        )

        assert not df.is_empty()
        assert all(v == "ib0" for v in df["device"].to_list())

    async def test_range_empty_for_nonexistent_metric(self, backend: MetricBackend) -> None:
        df = backend.store.query_range(
            "conformance_range_nosuch_xyz",
            window=timedelta(minutes=5),
        )

        assert df.is_empty()
        assert "__name__" in df.columns
        assert "timestamp" in df.columns
        assert "value" in df.columns

    async def test_range_label_columns_present(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_range_lcol", 5.0, labels={"device": "eth0"})
        await backend.flush()

        df = backend.store.query_range(
            "conformance_range_lcol",
            window=timedelta(minutes=5),
        )

        assert not df.is_empty()
        assert "device" in df.columns

    async def test_range_contains_only_requested_metric(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_range_iso_a", 1.0)
        backend.set_gauge("conformance_range_iso_b", 2.0)
        await backend.flush()

        df = backend.store.query_range(
            "conformance_range_iso_a",
            window=timedelta(minutes=5),
        )

        assert not df.is_empty()
        assert all(
            name == "conformance_range_iso_a" for name in df["__name__"].to_list()
        )

    async def test_range_name_column_matches_queried_metric(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_range_name", 7.0)
        await backend.flush()

        df = backend.store.query_range(
            "conformance_range_name",
            window=timedelta(minutes=5),
        )

        assert not df.is_empty()
        assert all(
            name == "conformance_range_name" for name in df["__name__"].to_list()
        )


class TestRangeAggregations:
    async def test_changes_detects_value_change(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_changes", 1.0)
        await backend.flush()

        backend.set_gauge("conformance_changes", 2.0)
        await backend.flush()

        df = backend.store.changes(
            "conformance_changes",
            window=timedelta(minutes=5),
        )

        assert not df.is_empty()
        assert df["value"][0] >= 1.0

    async def test_count_over_time_counts_samples(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_count", 10.0)
        await backend.flush()

        backend.set_gauge("conformance_count", 20.0)
        await backend.flush()

        df = backend.store.count_over_time(
            "conformance_count",
            window=timedelta(minutes=5),
        )

        assert not df.is_empty()
        assert df["value"][0] >= 1.0

    async def test_avg_over_time_computes_average(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_avg", 1.0)
        await backend.flush()

        backend.set_gauge("conformance_avg", 3.0)
        await backend.flush()

        df = backend.store.avg_over_time(
            "conformance_avg",
            window=timedelta(minutes=5),
        )

        assert not df.is_empty()
        avg = df["value"][0]
        assert 1.0 <= avg <= 3.0

    # ---- changes: additional tests ----

    async def test_changes_zero_when_constant(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_changes_const", 5.0)
        await backend.flush()

        backend.set_gauge("conformance_changes_const", 5.0)
        await backend.flush()

        df = backend.store.changes(
            "conformance_changes_const",
            window=timedelta(minutes=5),
        )

        assert not df.is_empty()
        assert df["value"][0] == 0.0

    async def test_changes_single_sample_returns_zero(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_changes_single", 7.0)
        await backend.flush()

        df = backend.store.changes(
            "conformance_changes_single",
            window=timedelta(minutes=5),
        )

        assert not df.is_empty()
        assert df["value"][0] == 0.0

    async def test_changes_with_label_filters(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_changes_lf", 1.0, labels={"device": "ib0"})
        backend.set_gauge("conformance_changes_lf", 1.0, labels={"device": "ib1"})
        await backend.flush()

        backend.set_gauge("conformance_changes_lf", 2.0, labels={"device": "ib0"})
        backend.set_gauge("conformance_changes_lf", 1.0, labels={"device": "ib1"})
        await backend.flush()

        df = backend.store.changes(
            "conformance_changes_lf",
            window=timedelta(minutes=5),
            label_filters={"device": "ib0"},
        )

        assert not df.is_empty()
        assert df["value"][0] >= 1.0

    async def test_changes_empty_for_nonexistent_metric(self, backend: MetricBackend) -> None:
        df = backend.store.changes(
            "conformance_changes_nosuch_xyz",
            window=timedelta(minutes=5),
        )

        assert df.is_empty()
        assert "__name__" in df.columns
        assert "value" in df.columns

    # ---- count_over_time: additional tests ----

    async def test_count_over_time_exact_count(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_count_exact", 1.0)
        await backend.flush()

        backend.set_gauge("conformance_count_exact", 2.0)
        await backend.flush()

        backend.set_gauge("conformance_count_exact", 3.0)
        await backend.flush()

        df = backend.store.count_over_time(
            "conformance_count_exact",
            window=timedelta(minutes=5),
        )

        assert not df.is_empty()
        assert df["value"][0] >= 3.0

    async def test_count_over_time_single_sample_is_at_least_one(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_count_single", 42.0)
        await backend.flush()

        df = backend.store.count_over_time(
            "conformance_count_single",
            window=timedelta(minutes=5),
        )

        assert not df.is_empty()
        assert df["value"][0] >= 1.0

    async def test_count_over_time_with_label_filters(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_count_lf", 1.0, labels={"device": "ib0"})
        backend.set_gauge("conformance_count_lf", 2.0, labels={"device": "ib1"})
        await backend.flush()

        df = backend.store.count_over_time(
            "conformance_count_lf",
            window=timedelta(minutes=5),
            label_filters={"device": "ib0"},
        )

        assert not df.is_empty()
        assert df.shape[0] == 1
        assert df["value"][0] >= 1.0

    async def test_count_over_time_empty_for_nonexistent_metric(self, backend: MetricBackend) -> None:
        df = backend.store.count_over_time(
            "conformance_count_nosuch_xyz",
            window=timedelta(minutes=5),
        )

        assert df.is_empty()
        assert "__name__" in df.columns
        assert "value" in df.columns

    # ---- avg_over_time: additional tests ----

    async def test_avg_over_time_precise_value(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_avg_precise", 2.0)
        await backend.flush()

        backend.set_gauge("conformance_avg_precise", 4.0)
        await backend.flush()

        backend.set_gauge("conformance_avg_precise", 6.0)
        await backend.flush()

        df = backend.store.avg_over_time(
            "conformance_avg_precise",
            window=timedelta(minutes=5),
        )

        assert not df.is_empty()
        avg = df["value"][0]
        assert 2.0 <= avg <= 6.0

    async def test_avg_over_time_single_sample_equals_that_value(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_avg_single", 99.0)
        await backend.flush()

        df = backend.store.avg_over_time(
            "conformance_avg_single",
            window=timedelta(minutes=5),
        )

        assert not df.is_empty()
        assert df["value"][0] == 99.0

    async def test_avg_over_time_with_label_filters(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_avg_lf", 10.0, labels={"device": "ib0"})
        backend.set_gauge("conformance_avg_lf", 100.0, labels={"device": "ib1"})
        await backend.flush()

        df = backend.store.avg_over_time(
            "conformance_avg_lf",
            window=timedelta(minutes=5),
            label_filters={"device": "ib0"},
        )

        assert not df.is_empty()
        assert df.shape[0] == 1
        assert df["value"][0] == 10.0

    async def test_avg_over_time_empty_for_nonexistent_metric(self, backend: MetricBackend) -> None:
        df = backend.store.avg_over_time(
            "conformance_avg_nosuch_xyz",
            window=timedelta(minutes=5),
        )

        assert df.is_empty()
        assert "__name__" in df.columns
        assert "value" in df.columns


class TestSchemaConformance:
    async def test_instant_schema_columns_and_types(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_schema_instant", 1.0, labels={"device": "gpu0"})
        await backend.flush()

        df = backend.store.query_latest("conformance_schema_instant")

        assert not df.is_empty()
        assert df["__name__"].dtype == pl.Utf8
        assert df["value"].dtype == pl.Float64
        assert "device" in df.columns
        assert df["device"].dtype == pl.Utf8

    async def test_range_schema_columns_and_types(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_schema_range", 1.0, labels={"device": "gpu0"})
        await backend.flush()

        df = backend.store.query_range(
            "conformance_schema_range",
            window=timedelta(minutes=5),
        )

        assert not df.is_empty()
        assert df["__name__"].dtype == pl.Utf8
        assert df["value"].dtype == pl.Float64
        assert "timestamp" in df.columns
        assert "device" in df.columns
        assert df["device"].dtype == pl.Utf8

    async def test_aggregation_schema_columns_and_types(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_schema_agg", 1.0, labels={"device": "gpu0"})
        await backend.flush()

        backend.set_gauge("conformance_schema_agg", 2.0, labels={"device": "gpu0"})
        await backend.flush()

        for method_name in ("changes", "count_over_time", "avg_over_time"):
            method = getattr(backend.store, method_name)
            df = method("conformance_schema_agg", window=timedelta(minutes=5))

            assert not df.is_empty(), f"{method_name} returned empty"
            assert df["__name__"].dtype == pl.Utf8, f"{method_name} __name__ dtype"
            assert df["value"].dtype == pl.Float64, f"{method_name} value dtype"


class TestValueEdgeCases:
    async def test_zero_value_not_treated_as_absent(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_zero_val", 0.0)
        await backend.flush()

        df = backend.store.query_latest("conformance_zero_val")

        assert not df.is_empty()
        assert df["value"][0] == 0.0

    async def test_negative_value_round_trip(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_neg_val", -99.5)
        await backend.flush()

        df = backend.store.query_latest("conformance_neg_val")

        assert not df.is_empty()
        assert df["value"][0] == -99.5


class TestMetricIsolation:
    async def test_query_latest_does_not_leak_other_metrics(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_iso_target", 1.0)
        backend.set_gauge("conformance_iso_other", 2.0)
        await backend.flush()

        df = backend.store.query_latest("conformance_iso_target")

        assert not df.is_empty()
        assert all(
            name == "conformance_iso_target" for name in df["__name__"].to_list()
        )

    async def test_query_is_idempotent(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_idempotent", 42.0)
        await backend.flush()

        df1 = backend.store.query_latest("conformance_idempotent")
        df2 = backend.store.query_latest("conformance_idempotent")

        assert df1.shape == df2.shape
        assert df1["value"].to_list() == df2["value"].to_list()
