"""Tests for MetricStoreProtocol split into query and lifecycle protocols."""

from __future__ import annotations

from datetime import timedelta

import polars as pl
import pytest

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.detectors.base import DetectorContext
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.metrics.prometheus_api.client import PrometheusClient
from miles.utils.ft.controller.types import (
    TimeSeriesQueryProtocol,
    TimeSeriesStoreLifecycle,
    TimeSeriesStoreProtocol,
    ScrapeTargetManagerProtocol,
    TrainingMetricStoreProtocol,
)


class TestMiniPrometheusProtocolCompliance:
    def test_satisfies_metric_query_protocol(self) -> None:
        store = MiniPrometheus()
        assert isinstance(store, TimeSeriesQueryProtocol)

    def test_satisfies_metric_store_lifecycle(self) -> None:
        store = MiniPrometheus()
        assert isinstance(store, TimeSeriesStoreLifecycle)

    def test_satisfies_metric_store_protocol(self) -> None:
        store = MiniPrometheus()
        assert isinstance(store, TimeSeriesStoreProtocol)

    def test_satisfies_scrape_target_manager_protocol(self) -> None:
        store = MiniPrometheus()
        assert isinstance(store, ScrapeTargetManagerProtocol)


class TestPrometheusClientProtocolCompliance:
    def test_satisfies_metric_query_protocol(self) -> None:
        client = PrometheusClient(url="http://localhost:9090")
        assert isinstance(client, TimeSeriesQueryProtocol)

    def test_satisfies_metric_store_lifecycle(self) -> None:
        client = PrometheusClient(url="http://localhost:9090")
        assert isinstance(client, TimeSeriesStoreLifecycle)

    def test_satisfies_metric_store_protocol(self) -> None:
        client = PrometheusClient(url="http://localhost:9090")
        assert isinstance(client, TimeSeriesStoreProtocol)


class TestMiniWandbProtocolCompliance:
    def test_satisfies_training_metric_store_protocol(self) -> None:
        store = MiniWandb()
        assert isinstance(store, TrainingMetricStoreProtocol)


class _QueryOnlyStore(TimeSeriesQueryProtocol):
    """Minimal implementation satisfying only MetricQueryProtocol."""

    def query_latest(
        self,
        metric_name: str,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        return pl.DataFrame()

    def query_range(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        return pl.DataFrame()

    def changes(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        return pl.DataFrame()

    def count_over_time(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        return pl.DataFrame()

    def avg_over_time(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        return pl.DataFrame()


class _FakeMiniWandb(TrainingMetricStoreProtocol):
    def latest(self, metric_name: str, rank: int | None = None) -> float | None:
        return None

    def query_last_n_steps(self, metric_name: str, last_n: int, rank: int | None = None) -> list:
        return []

    def query_time_window(self, metric_name: str, window: timedelta, rank: int | None = None) -> list:
        return []


class TestQueryOnlyProtocol:
    def test_satisfies_query_protocol(self) -> None:
        store = _QueryOnlyStore()
        assert isinstance(store, TimeSeriesQueryProtocol)

    def test_does_not_satisfy_lifecycle_protocol(self) -> None:
        store = _QueryOnlyStore()
        assert not isinstance(store, TimeSeriesStoreLifecycle)

    def test_does_not_satisfy_full_protocol(self) -> None:
        store = _QueryOnlyStore()
        assert not isinstance(store, TimeSeriesStoreProtocol)

    def test_query_only_store_usable_as_detector_context_metric_store(self) -> None:
        store = _QueryOnlyStore()
        ctx = DetectorContext(
            metric_store=store,
            mini_wandb=_FakeMiniWandb(),
            active_node_ids={"node-0"},
            job_status=JobStatus.RUNNING,
        )
        assert ctx.metric_store is store


class TestIncompleteSubclassRaisesTypeError:
    def test_incomplete_metric_query(self) -> None:
        class _Incomplete(TimeSeriesQueryProtocol):
            pass

        with pytest.raises(TypeError):
            _Incomplete()

    def test_incomplete_metric_store_lifecycle(self) -> None:
        class _Incomplete(TimeSeriesStoreLifecycle):
            pass

        with pytest.raises(TypeError):
            _Incomplete()

    def test_incomplete_training_metric_store(self) -> None:
        class _Incomplete(TrainingMetricStoreProtocol):
            pass

        with pytest.raises(TypeError):
            _Incomplete()
