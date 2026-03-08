"""Tests for MetricStoreProtocol split into query and lifecycle protocols."""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from miles.utils.ft.controller.detectors.base import DetectorContext
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.metrics.prometheus_api.store import PrometheusClient
from miles.utils.ft.controller.types import (
    MetricQueryProtocol,
    MetricStoreLifecycle,
    MetricStoreProtocol,
    ScrapeTargetManagerProtocol,
    TrainingMetricStoreProtocol,
)
from miles.utils.ft.adapters.types import JobStatus


class TestMiniPrometheusProtocolCompliance:
    def test_satisfies_metric_query_protocol(self) -> None:
        store = MiniPrometheus()
        assert isinstance(store, MetricQueryProtocol)

    def test_satisfies_metric_store_lifecycle(self) -> None:
        store = MiniPrometheus()
        assert isinstance(store, MetricStoreLifecycle)

    def test_satisfies_metric_store_protocol(self) -> None:
        store = MiniPrometheus()
        assert isinstance(store, MetricStoreProtocol)

    def test_satisfies_scrape_target_manager_protocol(self) -> None:
        store = MiniPrometheus()
        assert isinstance(store, ScrapeTargetManagerProtocol)


class TestPrometheusClientProtocolCompliance:
    def test_satisfies_metric_query_protocol(self) -> None:
        client = PrometheusClient(url="http://localhost:9090")
        assert isinstance(client, MetricQueryProtocol)

    def test_satisfies_metric_store_lifecycle(self) -> None:
        client = PrometheusClient(url="http://localhost:9090")
        assert isinstance(client, MetricStoreLifecycle)

    def test_satisfies_metric_store_protocol(self) -> None:
        client = PrometheusClient(url="http://localhost:9090")
        assert isinstance(client, MetricStoreProtocol)


class TestMiniWandbProtocolCompliance:
    def test_satisfies_training_metric_store_protocol(self) -> None:
        store = MiniWandb()
        assert isinstance(store, TrainingMetricStoreProtocol)


class _QueryOnlyStore:
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


class _FakeMiniWandb:
    def latest(self, metric_name: str, rank: int | None = None) -> float | None:
        return None

    def query_last_n_steps(self, metric_name: str, last_n: int, rank: int | None = None) -> list:
        return []

    def query_time_window(self, metric_name: str, window: timedelta, rank: int | None = None) -> list:
        return []


class TestQueryOnlyProtocol:
    def test_satisfies_query_protocol(self) -> None:
        store = _QueryOnlyStore()
        assert isinstance(store, MetricQueryProtocol)

    def test_does_not_satisfy_lifecycle_protocol(self) -> None:
        store = _QueryOnlyStore()
        assert not isinstance(store, MetricStoreLifecycle)

    def test_does_not_satisfy_full_protocol(self) -> None:
        store = _QueryOnlyStore()
        assert not isinstance(store, MetricStoreProtocol)

    def test_query_only_store_usable_as_detector_context_metric_store(self) -> None:
        store = _QueryOnlyStore()
        ctx = DetectorContext(
            metric_store=store,
            mini_wandb=_FakeMiniWandb(),
            rank_placement={0: "node-0"},
            job_status=JobStatus.RUNNING,
        )
        assert ctx.metric_store is store
