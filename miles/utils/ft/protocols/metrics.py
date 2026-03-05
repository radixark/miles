from __future__ import annotations

from datetime import datetime, timedelta
from typing import NamedTuple, Protocol, runtime_checkable

import polars as pl


class StepValue(NamedTuple):
    step: int
    value: float


class TimedStepValue(NamedTuple):
    step: int
    timestamp: datetime
    value: float


@runtime_checkable
class MetricQueryProtocol(Protocol):
    def query_latest(
        self,
        metric_name: str,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    def query_range(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    def changes(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    def count_over_time(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    def avg_over_time(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    def min_over_time(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    def max_over_time(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...


@runtime_checkable
class MetricStoreLifecycle(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...


@runtime_checkable
class MetricStoreProtocol(MetricQueryProtocol, MetricStoreLifecycle, Protocol):
    ...


class ScrapeTargetManagerProtocol(Protocol):
    def add_scrape_target(self, target_id: str, address: str) -> None: ...

    def remove_scrape_target(self, target_id: str) -> None: ...


class TrainingMetricStoreProtocol(Protocol):
    def latest(self, metric_name: str) -> float | None: ...

    def query_last_n_steps(
        self, metric_name: str, last_n: int,
    ) -> list[StepValue]: ...

    def query_time_window(
        self, metric_name: str, window: timedelta,
    ) -> list[TimedStepValue]: ...
