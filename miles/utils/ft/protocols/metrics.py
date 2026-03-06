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
    ) -> pl.DataFrame | None: ...

    def query_range(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame | None: ...

    def changes(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame | None: ...

    def count_over_time(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame | None: ...

    def avg_over_time(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame | None: ...



@runtime_checkable
class MetricStoreLifecycle(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...


@runtime_checkable
class MetricStoreProtocol(MetricQueryProtocol, MetricStoreLifecycle, Protocol):
    ...


@runtime_checkable
class ScrapeTargetManagerProtocol(Protocol):
    def add_scrape_target(self, target_id: str, address: str) -> None: ...

    def remove_scrape_target(self, target_id: str) -> None: ...


@runtime_checkable
class TrainingMetricStoreProtocol(Protocol):
    def latest(self, metric_name: str) -> float | None: ...

    def query_last_n_steps(
        self, metric_name: str, last_n: int,
    ) -> list[StepValue]: ...

    def query_time_window(
        self, metric_name: str, window: timedelta,
    ) -> list[TimedStepValue]: ...
