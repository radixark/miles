from __future__ import annotations

from datetime import timedelta
from typing import Protocol

import polars as pl


class MetricStoreProtocol(Protocol):
    def query_latest(
        self, metric_name: str, label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    def query_range(
        self, metric_name: str, window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    def changes(
        self, metric_name: str, window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    def count_over_time(
        self, metric_name: str, window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    def avg_over_time(
        self, metric_name: str, window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    def min_over_time(
        self, metric_name: str, window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    def max_over_time(
        self, metric_name: str, window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...


class ScrapeTargetManagerProtocol(Protocol):
    def add_scrape_target(self, target_id: str, address: str) -> None: ...

    def remove_scrape_target(self, target_id: str) -> None: ...
