"""Mixin implementing the five range aggregation methods of TimeSeriesStoreProtocol.

Subclasses must implement _dispatch_range_function().
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import timedelta

import polars as pl


class RangeAggregationMixin:
    def changes(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        return self._dispatch_range_function("changes", metric_name, window, label_filters)

    def count_over_time(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        return self._dispatch_range_function("count_over_time", metric_name, window, label_filters)

    def avg_over_time(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        return self._dispatch_range_function("avg_over_time", metric_name, window, label_filters)

    @abstractmethod
    def _dispatch_range_function(
        self,
        func_name: str,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None,
    ) -> pl.DataFrame: ...
