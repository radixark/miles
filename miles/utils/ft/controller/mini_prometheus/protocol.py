from __future__ import annotations

from datetime import datetime, timedelta
from typing import Protocol

import polars as pl


class MetricStoreProtocol(Protocol):
    def instant_query(self, query: str) -> pl.DataFrame: ...

    def range_query(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: timedelta,
    ) -> pl.DataFrame: ...


class ScrapeTargetManagerProtocol(Protocol):
    def add_scrape_target(self, target_id: str, address: str) -> None: ...

    def remove_scrape_target(self, target_id: str) -> None: ...
