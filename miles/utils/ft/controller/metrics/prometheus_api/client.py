from __future__ import annotations

import asyncio
import logging
import time
from datetime import timedelta
from typing import Any

import httpx
import polars as pl

from miles.utils.ft.controller.metrics.aggregation_mixin import RangeAggregationMixin
from miles.utils.ft.controller.metrics.mini_prometheus.query import AmbiguousSeriesError
from miles.utils.ft.controller.metrics.prometheus_api.errors import PrometheusQueryError
from miles.utils.ft.controller.metrics.prometheus_api.response_parser import (
    parse_instant_response,
    parse_range_response,
)
from miles.utils.ft.controller.types import TimeSeriesStoreProtocol
from miles.utils.ft.utils.retry import retry_sync

logger = logging.getLogger(__name__)

_DEFAULT_RANGE_QUERY_STEP_SECONDS: int = 15
_FETCH_MAX_RETRIES: int = 2
_FETCH_RETRY_DELAY_SECONDS: float = 0.5
_DEFAULT_HEALTH_PROBE_INTERVAL_SECONDS: float = 15.0
_DEFAULT_HEALTH_PROBE_FAILURE_THRESHOLD: int = 3


class PrometheusClient(RangeAggregationMixin, TimeSeriesStoreProtocol):
    """TimeSeriesStoreProtocol implementation backed by a real Prometheus HTTP API.

    Each typed method builds the corresponding PromQL query internally,
    sends it to Prometheus, and parses the JSON response into a Polars DataFrame
    with the same schema as MiniPrometheus.
    """

    def __init__(
        self,
        url: str,
        timeout: float = 10.0,
        range_query_step_seconds: int = _DEFAULT_RANGE_QUERY_STEP_SECONDS,
        health_probe_interval_seconds: float = _DEFAULT_HEALTH_PROBE_INTERVAL_SECONDS,
        health_probe_failure_threshold: int = _DEFAULT_HEALTH_PROBE_FAILURE_THRESHOLD,
    ) -> None:
        self._url = url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)
        self._range_query_step_seconds = range_query_step_seconds
        self._stop_event: asyncio.Event | None = None
        self._health_probe_interval = health_probe_interval_seconds
        self._health_probe_failure_threshold = health_probe_failure_threshold

    # -------------------------------------------------------------------
    # TimeSeriesStoreProtocol implementation (sync)
    # -------------------------------------------------------------------

    def query_single_latest(
        self,
        metric_name: str,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        df = self.query_latest(metric_name=metric_name, label_filters=label_filters)
        if len(df) > 1:
            logger.error(
                "prom_api: query_single_latest ambiguous metric=%s, filters=%s, matched=%d",
                metric_name,
                label_filters,
                len(df),
            )
            raise AmbiguousSeriesError(
                metric_name=metric_name,
                label_filters=label_filters,
                matched_count=len(df),
            )
        return df

    def query_latest(
        self,
        metric_name: str,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        promql = _build_selector(metric_name, label_filters)
        return self._instant_query_raw(promql)

    def query_range(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        promql = _build_selector(metric_name, label_filters)
        # TODO: This returns Prometheus query_range output sampled at a fixed
        # evaluation step, not the raw scrape samples returned by
        # MiniPrometheus query_range(). Some current detectors still assume
        # raw-sample semantics here (edge counting / span inference), and we
        # are intentionally not reconciling that backend mismatch for now.
        return self._range_query_raw(promql, window)

    async def start(self) -> None:
        logger.info(
            "prom_api: starting health probe loop url=%s, interval=%.1fs", self._url, self._health_probe_interval
        )
        self._stop_event = asyncio.Event()
        consecutive_failures = 0

        while not self._stop_event.is_set():
            try:
                await asyncio.to_thread(self._health_probe)
                if consecutive_failures > 0:
                    logger.info("prom_api: health probe recovered after %d failures", consecutive_failures)
                consecutive_failures = 0
            except Exception:
                consecutive_failures += 1
                logger.warning(
                    "prom_api: health probe failed consecutive=%d",
                    consecutive_failures,
                    exc_info=True,
                )
                if consecutive_failures >= self._health_probe_failure_threshold:
                    logger.error(
                        "prom_api: health probe threshold reached, consecutive=%d >= threshold=%d",
                        consecutive_failures,
                        self._health_probe_failure_threshold,
                    )
                    raise

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._health_probe_interval,
                )
                break
            except asyncio.TimeoutError:
                pass

    def _health_probe(self) -> None:
        response = self._client.get(f"{self._url}/-/ready")
        response.raise_for_status()

    async def stop(self) -> None:
        logger.info("prom_api: stopping client")
        if self._stop_event is not None:
            self._stop_event.set()
        self._client.close()

    # -------------------------------------------------------------------
    # Async query helpers (non-blocking for callers on the event loop)
    # -------------------------------------------------------------------

    async def aquery_latest(
        self,
        metric_name: str,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        promql = _build_selector(metric_name, label_filters)
        data = await self._afetch_json("/api/v1/query", params={"query": promql})
        return parse_instant_response(data)

    async def aquery_range(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        promql = _build_selector(metric_name, label_filters)
        now = time.time()
        start = now - window.total_seconds()
        data = await self._afetch_json(
            "/api/v1/query_range",
            params={"query": promql, "start": start, "end": now, "step": self._range_query_step_seconds},
        )
        return parse_range_response(data)

    # -------------------------------------------------------------------
    # Internal: query execution
    # -------------------------------------------------------------------

    def _dispatch_range_function(
        self,
        func_name: str,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None,
    ) -> pl.DataFrame:
        selector = _build_selector(metric_name, label_filters)
        promql = f"{func_name}({selector}[{_format_duration(window)}])"
        return self._instant_query_raw(promql)

    def _instant_query_raw(self, promql: str) -> pl.DataFrame:
        data = self._fetch_json("/api/v1/query", params={"query": promql})
        return parse_instant_response(data)

    def _range_query_raw(self, promql: str, window: timedelta) -> pl.DataFrame:
        now = time.time()
        start = now - window.total_seconds()

        data = self._fetch_json(
            "/api/v1/query_range",
            params={"query": promql, "start": start, "end": now, "step": self._range_query_step_seconds},
        )
        return parse_range_response(data)

    def _fetch_json(self, path: str, params: dict[str, object]) -> dict[str, Any]:
        def _do_fetch() -> dict[str, Any]:
            response = self._client.get(f"{self._url}{path}", params=params)
            response.raise_for_status()
            return response.json()

        result = retry_sync(
            func=_do_fetch,
            description=f"prometheus_fetch({path})",
            max_retries=_FETCH_MAX_RETRIES,
            backoff_base=_FETCH_RETRY_DELAY_SECONDS,
            max_backoff=_FETCH_RETRY_DELAY_SECONDS,
        )
        if not result.ok:
            logger.error("prom_api: query failed after %d retries path=%s", _FETCH_MAX_RETRIES, path)
            raise PrometheusQueryError(
                f"Prometheus query failed after {_FETCH_MAX_RETRIES} retries: {path}"
            ) from result.exception
        return result.value  # type: ignore[return-value]

    async def _afetch_json(self, path: str, params: dict[str, object]) -> dict[str, Any]:
        return await asyncio.to_thread(self._fetch_json, path, params)


# ---------------------------------------------------------------------------
# PromQL construction helpers
# ---------------------------------------------------------------------------


def _escape_promql_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _build_selector(metric_name: str, label_filters: dict[str, str] | None) -> str:
    if not label_filters:
        return metric_name
    labels_str = ", ".join(f'{k}="{_escape_promql_label_value(v)}"' for k, v in sorted(label_filters.items()))
    return f"{metric_name}{{{labels_str}}}"


def _format_duration(td: timedelta) -> str:
    total_seconds = int(td.total_seconds())
    if total_seconds >= 86400 and total_seconds % 86400 == 0:
        return f"{total_seconds // 86400}d"
    if total_seconds >= 3600 and total_seconds % 3600 == 0:
        return f"{total_seconds // 3600}h"
    if total_seconds >= 60 and total_seconds % 60 == 0:
        return f"{total_seconds // 60}m"
    return f"{total_seconds}s"
