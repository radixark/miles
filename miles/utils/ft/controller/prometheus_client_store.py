from __future__ import annotations

import logging
import time
from datetime import timedelta
from typing import Any

import httpx
import polars as pl

from miles.utils.ft.controller.metric_store_mixin import RangeAggregationMixin
from miles.utils.ft.controller.mini_prometheus.query import EMPTY_INSTANT, EMPTY_RANGE

logger = logging.getLogger(__name__)


class PrometheusClient(RangeAggregationMixin):
    """MetricStoreProtocol implementation backed by a real Prometheus HTTP API.

    Each typed method builds the corresponding PromQL query internally,
    sends it to Prometheus, and parses the JSON response into a Polars DataFrame
    with the same schema as MiniPrometheus.
    """

    def __init__(self, url: str, timeout: float = 10.0) -> None:
        self._url = url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    # -------------------------------------------------------------------
    # MetricStoreProtocol implementation
    # -------------------------------------------------------------------

    def query_latest(
        self, metric_name: str, label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        promql = _build_selector(metric_name, label_filters)
        return self._instant_query_raw(promql)

    def query_range(
        self, metric_name: str, window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        promql = _build_selector(metric_name, label_filters)
        return self._range_query_raw(promql, window)

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        self._client.close()

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
        if data is None:
            return EMPTY_INSTANT
        return _parse_instant_response(data)

    def _range_query_raw(self, promql: str, window: timedelta) -> pl.DataFrame:
        now = time.time()
        start = now - window.total_seconds()

        data = self._fetch_json(
            "/api/v1/query_range",
            params={"query": promql, "start": start, "end": now, "step": 15},
        )
        if data is None:
            return EMPTY_RANGE
        return _parse_range_response(data)

    def _fetch_json(self, path: str, params: dict[str, object]) -> dict[str, Any] | None:
        try:
            response = self._client.get(f"{self._url}{path}", params=params)
            response.raise_for_status()
            return response.json()
        except Exception:
            logger.warning("prometheus_query_failed path=%s params=%s", path, params, exc_info=True)
            return None


# ---------------------------------------------------------------------------
# PromQL construction helpers
# ---------------------------------------------------------------------------


def _build_selector(metric_name: str, label_filters: dict[str, str] | None) -> str:
    if not label_filters:
        return metric_name
    labels_str = ", ".join(f'{k}="{v}"' for k, v in sorted(label_filters.items()))
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


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------



def _parse_instant_response(data: dict[str, Any]) -> pl.DataFrame:
    extracted = _extract_results(data)
    if extracted is None:
        return EMPTY_INSTANT

    result, result_type = extracted
    if result_type == "vector":
        return _parse_vector(result)
    if result_type == "scalar":
        return _parse_scalar(result)

    logger.warning("prometheus_unsupported_result_type type=%s", result_type)
    return EMPTY_INSTANT


def _parse_range_response(data: dict[str, Any]) -> pl.DataFrame:
    extracted = _extract_results(data)
    if extracted is None:
        return EMPTY_RANGE

    result, result_type = extracted
    if result_type != "matrix":
        logger.warning("prometheus_unsupported_range_result_type type=%s", result_type)
        return EMPTY_RANGE

    return _parse_matrix(result)


def _extract_results(data: dict[str, Any]) -> tuple[list[dict[str, Any]], str] | None:
    if data.get("status") != "success":
        logger.warning("prometheus_query_error response=%s", data)
        return None

    data_section = data.get("data") or {}
    result = data_section.get("result") or []
    if not result:
        return None

    return result, data_section.get("resultType", "")


def _parse_vector(result: list[dict[str, Any]]) -> pl.DataFrame:
    sorted_label_keys = _collect_sorted_label_keys(result)
    records: list[dict[str, object]] = []

    for item in result:
        metric: dict[str, str] = item.get("metric") or {}
        value_pair = item.get("value") or [0, "0"]
        try:
            parsed_value = float(value_pair[1])
        except (IndexError, TypeError, ValueError):
            continue

        record = _build_label_record(metric, sorted_label_keys)
        record["value"] = parsed_value
        records.append(record)

    if not records:
        return EMPTY_INSTANT

    return pl.DataFrame(records)


def _parse_scalar(result: list[Any]) -> pl.DataFrame:
    if len(result) != 2:
        return EMPTY_INSTANT

    try:
        return pl.DataFrame({"__name__": [""], "value": [float(result[1])]})
    except (TypeError, ValueError):
        return EMPTY_INSTANT


def _parse_matrix(result: list[dict[str, Any]]) -> pl.DataFrame:
    sorted_label_keys = _collect_sorted_label_keys(result)
    records: list[dict[str, object]] = []

    for item in result:
        metric: dict[str, str] = item.get("metric") or {}
        for ts, value_str in (item.get("values") or []):
            try:
                parsed_value = float(value_str)
                parsed_ts = float(ts)
            except (TypeError, ValueError):
                continue

            record = _build_label_record(metric, sorted_label_keys)
            record["timestamp"] = parsed_ts
            record["value"] = parsed_value
            records.append(record)

    if not records:
        return EMPTY_RANGE

    return pl.DataFrame(records)


def _collect_sorted_label_keys(result: list[dict[str, Any]]) -> list[str]:
    all_keys: set[str] = set()
    for item in result:
        all_keys.update((item.get("metric") or {}).keys())
    return sorted(k for k in all_keys if k != "__name__")


def _build_label_record(
    metric: dict[str, str],
    sorted_label_keys: list[str],
) -> dict[str, object]:
    record: dict[str, object] = {"__name__": metric.get("__name__", "")}
    for key in sorted_label_keys:
        record[key] = metric.get(key, "")
    return record
