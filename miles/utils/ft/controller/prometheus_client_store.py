from __future__ import annotations

import logging
from datetime import datetime, timedelta

import httpx
import polars as pl

logger = logging.getLogger(__name__)


class PrometheusClient:
    """MetricStoreProtocol implementation backed by a real Prometheus HTTP API.

    Queries are forwarded to Prometheus and results parsed into Polars DataFrames
    with the same schema as MiniPrometheus, so detectors work without modification.
    """

    def __init__(self, url: str, timeout: float = 10.0) -> None:
        self._url = url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def instant_query(self, query: str) -> pl.DataFrame:
        try:
            response = self._client.get(
                f"{self._url}/api/v1/query",
                params={"query": query},
            )
            response.raise_for_status()
            data = response.json()
        except Exception:
            logger.warning("prometheus_instant_query_failed query=%s", query, exc_info=True)
            return _empty_instant_dataframe()

        return _parse_instant_response(data)

    def range_query(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: timedelta,
    ) -> pl.DataFrame:
        try:
            response = self._client.get(
                f"{self._url}/api/v1/query_range",
                params={
                    "query": query,
                    "start": start.timestamp(),
                    "end": end.timestamp(),
                    "step": step.total_seconds(),
                },
            )
            response.raise_for_status()
            data = response.json()
        except Exception:
            logger.warning("prometheus_range_query_failed query=%s", query, exc_info=True)
            return _empty_range_dataframe()

        return _parse_range_response(data)

    def close(self) -> None:
        self._client.close()


def _empty_instant_dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {"__name__": pl.Series([], dtype=pl.Utf8), "value": pl.Series([], dtype=pl.Float64)}
    )


def _empty_range_dataframe() -> pl.DataFrame:
    return pl.DataFrame({
        "__name__": pl.Series([], dtype=pl.Utf8),
        "timestamp": pl.Series([], dtype=pl.Float64),
        "value": pl.Series([], dtype=pl.Float64),
    })


def _parse_instant_response(data: dict) -> pl.DataFrame:
    if data.get("status") != "success":
        logger.warning("prometheus_query_error response=%s", data)
        return _empty_instant_dataframe()

    data_section = data.get("data", {})
    result = data_section.get("result", [])
    if not result:
        return _empty_instant_dataframe()

    result_type = data_section.get("resultType", "")
    if result_type == "vector":
        return _parse_vector(result)
    if result_type == "scalar":
        return _parse_scalar(result)

    logger.warning("prometheus_unsupported_result_type type=%s", result_type)
    return _empty_instant_dataframe()


def _parse_vector(result: list[dict]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    all_label_keys: set[str] = set()

    for item in result:
        metric = item.get("metric", {})
        _, value_str = item.get("value", [0, "0"])
        all_label_keys.update(metric.keys())
        rows.append({
            "_metric": metric,
            "_value": float(value_str),
        })

    records: list[dict[str, object]] = []
    for row in rows:
        metric: dict = row["_metric"]  # type: ignore[assignment]
        record: dict[str, object] = {"__name__": metric.get("__name__", "")}
        for key in sorted(all_label_keys):
            if key != "__name__":
                record[key] = metric.get(key, "")
        record["value"] = row["_value"]
        records.append(record)

    return pl.DataFrame(records)


def _parse_scalar(result: list) -> pl.DataFrame:
    _, value_str = result
    return pl.DataFrame({"__name__": [""], "value": [float(value_str)]})


def _parse_range_response(data: dict) -> pl.DataFrame:
    if data.get("status") != "success":
        logger.warning("prometheus_query_error response=%s", data)
        return _empty_range_dataframe()

    data_section = data.get("data", {})
    result = data_section.get("result", [])
    if not result:
        return _empty_range_dataframe()

    result_type = data_section.get("resultType", "")
    if result_type != "matrix":
        logger.warning("prometheus_unsupported_range_result_type type=%s", result_type)
        return _empty_range_dataframe()

    return _parse_matrix(result)


def _parse_matrix(result: list[dict]) -> pl.DataFrame:
    all_label_keys: set[str] = set()
    for item in result:
        all_label_keys.update(item.get("metric", {}).keys())

    records: list[dict[str, object]] = []
    for item in result:
        metric = item.get("metric", {})
        for ts, value_str in item.get("values", []):
            record: dict[str, object] = {"__name__": metric.get("__name__", "")}
            for key in sorted(all_label_keys):
                if key != "__name__":
                    record[key] = metric.get(key, "")
            record["timestamp"] = float(ts)
            record["value"] = float(value_str)
            records.append(record)

    if not records:
        return _empty_range_dataframe()

    return pl.DataFrame(records)
