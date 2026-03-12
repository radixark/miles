from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import polars as pl

from miles.utils.ft.controller.metrics.mini_prometheus.query import EMPTY_INSTANT, EMPTY_RANGE
from miles.utils.ft.controller.metrics.prometheus_api.errors import PrometheusQueryError

logger = logging.getLogger(__name__)


def parse_instant_response(data: dict[str, Any]) -> pl.DataFrame:
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


def parse_range_response(data: dict[str, Any]) -> pl.DataFrame:
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
        raise PrometheusQueryError(f"Prometheus returned status={data.get('status')}: {data.get('error', '')}")

    data_section = data.get("data") or {}
    result = data_section.get("result") or []
    if not result:
        return None

    return result, data_section.get("resultType", "")


def _parse_vector(result: list[dict[str, Any]]) -> pl.DataFrame:
    sorted_label_keys = _collect_sorted_label_keys(result)
    records: list[dict[str, object]] = []

    for item in result:
        record = _parse_vector_item(item, sorted_label_keys)
        if record is not None:
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
        records.extend(_parse_matrix_item(item, sorted_label_keys))

    if not records:
        return EMPTY_RANGE

    return pl.DataFrame(records)


def _parse_vector_item(
    item: dict[str, Any],
    sorted_label_keys: list[str],
) -> dict[str, object] | None:
    metric: dict[str, str] = item.get("metric") or {}
    value_pair = item.get("value") or [0, "0"]
    try:
        parsed_value = float(value_pair[1])
    except (IndexError, TypeError, ValueError):
        return None

    record = _build_label_record(metric, sorted_label_keys)
    record["value"] = parsed_value
    return record


def _parse_matrix_item(
    item: dict[str, Any],
    sorted_label_keys: list[str],
) -> list[dict[str, object]]:
    metric: dict[str, str] = item.get("metric") or {}
    records: list[dict[str, object]] = []
    for ts, value_str in item.get("values") or []:
        try:
            parsed_value = float(value_str)
            parsed_ts = float(ts)
        except (TypeError, ValueError):
            continue

        record = _build_label_record(metric, sorted_label_keys)
        record["timestamp"] = datetime.fromtimestamp(parsed_ts, tz=timezone.utc)
        record["value"] = parsed_value
        records.append(record)
    return records


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
