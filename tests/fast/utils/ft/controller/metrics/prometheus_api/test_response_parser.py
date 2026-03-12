"""Tests for prometheus_api/response_parser.py error and edge-case paths."""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from miles.utils.ft.controller.metrics.prometheus_api.errors import PrometheusQueryError
from miles.utils.ft.controller.metrics.prometheus_api.response_parser import (
    parse_instant_response,
    parse_range_response,
)


class TestParseInstantResponseErrors:
    def test_non_success_status_raises(self) -> None:
        data = {"status": "error", "errorType": "bad_data", "error": "parse error"}
        with pytest.raises(PrometheusQueryError, match="status=error"):
            parse_instant_response(data)

    def test_missing_status_raises(self) -> None:
        with pytest.raises(PrometheusQueryError, match="status=None"):
            parse_instant_response({})

    def test_empty_result_returns_empty(self) -> None:
        data = {"status": "success", "data": {"resultType": "vector", "result": []}}
        result = parse_instant_response(data)
        assert len(result) == 0

    def test_unsupported_result_type_returns_empty(self) -> None:
        data = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [{"metric": {}, "values": [[1.0, "42"]]}],
            },
        }
        result = parse_instant_response(data)
        assert len(result) == 0


class TestParseInstantResponseScalar:
    def test_scalar_result_type(self) -> None:
        data = {
            "status": "success",
            "data": {"resultType": "scalar", "result": [1609459200.0, "42.5"]},
        }
        result = parse_instant_response(data)
        assert len(result) == 1
        assert result["value"][0] == 42.5

    def test_scalar_wrong_length_returns_empty(self) -> None:
        data = {
            "status": "success",
            "data": {"resultType": "scalar", "result": [1609459200.0]},
        }
        result = parse_instant_response(data)
        assert len(result) == 0

    def test_scalar_non_numeric_value_returns_empty(self) -> None:
        data = {
            "status": "success",
            "data": {"resultType": "scalar", "result": [1609459200.0, "not_a_number"]},
        }
        result = parse_instant_response(data)
        assert len(result) == 0


class TestParseInstantResponseVector:
    def test_vector_with_bad_value_pair_skipped(self) -> None:
        data = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {"metric": {"__name__": "a"}, "value": [1.0, "99.5"]},
                    {"metric": {"__name__": "b"}, "value": [1.0, "not_a_number"]},
                ],
            },
        }
        result = parse_instant_response(data)
        assert len(result) == 1
        assert result["__name__"][0] == "a"
        assert result["value"][0] == 99.5

    def test_vector_all_bad_values_returns_empty(self) -> None:
        data = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {"metric": {"__name__": "a"}, "value": [1.0, "NaN_garbage"]},
                ],
            },
        }
        result = parse_instant_response(data)
        assert len(result) == 0


class TestParseRangeResponseErrors:
    def test_non_success_raises(self) -> None:
        with pytest.raises(PrometheusQueryError, match="status=error"):
            parse_range_response({"status": "error"})

    def test_non_matrix_type_returns_empty(self) -> None:
        data = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [{"metric": {}, "value": [1.0, "42"]}],
            },
        }
        result = parse_range_response(data)
        assert len(result) == 0

    def test_matrix_with_bad_values_skipped(self) -> None:
        data = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {"__name__": "m"},
                        "values": [
                            [1.0, "100"],
                            [2.0, "not_a_number"],
                            [3.0, "300"],
                        ],
                    },
                ],
            },
        }
        result = parse_range_response(data)
        assert len(result) == 2
        values = sorted(result["value"].to_list())
        assert values == [100.0, 300.0]

    def test_matrix_all_bad_values_returns_empty(self) -> None:
        data = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {"metric": {"__name__": "m"}, "values": [["bad_ts", "bad_val"]]},
                ],
            },
        }
        result = parse_range_response(data)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# P2 item 24: additional response_parser edge cases
# ---------------------------------------------------------------------------


class TestParseRangeResponseEdgeCases:
    def test_empty_result_array_returns_empty(self) -> None:
        data = {"status": "success", "data": {"resultType": "matrix", "result": []}}
        result = parse_range_response(data)
        assert len(result) == 0

    def test_matrix_with_nan_inf_strings(self) -> None:
        """NaN and Inf as string values should parse as valid floats."""
        data = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {"__name__": "m"},
                        "values": [
                            [1.0, "NaN"],
                            [2.0, "Inf"],
                            [3.0, "-Inf"],
                            [4.0, "42.0"],
                        ],
                    },
                ],
            },
        }
        result = parse_range_response(data)
        assert len(result) == 4

    def test_missing_data_section_returns_empty(self) -> None:
        data = {"status": "success"}
        result = parse_range_response(data)
        assert len(result) == 0


class TestParseRangeTimestampType:
    """C-1: PrometheusClient used to return float timestamps, causing
    AttributeError on (max - min).total_seconds(). Now timestamps are
    converted to datetime in _parse_matrix_item."""

    def test_range_response_returns_datetime_timestamps(self) -> None:
        data = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {"__name__": "m"},
                        "values": [
                            [1700000000.0, "1.0"],
                            [1700000060.0, "2.0"],
                        ],
                    },
                ],
            },
        }
        result = parse_range_response(data)
        assert result["timestamp"].dtype == pl.Datetime("us", "UTC")
        ts_min = result["timestamp"].min()
        ts_max = result["timestamp"].max()
        assert isinstance(ts_min, datetime)
        assert isinstance(ts_max, datetime)
        time_span = (ts_max - ts_min).total_seconds()
        assert time_span == pytest.approx(60.0)

    def test_empty_range_response_has_datetime_dtype(self) -> None:
        data = {"status": "success", "data": {"resultType": "matrix", "result": []}}
        result = parse_range_response(data)
        assert result["timestamp"].dtype == pl.Datetime("us", "UTC")
