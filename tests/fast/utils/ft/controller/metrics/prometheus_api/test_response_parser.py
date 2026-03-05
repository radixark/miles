"""Tests for prometheus_api/response_parser.py error and edge-case paths."""
from __future__ import annotations

from miles.utils.ft.controller.metrics.mini_prometheus.query import EMPTY_INSTANT, EMPTY_RANGE
from miles.utils.ft.controller.metrics.prometheus_api.response_parser import (
    parse_instant_response,
    parse_range_response,
)


class TestParseInstantResponseErrors:
    def test_non_success_status_returns_empty(self) -> None:
        data = {"status": "error", "errorType": "bad_data", "error": "parse error"}
        result = parse_instant_response(data)
        assert result.shape == EMPTY_INSTANT.shape
        assert len(result) == 0

    def test_missing_status_returns_empty(self) -> None:
        result = parse_instant_response({})
        assert len(result) == 0

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
    def test_non_success_returns_empty(self) -> None:
        result = parse_range_response({"status": "error"})
        assert result.shape == EMPTY_RANGE.shape
        assert len(result) == 0

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
