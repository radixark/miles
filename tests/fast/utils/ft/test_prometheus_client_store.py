"""Tests for PrometheusClient (MetricStoreProtocol backed by real Prometheus HTTP API)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import httpx
import polars as pl
import pytest

from miles.utils.ft.controller.prometheus_client_store import PrometheusClient


def _make_response(json_data: dict[str, Any], status_code: int = 200) -> httpx.Response:
    """Build a fake httpx.Response from a dict."""
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("GET", "http://fake:9090/api/v1/query"),
    )


class TestInstantQueryVector:
    def test_single_series(self) -> None:
        json_data = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {
                        "metric": {"__name__": "up", "job": "node", "instance": "localhost:9090"},
                        "value": [1709000000, "1"],
                    }
                ],
            },
        }

        with patch.object(httpx.Client, "get", return_value=_make_response(json_data)):
            client = PrometheusClient(url="http://fake:9090")
            df = client.instant_query("up")

        assert df.shape[0] == 1
        assert "__name__" in df.columns
        assert "value" in df.columns
        assert df["__name__"][0] == "up"
        assert df["value"][0] == 1.0

    def test_multiple_series(self) -> None:
        json_data = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {
                        "metric": {"__name__": "cpu_usage", "host": "a"},
                        "value": [1709000000, "0.5"],
                    },
                    {
                        "metric": {"__name__": "cpu_usage", "host": "b"},
                        "value": [1709000000, "0.8"],
                    },
                ],
            },
        }

        with patch.object(httpx.Client, "get", return_value=_make_response(json_data)):
            client = PrometheusClient(url="http://fake:9090")
            df = client.instant_query("cpu_usage")

        assert df.shape[0] == 2
        assert "host" in df.columns
        values = sorted(df["value"].to_list())
        assert values == [0.5, 0.8]

    def test_empty_result(self) -> None:
        json_data = {
            "status": "success",
            "data": {"resultType": "vector", "result": []},
        }

        with patch.object(httpx.Client, "get", return_value=_make_response(json_data)):
            client = PrometheusClient(url="http://fake:9090")
            df = client.instant_query("nonexistent")

        assert df.is_empty()
        assert "__name__" in df.columns
        assert "value" in df.columns


class TestInstantQueryErrors:
    def test_http_500_returns_empty(self) -> None:
        error_response = httpx.Response(
            status_code=500,
            text="Internal Server Error",
            request=httpx.Request("GET", "http://fake:9090/api/v1/query"),
        )

        with patch.object(httpx.Client, "get", return_value=error_response):
            client = PrometheusClient(url="http://fake:9090")
            df = client.instant_query("up")

        assert df.is_empty()
        assert "__name__" in df.columns
        assert "value" in df.columns

    def test_timeout_returns_empty(self) -> None:
        with patch.object(httpx.Client, "get", side_effect=httpx.TimeoutException("timed out")):
            client = PrometheusClient(url="http://fake:9090")
            df = client.instant_query("up")

        assert df.is_empty()

    def test_prometheus_error_status(self) -> None:
        json_data = {
            "status": "error",
            "errorType": "bad_data",
            "error": "invalid query",
        }

        with patch.object(httpx.Client, "get", return_value=_make_response(json_data)):
            client = PrometheusClient(url="http://fake:9090")
            df = client.instant_query("bad{[query")

        assert df.is_empty()


class TestRangeQuery:
    def test_matrix_result(self) -> None:
        json_data = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {"__name__": "cpu_usage", "host": "a"},
                        "values": [
                            [1709000000, "0.3"],
                            [1709000060, "0.5"],
                            [1709000120, "0.7"],
                        ],
                    }
                ],
            },
        }

        with patch.object(httpx.Client, "get", return_value=_make_response(json_data)):
            client = PrometheusClient(url="http://fake:9090")
            df = client.range_query(
                query="cpu_usage",
                start=datetime(2024, 2, 27, tzinfo=timezone.utc),
                end=datetime(2024, 2, 27, 1, tzinfo=timezone.utc),
                step=timedelta(minutes=1),
            )

        assert df.shape[0] == 3
        assert "__name__" in df.columns
        assert "timestamp" in df.columns
        assert "value" in df.columns
        assert df["value"].to_list() == [0.3, 0.5, 0.7]

    def test_multiple_series_matrix(self) -> None:
        json_data = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {"__name__": "m", "host": "a"},
                        "values": [[1709000000, "1"]],
                    },
                    {
                        "metric": {"__name__": "m", "host": "b"},
                        "values": [[1709000000, "2"]],
                    },
                ],
            },
        }

        with patch.object(httpx.Client, "get", return_value=_make_response(json_data)):
            client = PrometheusClient(url="http://fake:9090")
            df = client.range_query(
                query="m",
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
                step=timedelta(hours=1),
            )

        assert df.shape[0] == 2
        assert sorted(df["value"].to_list()) == [1.0, 2.0]

    def test_empty_range_result(self) -> None:
        json_data = {
            "status": "success",
            "data": {"resultType": "matrix", "result": []},
        }

        with patch.object(httpx.Client, "get", return_value=_make_response(json_data)):
            client = PrometheusClient(url="http://fake:9090")
            df = client.range_query(
                query="nonexistent",
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
                step=timedelta(hours=1),
            )

        assert df.is_empty()
        assert "__name__" in df.columns
        assert "timestamp" in df.columns
        assert "value" in df.columns


class TestRangeQueryErrors:
    def test_timeout_returns_empty(self) -> None:
        with patch.object(httpx.Client, "get", side_effect=httpx.TimeoutException("timed out")):
            client = PrometheusClient(url="http://fake:9090")
            df = client.range_query(
                query="up",
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
                step=timedelta(hours=1),
            )

        assert df.is_empty()
        assert "timestamp" in df.columns


class TestLabelColumns:
    def test_label_columns_present_in_vector(self) -> None:
        json_data = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {
                        "metric": {"__name__": "up", "node_id": "node-0", "gpu": "0"},
                        "value": [1709000000, "1"],
                    },
                ],
            },
        }

        with patch.object(httpx.Client, "get", return_value=_make_response(json_data)):
            client = PrometheusClient(url="http://fake:9090")
            df = client.instant_query("up")

        assert "node_id" in df.columns
        assert "gpu" in df.columns
        assert df["node_id"][0] == "node-0"
        assert df["gpu"][0] == "0"


class TestInstantQueryScalar:
    def test_valid_scalar(self) -> None:
        json_data: dict[str, Any] = {
            "status": "success",
            "data": {"resultType": "scalar", "result": [1709000000, "42"]},
        }

        with patch.object(httpx.Client, "get", return_value=_make_response(json_data)):
            client = PrometheusClient(url="http://fake:9090")
            df = client.instant_query("scalar(up)")

        assert df.shape[0] == 1
        assert df["value"][0] == 42.0

    def test_malformed_scalar_short_list(self) -> None:
        json_data: dict[str, Any] = {
            "status": "success",
            "data": {"resultType": "scalar", "result": [1709000000]},
        }

        with patch.object(httpx.Client, "get", return_value=_make_response(json_data)):
            client = PrometheusClient(url="http://fake:9090")
            df = client.instant_query("scalar(up)")

        assert df.is_empty()

    def test_non_numeric_scalar_value(self) -> None:
        json_data: dict[str, Any] = {
            "status": "success",
            "data": {"resultType": "scalar", "result": [1709000000, "not_a_number"]},
        }

        with patch.object(httpx.Client, "get", return_value=_make_response(json_data)):
            client = PrometheusClient(url="http://fake:9090")
            df = client.instant_query("scalar(up)")

        assert df.is_empty()


class TestUnsupportedResultTypes:
    def test_unsupported_instant_result_type(self) -> None:
        json_data: dict[str, Any] = {
            "status": "success",
            "data": {"resultType": "string", "result": "hello"},
        }

        with patch.object(httpx.Client, "get", return_value=_make_response(json_data)):
            client = PrometheusClient(url="http://fake:9090")
            df = client.instant_query("some_string_query")

        assert df.is_empty()
        assert "__name__" in df.columns
        assert "value" in df.columns

    def test_unsupported_range_result_type(self) -> None:
        json_data: dict[str, Any] = {
            "status": "success",
            "data": {"resultType": "vector", "result": []},
        }

        with patch.object(httpx.Client, "get", return_value=_make_response(json_data)):
            client = PrometheusClient(url="http://fake:9090")
            df = client.range_query(
                query="up",
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
                step=timedelta(hours=1),
            )

        assert df.is_empty()

    def test_null_data_section(self) -> None:
        json_data: dict[str, Any] = {"status": "success", "data": None}

        with patch.object(httpx.Client, "get", return_value=_make_response(json_data)):
            client = PrometheusClient(url="http://fake:9090")
            df = client.instant_query("up")

        assert df.is_empty()


class TestMalformedVectorValues:
    def test_null_value_pair_skipped(self) -> None:
        json_data: dict[str, Any] = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {"metric": {"__name__": "m"}, "value": None},
                    {"metric": {"__name__": "m"}, "value": [1709000000, "1"]},
                ],
            },
        }

        with patch.object(httpx.Client, "get", return_value=_make_response(json_data)):
            client = PrometheusClient(url="http://fake:9090")
            df = client.instant_query("m")

        assert df.shape[0] == 1
        assert df["value"][0] == 1.0

    def test_non_numeric_value_skipped(self) -> None:
        json_data: dict[str, Any] = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {"metric": {"__name__": "m"}, "value": [1709000000, "bad"]},
                    {"metric": {"__name__": "m"}, "value": [1709000000, "2.5"]},
                ],
            },
        }

        with patch.object(httpx.Client, "get", return_value=_make_response(json_data)):
            client = PrometheusClient(url="http://fake:9090")
            df = client.instant_query("m")

        assert df.shape[0] == 1
        assert df["value"][0] == 2.5


class TestMalformedMatrixValues:
    def test_non_numeric_value_str_skipped(self) -> None:
        json_data: dict[str, Any] = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {"__name__": "m"},
                        "values": [
                            [1709000000, "bad"],
                            [1709000060, "1.5"],
                        ],
                    }
                ],
            },
        }

        with patch.object(httpx.Client, "get", return_value=_make_response(json_data)):
            client = PrometheusClient(url="http://fake:9090")
            df = client.range_query(
                query="m",
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
                step=timedelta(hours=1),
            )

        assert df.shape[0] == 1
        assert df["value"][0] == 1.5

    def test_null_metric_uses_empty_name(self) -> None:
        json_data: dict[str, Any] = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": None,
                        "values": [[1709000000, "3.0"]],
                    }
                ],
            },
        }

        with patch.object(httpx.Client, "get", return_value=_make_response(json_data)):
            client = PrometheusClient(url="http://fake:9090")
            df = client.range_query(
                query="m",
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
                step=timedelta(hours=1),
            )

        assert df.shape[0] == 1
        assert df["__name__"][0] == ""
