"""Tests for PrometheusClient (MetricStoreProtocol backed by real Prometheus HTTP API)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from miles.utils.ft.controller.metrics.prometheus_api.client import (
    PrometheusClient,
    _build_selector,
    _escape_promql_label_value,
    _format_duration,
)
from miles.utils.ft.controller.metrics.prometheus_api.errors import PrometheusQueryError


def _make_response(json_data: dict[str, Any], status_code: int = 200) -> httpx.Response:
    """Build a fake httpx.Response from a dict."""
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("GET", "http://fake:9090/api/v1/query"),
    )


@contextmanager
def _mock_prometheus_client(response_json: dict[str, Any]) -> Iterator[PrometheusClient]:
    with patch.object(httpx.Client, "get", return_value=_make_response(response_json)):
        yield PrometheusClient(url="http://fake:9090")


class TestQueryLatestVector:
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

        with _mock_prometheus_client(json_data) as client:
            df = client.query_latest("up")

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

        with _mock_prometheus_client(json_data) as client:
            df = client.query_latest("cpu_usage")

        assert df.shape[0] == 2
        assert "host" in df.columns
        values = sorted(df["value"].to_list())
        assert values == [0.5, 0.8]

    def test_empty_result(self) -> None:
        json_data = {
            "status": "success",
            "data": {"resultType": "vector", "result": []},
        }

        with _mock_prometheus_client(json_data) as client:
            df = client.query_latest("nonexistent")

        assert df.is_empty()
        assert "__name__" in df.columns
        assert "value" in df.columns


class TestQueryLatestErrors:
    def test_http_500_raises(self) -> None:
        error_response = httpx.Response(
            status_code=500,
            text="Internal Server Error",
            request=httpx.Request("GET", "http://fake:9090/api/v1/query"),
        )

        with patch.object(httpx.Client, "get", return_value=error_response):
            client = PrometheusClient(url="http://fake:9090")
            with pytest.raises(PrometheusQueryError):
                client.query_latest("up")

    def test_timeout_raises(self) -> None:
        with patch.object(httpx.Client, "get", side_effect=httpx.TimeoutException("timed out")):
            client = PrometheusClient(url="http://fake:9090")
            with pytest.raises(PrometheusQueryError):
                client.query_latest("up")

    def test_prometheus_error_status_raises(self) -> None:
        json_data = {
            "status": "error",
            "errorType": "bad_data",
            "error": "invalid query",
        }

        with _mock_prometheus_client(json_data) as client:
            with pytest.raises(PrometheusQueryError, match="status=error"):
                client.query_latest("some_metric")


class TestQueryRange:
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

        with _mock_prometheus_client(json_data) as client:
            df = client.query_range("cpu_usage", window=timedelta(hours=1))

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

        with _mock_prometheus_client(json_data) as client:
            df = client.query_range("m", window=timedelta(hours=24))

        assert df.shape[0] == 2
        assert sorted(df["value"].to_list()) == [1.0, 2.0]

    def test_empty_range_result(self) -> None:
        json_data = {
            "status": "success",
            "data": {"resultType": "matrix", "result": []},
        }

        with _mock_prometheus_client(json_data) as client:
            df = client.query_range("nonexistent", window=timedelta(hours=24))

        assert df.is_empty()
        assert "__name__" in df.columns
        assert "timestamp" in df.columns
        assert "value" in df.columns


class TestQueryRangeErrors:
    def test_timeout_raises(self) -> None:
        with patch.object(httpx.Client, "get", side_effect=httpx.TimeoutException("timed out")):
            client = PrometheusClient(url="http://fake:9090")
            with pytest.raises(PrometheusQueryError):
                client.query_range("up", window=timedelta(hours=24))


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

        with _mock_prometheus_client(json_data) as client:
            df = client.query_latest("up")

        assert "node_id" in df.columns
        assert "gpu" in df.columns
        assert df["node_id"][0] == "node-0"
        assert df["gpu"][0] == "0"


class TestQueryLatestScalar:
    def test_valid_scalar(self) -> None:
        json_data: dict[str, Any] = {
            "status": "success",
            "data": {"resultType": "scalar", "result": [1709000000, "42"]},
        }

        with _mock_prometheus_client(json_data) as client:
            df = client.query_latest("scalar_metric")

        assert df.shape[0] == 1
        assert df["value"][0] == 42.0

    def test_malformed_scalar_short_list(self) -> None:
        json_data: dict[str, Any] = {
            "status": "success",
            "data": {"resultType": "scalar", "result": [1709000000]},
        }

        with _mock_prometheus_client(json_data) as client:
            df = client.query_latest("scalar_metric")

        assert df.is_empty()

    def test_non_numeric_scalar_value(self) -> None:
        json_data: dict[str, Any] = {
            "status": "success",
            "data": {"resultType": "scalar", "result": [1709000000, "not_a_number"]},
        }

        with _mock_prometheus_client(json_data) as client:
            df = client.query_latest("scalar_metric")

        assert df.is_empty()


class TestUnsupportedResultTypes:
    def test_unsupported_instant_result_type(self) -> None:
        json_data: dict[str, Any] = {
            "status": "success",
            "data": {"resultType": "string", "result": "hello"},
        }

        with _mock_prometheus_client(json_data) as client:
            df = client.query_latest("some_metric")

        assert df.is_empty()
        assert "__name__" in df.columns
        assert "value" in df.columns

    def test_unsupported_range_result_type(self) -> None:
        json_data: dict[str, Any] = {
            "status": "success",
            "data": {"resultType": "vector", "result": []},
        }

        with _mock_prometheus_client(json_data) as client:
            df = client.query_range("up", window=timedelta(hours=24))

        assert df.is_empty()

    def test_null_data_section(self) -> None:
        json_data: dict[str, Any] = {"status": "success", "data": None}

        with _mock_prometheus_client(json_data) as client:
            df = client.query_latest("up")

        assert df.is_empty()


class TestMalformedVectorValues:
    def test_null_value_pair_defaults_to_zero(self) -> None:
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

        with _mock_prometheus_client(json_data) as client:
            df = client.query_latest("m")

        assert df.shape[0] == 2
        assert sorted(df["value"].to_list()) == [0.0, 1.0]

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

        with _mock_prometheus_client(json_data) as client:
            df = client.query_latest("m")

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

        with _mock_prometheus_client(json_data) as client:
            df = client.query_range("m", window=timedelta(hours=24))

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

        with _mock_prometheus_client(json_data) as client:
            df = client.query_range("m", window=timedelta(hours=24))

        assert df.shape[0] == 1
        assert df["__name__"][0] == ""


class TestRangeFunctionQueries:
    def test_changes_builds_correct_promql(self) -> None:
        json_data = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {
                        "metric": {"__name__": "training_iteration"},
                        "value": [1709000000, "3"],
                    }
                ],
            },
        }

        with patch.object(httpx.Client, "get", return_value=_make_response(json_data)) as mock_get:
            client = PrometheusClient(url="http://fake:9090")
            df = client.changes("training_iteration", window=timedelta(minutes=10))

            call_args = mock_get.call_args
            params = call_args.kwargs.get("params") or call_args[1].get("params")
            assert params["query"] == "changes(training_iteration[10m])"

        assert df.shape[0] == 1
        assert df["value"][0] == 3.0

    def test_count_over_time_with_label_filters(self) -> None:
        json_data = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {
                        "metric": {"__name__": "alerts", "node_id": "node-0"},
                        "value": [1709000000, "5"],
                    }
                ],
            },
        }

        with patch.object(httpx.Client, "get", return_value=_make_response(json_data)) as mock_get:
            client = PrometheusClient(url="http://fake:9090")
            df = client.count_over_time(
                "alerts",
                window=timedelta(minutes=5),
                label_filters={"node_id": "node-0"},
            )

            call_args = mock_get.call_args
            params = call_args.kwargs.get("params") or call_args[1].get("params")
            assert params["query"] == 'count_over_time(alerts{node_id="node-0"}[5m])'

        assert df.shape[0] == 1
        assert df["value"][0] == 5.0


class TestAsyncQueryLatest:
    @pytest.mark.asyncio
    async def test_aquery_latest_delegates_to_fetch_json(self) -> None:
        json_data = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {
                        "metric": {"__name__": "up", "node_id": "node-0"},
                        "value": [1709000000, "1"],
                    }
                ],
            },
        }

        with _mock_prometheus_client(json_data) as client:
            df = await client.aquery_latest("up")

        assert df.shape[0] == 1
        assert df["value"][0] == 1.0

    @pytest.mark.asyncio
    async def test_aquery_latest_raises_on_error(self) -> None:
        with patch.object(httpx.Client, "get", side_effect=httpx.TimeoutException("timed out")):
            client = PrometheusClient(url="http://fake:9090")
            with pytest.raises(PrometheusQueryError):
                await client.aquery_latest("up")


class TestAsyncQueryRange:
    @pytest.mark.asyncio
    async def test_aquery_range_returns_data(self) -> None:
        json_data = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {"__name__": "cpu", "host": "a"},
                        "values": [
                            [1709000000, "0.3"],
                            [1709000060, "0.5"],
                        ],
                    }
                ],
            },
        }

        with _mock_prometheus_client(json_data) as client:
            df = await client.aquery_range("cpu", window=timedelta(hours=1))

        assert df.shape[0] == 2
        assert df["value"].to_list() == [0.3, 0.5]

    @pytest.mark.asyncio
    async def test_aquery_range_raises_on_error(self) -> None:
        with patch.object(httpx.Client, "get", side_effect=httpx.TimeoutException("timed out")):
            client = PrometheusClient(url="http://fake:9090")
            with pytest.raises(PrometheusQueryError):
                await client.aquery_range("up", window=timedelta(hours=1))


class TestFormatDuration:
    def test_exact_days(self) -> None:
        assert _format_duration(timedelta(days=1)) == "1d"
        assert _format_duration(timedelta(days=7)) == "7d"

    def test_exact_hours(self) -> None:
        assert _format_duration(timedelta(hours=1)) == "1h"
        assert _format_duration(timedelta(hours=12)) == "12h"

    def test_exact_minutes(self) -> None:
        assert _format_duration(timedelta(minutes=5)) == "5m"
        assert _format_duration(timedelta(minutes=30)) == "30m"

    def test_fallback_to_seconds(self) -> None:
        assert _format_duration(timedelta(seconds=45)) == "45s"
        assert _format_duration(timedelta(minutes=1, seconds=30)) == "90s"

    def test_non_even_hours_falls_to_minutes(self) -> None:
        assert _format_duration(timedelta(hours=1, minutes=30)) == "90m"


# ---------------------------------------------------------------------------
# NaN / Inf handling
# ---------------------------------------------------------------------------


class TestNanAndInfValues:
    def test_nan_value_parsed_correctly(self) -> None:
        json_data = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {"metric": {"__name__": "m"}, "value": [1709000000, "NaN"]},
                ],
            },
        }

        with _mock_prometheus_client(json_data) as client:
            df = client.query_latest("m")

        assert df.shape[0] == 1
        import math

        assert math.isnan(df["value"][0])

    def test_positive_inf_value_parsed(self) -> None:
        json_data = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {"metric": {"__name__": "m"}, "value": [1709000000, "+Inf"]},
                ],
            },
        }

        with _mock_prometheus_client(json_data) as client:
            df = client.query_latest("m")

        assert df.shape[0] == 1
        assert df["value"][0] == float("inf")

    def test_negative_inf_value_parsed(self) -> None:
        json_data = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {"metric": {"__name__": "m"}, "value": [1709000000, "-Inf"]},
                ],
            },
        }

        with _mock_prometheus_client(json_data) as client:
            df = client.query_latest("m")

        assert df.shape[0] == 1
        assert df["value"][0] == float("-inf")

    def test_nan_in_matrix_values(self) -> None:
        json_data = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {"__name__": "m"},
                        "values": [
                            [1709000000, "1.0"],
                            [1709000060, "NaN"],
                            [1709000120, "3.0"],
                        ],
                    }
                ],
            },
        }

        with _mock_prometheus_client(json_data) as client:
            df = client.query_range("m", window=timedelta(hours=1))

        assert df.shape[0] == 3
        import math

        assert df["value"][0] == 1.0
        assert math.isnan(df["value"][1])
        assert df["value"][2] == 3.0


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


class TestRetryBehaviour:
    def test_retry_succeeds_on_second_attempt(self) -> None:
        success_json = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {"metric": {"__name__": "up"}, "value": [1709000000, "1"]},
                ],
            },
        }
        error_response = httpx.Response(
            status_code=500,
            text="Internal Server Error",
            request=httpx.Request("GET", "http://fake:9090/api/v1/query"),
        )
        call_count = 0

        def _side_effect(*args: object, **kwargs: object) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return error_response
            return _make_response(success_json)

        with patch.object(httpx.Client, "get", side_effect=_side_effect):
            client = PrometheusClient(url="http://fake:9090")
            df = client.query_latest("up")

        assert call_count == 2
        assert df.shape[0] == 1
        assert df["value"][0] == 1.0

    def test_retry_exhausted_raises(self) -> None:
        error_response = httpx.Response(
            status_code=500,
            text="Internal Server Error",
            request=httpx.Request("GET", "http://fake:9090/api/v1/query"),
        )
        call_count = 0

        def _side_effect(*args: object, **kwargs: object) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return error_response

        with patch.object(httpx.Client, "get", side_effect=_side_effect):
            client = PrometheusClient(url="http://fake:9090")
            with pytest.raises(PrometheusQueryError):
                client.query_latest("up")

        assert call_count == 2


# ---------------------------------------------------------------------------
# avg_over_time PromQL construction
# ---------------------------------------------------------------------------


class TestAvgOverTimePromQL:
    def test_avg_over_time_builds_correct_promql(self) -> None:
        json_data = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {"metric": {"__name__": "cpu"}, "value": [1709000000, "0.75"]},
                ],
            },
        }

        with patch.object(httpx.Client, "get", return_value=_make_response(json_data)) as mock_get:
            client = PrometheusClient(url="http://fake:9090")
            df = client.avg_over_time("cpu", window=timedelta(minutes=5))

            call_args = mock_get.call_args
            params = call_args.kwargs.get("params") or call_args[1].get("params")
            assert params["query"] == "avg_over_time(cpu[5m])"

        assert df.shape[0] == 1
        assert df["value"][0] == 0.75

    def test_avg_over_time_with_label_filters(self) -> None:
        json_data = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {
                        "metric": {"__name__": "cpu", "node_id": "node-0"},
                        "value": [1709000000, "0.6"],
                    },
                ],
            },
        }

        with patch.object(httpx.Client, "get", return_value=_make_response(json_data)) as mock_get:
            client = PrometheusClient(url="http://fake:9090")
            client.avg_over_time(
                "cpu",
                window=timedelta(hours=1),
                label_filters={"node_id": "node-0"},
            )

            call_args = mock_get.call_args
            params = call_args.kwargs.get("params") or call_args[1].get("params")
            assert params["query"] == 'avg_over_time(cpu{node_id="node-0"}[1h])'


# ---------------------------------------------------------------------------
# PromQL label escaping
# ---------------------------------------------------------------------------


class TestPromQLLabelEscaping:
    def test_label_value_with_double_quotes(self) -> None:
        escaped = _escape_promql_label_value('value with "quotes"')
        assert escaped == 'value with \\"quotes\\"'

    def test_label_value_with_backslash(self) -> None:
        escaped = _escape_promql_label_value("path\\to\\file")
        assert escaped == "path\\\\to\\\\file"

    def test_label_value_with_newline(self) -> None:
        escaped = _escape_promql_label_value("line1\nline2")
        assert escaped == "line1\\nline2"

    def test_build_selector_escapes_label_values(self) -> None:
        selector = _build_selector("metric", {"key": 'val"ue'})
        assert selector == 'metric{key="val\\"ue"}'


# ---------------------------------------------------------------------------
# Lifecycle: stop() then query
# ---------------------------------------------------------------------------


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_query_after_stop_raises(self) -> None:
        client = PrometheusClient(url="http://fake:9090")
        await client.stop()

        with pytest.raises(PrometheusQueryError):
            client.query_latest("some_metric")
