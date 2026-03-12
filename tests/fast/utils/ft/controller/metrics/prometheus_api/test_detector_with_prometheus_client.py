"""Detector-level integration tests using PrometheusClient as metric store.

All existing detector tests use MiniPrometheus. These tests verify that
detector logic works correctly with PrometheusClient's DataFrame output,
especially the float timestamp (vs MiniPrometheus's datetime).
"""

from __future__ import annotations

import time
from datetime import timedelta
from typing import Any
from unittest.mock import patch

import httpx

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.detectors.base import DetectorContext
from miles.utils.ft.controller.detectors.checks.hardware import (
    check_disk_fault,
    check_majority_nic_down,
    check_nic_down_in_window,
)
from miles.utils.ft.controller.detectors.core.hang import HangDetector, HangDetectorConfig
from miles.utils.ft.controller.metrics.metric_names import (
    AGENT_HEARTBEAT,
    NODE_FILESYSTEM_AVAIL_BYTES,
    NODE_NETWORK_UP,
    PHASE_TRAINING,
    TRAINING_PHASE,
)
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.metrics.prometheus_api.client import PrometheusClient
from miles.utils.ft.controller.types import ActionType, MetricStore


def _make_response(json_data: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json=json_data,
        request=httpx.Request("GET", "http://fake:9090/api/v1/query"),
    )


def _vector_json(
    metric_name: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": "success",
        "data": {"resultType": "vector", "result": results},
    }


def _matrix_json(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": "success",
        "data": {"resultType": "matrix", "result": results},
    }


def _vector_item(
    metric_name: str,
    value: float,
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    metric = {"__name__": metric_name, **(labels or {})}
    return {"metric": metric, "value": [time.time(), str(value)]}


def _matrix_item(
    metric_name: str,
    values: list[tuple[float, float]],
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    metric = {"__name__": metric_name, **(labels or {})}
    return {
        "metric": metric,
        "values": [[ts, str(v)] for ts, v in values],
    }


class _ResponseRouter:
    """Routes different Prometheus API calls to different JSON responses.

    HangDetector makes multiple queries (query_latest for phase, instant query
    for changes). This router dispatches based on the `query` parameter.
    """

    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self._responses = responses
        self._default = {"status": "success", "data": {"resultType": "vector", "result": []}}

    def __call__(self, url: str, **kwargs: Any) -> httpx.Response:
        params = kwargs.get("params", {})
        query = params.get("query", "")
        for key, resp in self._responses.items():
            if key in query:
                return _make_response(resp)
        return _make_response(self._default)


class TestHangDetectorWithPrometheusClient:
    def test_hang_detected_when_heartbeat_stalled(self) -> None:
        router = _ResponseRouter(
            {
                TRAINING_PHASE: _vector_json(
                    TRAINING_PHASE,
                    [_vector_item(TRAINING_PHASE, PHASE_TRAINING, labels={"rank": "0"})],
                ),
                AGENT_HEARTBEAT: _vector_json(
                    AGENT_HEARTBEAT,
                    [_vector_item(AGENT_HEARTBEAT, 0.0, labels={"rank": "0"})],
                ),
            }
        )

        with patch.object(httpx.Client, "get", side_effect=router):
            client = PrometheusClient(url="http://fake:9090")
            ctx = DetectorContext(
                metric_store=MetricStore(time_series_store=client, mini_wandb=MiniWandb()),
                active_node_ids={"node-0"},
                job_status=JobStatus.RUNNING,
            )

            detector = HangDetector(config=HangDetectorConfig(training_timeout_minutes=5))
            decision = detector.evaluate(ctx)

        assert decision.action == ActionType.ENTER_RECOVERY

    def test_no_hang_when_heartbeat_progressing(self) -> None:
        router = _ResponseRouter(
            {
                TRAINING_PHASE: _vector_json(
                    TRAINING_PHASE,
                    [_vector_item(TRAINING_PHASE, PHASE_TRAINING, labels={"rank": "0"})],
                ),
                AGENT_HEARTBEAT: _vector_json(
                    AGENT_HEARTBEAT,
                    [_vector_item(AGENT_HEARTBEAT, 5.0, labels={"rank": "0"})],
                ),
            }
        )

        with patch.object(httpx.Client, "get", side_effect=router):
            client = PrometheusClient(url="http://fake:9090")
            ctx = DetectorContext(
                metric_store=MetricStore(time_series_store=client, mini_wandb=MiniWandb()),
                active_node_ids={"node-0"},
                job_status=JobStatus.RUNNING,
            )

            detector = HangDetector()
            decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE


class TestNicDownDetectionWithPrometheusClient:
    def test_nic_down_transitions_detected(self) -> None:
        """NIC goes up→down: check_nic_down_in_window returns a fault."""
        now = time.time()
        matrix_result = _matrix_json(
            [
                _matrix_item(
                    NODE_NETWORK_UP,
                    values=[
                        (now - 30, 1.0),
                        (now - 20, 0.0),
                        (now - 10, 1.0),
                        (now, 0.0),
                    ],
                    labels={"node_id": "node-0", "device": "ib0"},
                ),
            ]
        )

        with patch.object(httpx.Client, "get", return_value=_make_response(matrix_result)):
            client = PrometheusClient(url="http://fake:9090")
            faults = check_nic_down_in_window(
                metric_store=client,
                window=timedelta(minutes=5),
                threshold=1,
            )

        assert len(faults) == 1
        assert faults[0].node_id == "node-0"

    def test_no_nic_fault_when_stable(self) -> None:
        now = time.time()
        matrix_result = _matrix_json(
            [
                _matrix_item(
                    NODE_NETWORK_UP,
                    values=[(now - 30, 1.0), (now - 20, 1.0), (now - 10, 1.0), (now, 1.0)],
                    labels={"node_id": "node-0", "device": "ib0"},
                ),
            ]
        )

        with patch.object(httpx.Client, "get", return_value=_make_response(matrix_result)):
            client = PrometheusClient(url="http://fake:9090")
            faults = check_nic_down_in_window(
                metric_store=client,
                window=timedelta(minutes=5),
                threshold=1,
            )

        assert faults == []


class TestMajorityNicDownWithPrometheusClient:
    def test_majority_nic_down_detected(self) -> None:
        vector_result = _vector_json(
            NODE_NETWORK_UP,
            [
                _vector_item(NODE_NETWORK_UP, 0.0, labels={"node_id": "node-0", "device": "ib0"}),
                _vector_item(NODE_NETWORK_UP, 0.0, labels={"node_id": "node-0", "device": "ib1"}),
                _vector_item(NODE_NETWORK_UP, 1.0, labels={"node_id": "node-0", "device": "ib2"}),
            ],
        )

        with patch.object(httpx.Client, "get", return_value=_make_response(vector_result)):
            client = PrometheusClient(url="http://fake:9090")
            faults = check_majority_nic_down(client)

        assert len(faults) == 1
        assert faults[0].node_id == "node-0"

    def test_no_majority_nic_down_when_healthy(self) -> None:
        vector_result = _vector_json(
            NODE_NETWORK_UP,
            [
                _vector_item(NODE_NETWORK_UP, 1.0, labels={"node_id": "node-0", "device": "ib0"}),
                _vector_item(NODE_NETWORK_UP, 1.0, labels={"node_id": "node-0", "device": "ib1"}),
                _vector_item(NODE_NETWORK_UP, 0.0, labels={"node_id": "node-0", "device": "ib2"}),
            ],
        )

        with patch.object(httpx.Client, "get", return_value=_make_response(vector_result)):
            client = PrometheusClient(url="http://fake:9090")
            faults = check_majority_nic_down(client)

        assert faults == []


class TestDiskFaultWithPrometheusClient:
    def test_disk_fault_detected(self) -> None:
        vector_result = _vector_json(
            NODE_FILESYSTEM_AVAIL_BYTES,
            [
                _vector_item(
                    NODE_FILESYSTEM_AVAIL_BYTES,
                    500_000_000.0,
                    labels={"node_id": "node-0", "mountpoint": "/data"},
                ),
            ],
        )

        with patch.object(httpx.Client, "get", return_value=_make_response(vector_result)):
            client = PrometheusClient(url="http://fake:9090")
            faults = check_disk_fault(client)

        assert len(faults) == 1
        assert faults[0].node_id == "node-0"

    def test_no_disk_fault_when_space_sufficient(self) -> None:
        vector_result = _vector_json(
            NODE_FILESYSTEM_AVAIL_BYTES,
            [
                _vector_item(
                    NODE_FILESYSTEM_AVAIL_BYTES,
                    500_000_000_000.0,
                    labels={"node_id": "node-0", "mountpoint": "/data"},
                ),
            ],
        )

        with patch.object(httpx.Client, "get", return_value=_make_response(vector_result)):
            client = PrometheusClient(url="http://fake:9090")
            faults = check_disk_fault(client)

        assert faults == []
