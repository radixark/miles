"""Integration test: Controller with PrometheusClient in prometheus mode.

Verifies that the Controller can tick with PrometheusClient as the metric store
and that the ControllerExporter gauges are updated correctly.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
import pytest
from tests.fast.utils.ft.conftest import FakeMainJob, FakeNodeManager, get_sample_value, make_test_exporter

import miles.utils.ft.controller.metrics.metric_names as mn
from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.factory import create_ft_controller
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.metrics.prometheus_api.client import PrometheusClient
from miles.utils.ft.controller.types import MetricStore


def _make_prom_response(
    result_type: str = "vector",
    result: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "status": "success",
        "data": {"resultType": result_type, "result": result or []},
    }


def _make_http_response(json_data: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json=json_data,
        request=httpx.Request("GET", "http://fake:9090/api/v1/query"),
    )


class TestControllerPrometheusMode:
    @pytest.mark.anyio
    async def test_tick_with_prometheus_client_updates_exporter(self) -> None:
        registry, exporter = make_test_exporter()
        prom_client = PrometheusClient(url="http://fake:9090")

        bundle = create_ft_controller(
            node_manager=FakeNodeManager(),
            main_job=FakeMainJob(status_sequence=[JobStatus.RUNNING]),
            metric_store=MetricStore(time_series_store=prom_client, mini_wandb=MiniWandb()),
            controller_exporter=exporter,
        )

        with patch.object(httpx.Client, "get", return_value=_make_http_response(_make_prom_response())):
            await bundle.controller._tick()

        assert get_sample_value(registry, mn.MAIN_JOB_STATUS) == 1.0
        assert get_sample_value(registry, mn.CONTROLLER_TICK_COUNT + "_total") == 1.0

    @pytest.mark.anyio
    async def test_no_scrape_target_manager_in_prometheus_mode(self) -> None:
        """In prometheus mode, register_training_rank should not fail even without scrape_target_manager."""
        _, exporter = make_test_exporter()
        prom_client = PrometheusClient(url="http://fake:9090")

        bundle = create_ft_controller(
            node_manager=FakeNodeManager(),
            main_job=FakeMainJob(),
            metric_store=MetricStore(time_series_store=prom_client, mini_wandb=MiniWandb()),
            controller_exporter=exporter,
        )

        bundle.controller._activate_run("run-1")
        bundle.subsystem_hub.training_rank_roster.register_training_rank(
            run_id="run-1",
            rank=0,
            world_size=2,
            node_id="node-0",
            exporter_address="http://node-0:9090",
            pid=1,
        )

        assert bundle.subsystem_hub.training_rank_roster.rank_placement == {0: "node-0"}

    @pytest.mark.anyio
    async def test_training_metrics_propagated_to_exporter(self) -> None:
        registry, exporter = make_test_exporter()
        mini_wandb = MiniWandb()

        bundle = create_ft_controller(
            node_manager=FakeNodeManager(),
            main_job=FakeMainJob(),
            metric_store=MetricStore(time_series_store=PrometheusClient(url="http://fake:9090"), mini_wandb=mini_wandb),
            controller_exporter=exporter,
        )

        bundle.controller._activate_run("run-1")
        bundle.subsystem_hub.training_rank_roster.register_training_rank(
            run_id="run-1",
            rank=0,
            world_size=1,
            node_id="node-0",
            exporter_address="http://node-0:9090",
            pid=1,
        )
        bundle.controller.metric_store.mini_wandb.log_step(
            run_id="run-1",
            step=1,
            metrics={"loss": 2.5, "mfu": 0.42},
        )

        with patch.object(httpx.Client, "get", return_value=_make_http_response(_make_prom_response())):
            await bundle.controller._tick()

        assert get_sample_value(registry, mn.TRAINING_LOSS_LATEST) == 2.5
        assert get_sample_value(registry, mn.TRAINING_MFU_LATEST) == 0.42
