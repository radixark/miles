"""Integration test: Controller with PrometheusClient in prometheus mode.

Verifies that the Controller can tick with PrometheusClient as the metric store
and that the ControllerExporter gauges are updated correctly.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
import pytest
from prometheus_client import CollectorRegistry

from miles.utils.ft.controller.controller import FtController
from miles.utils.ft.controller.controller_exporter import ControllerExporter
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.controller.prometheus_client_store import PrometheusClient
from miles.utils.ft.platform.protocols import JobStatus
from tests.fast.utils.ft.conftest import FakeNodeManager, FakeTrainingJob, get_sample_value


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
    @pytest.mark.asyncio
    async def test_tick_with_prometheus_client_updates_exporter(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)
        prom_client = PrometheusClient(url="http://fake:9090")

        controller = FtController(
            node_manager=FakeNodeManager(),
            training_job=FakeTrainingJob(status_sequence=[JobStatus.RUNNING]),
            metric_store=prom_client,
            mini_wandb=MiniWandb(),
            controller_exporter=exporter,
            scrape_target_manager=None,
        )

        with patch.object(httpx.Client, "get", return_value=_make_http_response(_make_prom_response())):
            await controller._tick()

        assert get_sample_value(registry, "ft_training_job_status") == 1.0
        assert get_sample_value(registry, "ft_controller_tick_count_total") == 1.0

    @pytest.mark.asyncio
    async def test_no_scrape_target_manager_in_prometheus_mode(self) -> None:
        """In prometheus mode, register_rank should not fail even without scrape_target_manager."""
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)
        prom_client = PrometheusClient(url="http://fake:9090")

        controller = FtController(
            node_manager=FakeNodeManager(),
            training_job=FakeTrainingJob(),
            metric_store=prom_client,
            mini_wandb=MiniWandb(),
            controller_exporter=exporter,
            scrape_target_manager=None,
        )

        await controller.register_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )

        assert controller._rank_placement == {0: "node-0"}

    @pytest.mark.asyncio
    async def test_training_metrics_propagated_to_exporter(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)
        mini_wandb = MiniWandb()

        controller = FtController(
            node_manager=FakeNodeManager(),
            training_job=FakeTrainingJob(),
            metric_store=PrometheusClient(url="http://fake:9090"),
            mini_wandb=mini_wandb,
            controller_exporter=exporter,
            scrape_target_manager=None,
        )

        await controller.register_rank(
            run_id="run-1", rank=0, world_size=1,
            node_id="node-0", exporter_address="http://node-0:9090",
        )
        await controller.log_step(
            run_id="run-1", rank=0, step=1,
            metrics={"loss": 2.5, "mfu": 0.42},
        )

        with patch.object(httpx.Client, "get", return_value=_make_http_response(_make_prom_response())):
            await controller._tick()

        assert get_sample_value(registry, "ft_training_loss_latest") == 2.5
        assert get_sample_value(registry, "ft_training_mfu_latest") == 0.42
