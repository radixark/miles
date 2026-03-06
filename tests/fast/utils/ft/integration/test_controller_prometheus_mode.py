"""Integration test: Controller with PrometheusClient in prometheus mode.

Verifies that the Controller can tick with PrometheusClient as the metric store
and that the ControllerExporter gauges are updated correctly.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
import pytest

import miles.utils.ft.metric_names as mn
from miles.utils.ft.controller.controller import FtController
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.metrics.prometheus_api.store import PrometheusClient
from miles.utils.ft.controller.rank_registry import RankRegistry
from miles.utils.ft.platform.protocols import JobStatus
from tests.fast.utils.ft.conftest import FakeNodeManager, FakeTrainingJob, get_sample_value, make_test_exporter


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
        registry, exporter = make_test_exporter()
        prom_client = PrometheusClient(url="http://fake:9090")

        controller = FtController.create(
            node_manager=FakeNodeManager(),
            training_job=FakeTrainingJob(status_sequence=[JobStatus.RUNNING]),
            metric_store=prom_client,
            rank_registry=RankRegistry(mini_wandb=MiniWandb()),
            controller_exporter=exporter,
        )

        with patch.object(httpx.Client, "get", return_value=_make_http_response(_make_prom_response())):
            await controller._tick()

        assert get_sample_value(registry, mn.TRAINING_JOB_STATUS) == 1.0
        assert get_sample_value(registry, mn.CONTROLLER_TICK_COUNT + "_total") == 1.0

    @pytest.mark.asyncio
    async def test_no_scrape_target_manager_in_prometheus_mode(self) -> None:
        """In prometheus mode, register_training_rank should not fail even without scrape_target_manager."""
        _, exporter = make_test_exporter()
        prom_client = PrometheusClient(url="http://fake:9090")

        controller = FtController.create(
            node_manager=FakeNodeManager(),
            training_job=FakeTrainingJob(),
            metric_store=prom_client,
            rank_registry=RankRegistry(mini_wandb=MiniWandb()),
            controller_exporter=exporter,
        )

        controller.rank_registry.register_training_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )

        assert controller._rank_registry.rank_placement == {0: "node-0"}

    @pytest.mark.asyncio
    async def test_training_metrics_propagated_to_exporter(self) -> None:
        registry, exporter = make_test_exporter()
        mini_wandb = MiniWandb()

        controller = FtController.create(
            node_manager=FakeNodeManager(),
            training_job=FakeTrainingJob(),
            metric_store=PrometheusClient(url="http://fake:9090"),
            rank_registry=RankRegistry(mini_wandb=mini_wandb),
            controller_exporter=exporter,
        )

        controller.rank_registry.register_training_rank(
            run_id="run-1", rank=0, world_size=1,
            node_id="node-0", exporter_address="http://node-0:9090",
        )
        controller.rank_registry.log_step(
            run_id="run-1", step=1,
            metrics={"loss": 2.5, "mfu": 0.42},
        )

        with patch.object(httpx.Client, "get", return_value=_make_http_response(_make_prom_response())):
            await controller._tick()

        assert get_sample_value(registry, mn.TRAINING_LOSS_LATEST) == 2.5
        assert get_sample_value(registry, mn.TRAINING_MFU_LATEST) == 0.42
