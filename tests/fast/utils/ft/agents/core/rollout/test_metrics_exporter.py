from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from miles.utils.ft.agents.core.rollout.cell_agent import CellHealthResult
from miles.utils.ft.agents.core.rollout.metrics_exporter import RolloutMetricsExporter
from tests.fast.utils.ft.utils.metric_injectors import get_sample_value


class TestUpdate:
    def test_healthy_result_sets_all_gauges_to_one(self) -> None:
        exporter = RolloutMetricsExporter()

        try:
            result = CellHealthResult(
                cell_id="a0",
                total_engines=3,
                alive_engines=3,
                dead_engine_indices=(),
                is_healthy=True,
                checked_at=0.0,
            )
            exporter.update(result)

            assert get_sample_value(exporter.registry, "rollout_cell_alive", {"cell_id": "a0"}) == 1.0
            for i in range(3):
                assert get_sample_value(exporter.registry, "rollout_engine_alive", {"cell_id": "a0", "engine_index": str(i)}) == 1.0
        finally:
            exporter.shutdown()

    def test_partial_death_sets_correct_gauges(self) -> None:
        exporter = RolloutMetricsExporter()

        try:
            result = CellHealthResult(
                cell_id="a0",
                total_engines=3,
                alive_engines=2,
                dead_engine_indices=(1,),
                is_healthy=False,
                checked_at=0.0,
            )
            exporter.update(result)

            assert get_sample_value(exporter.registry, "rollout_cell_alive", {"cell_id": "a0"}) == 0.0
            assert get_sample_value(exporter.registry, "rollout_engine_alive", {"cell_id": "a0", "engine_index": "0"}) == 1.0
            assert get_sample_value(exporter.registry, "rollout_engine_alive", {"cell_id": "a0", "engine_index": "1"}) == 0.0
            assert get_sample_value(exporter.registry, "rollout_engine_alive", {"cell_id": "a0", "engine_index": "2"}) == 1.0
        finally:
            exporter.shutdown()


class TestAddress:
    def test_address_available_immediately(self) -> None:
        exporter = RolloutMetricsExporter()

        try:
            assert "http://localhost:" in exporter.address
            port = int(exporter.address.split(":")[-1])
            assert port > 0
        finally:
            exporter.shutdown()


class TestShutdown:
    def test_shutdown_is_safe(self) -> None:
        exporter = RolloutMetricsExporter()
        exporter.shutdown()


class TestRegisterWithController:
    @pytest.mark.anyio
    async def test_calls_add_scrape_target_with_correct_address(self) -> None:
        exporter = RolloutMetricsExporter()

        try:
            controller_handle = MagicMock()
            controller_handle.add_scrape_target = MagicMock()
            controller_handle.add_scrape_target.remote = AsyncMock()

            await exporter.register_with_controller(controller_handle)

            controller_handle.add_scrape_target.remote.assert_called_once_with(
                target_id="rollout-ft-agent",
                address=exporter.address,
            )
        finally:
            exporter.shutdown()
