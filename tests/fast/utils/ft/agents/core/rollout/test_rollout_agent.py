from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from miles.utils.ft.agents.core.rollout.rollout_agent import FtRolloutAgent
from tests.fast.utils.ft.agents.core.rollout.conftest import (
    MockEngine,
    MockRolloutCellAgent,
    mock_health_checker,
)
from tests.fast.utils.ft.utils.metric_injectors import get_sample_value


def _make_agent(
    engine_alive: list[bool],
    check_interval: float = 0.05,
) -> tuple[FtRolloutAgent, list[MockEngine]]:
    engines = [MockEngine(a) for a in engine_alive]
    rm = MagicMock()
    rm.all_rollout_engines = engines
    return FtRolloutAgent(rm, health_checker=mock_health_checker, check_interval=check_interval), engines


def _make_multicell_agent(
    cells: dict[str, MockRolloutCellAgent],
    check_interval: float = 0.05,
) -> FtRolloutAgent:
    with patch.object(FtRolloutAgent, '_build_cells', return_value=cells):
        return FtRolloutAgent(MagicMock(), health_checker=AsyncMock(), check_interval=check_interval)


class TestHealthCheckLoopUpdatesMetrics:
    @pytest.mark.anyio
    async def test_metrics_reflect_healthy_state(self) -> None:
        agent, _ = _make_agent([True, True])

        try:
            # Step 1: wait for at least one health check cycle
            await asyncio.sleep(0.15)

            # Step 2: verify metrics
            registry = agent._metrics_exporter.registry
            assert get_sample_value(registry, "rollout_cell_alive", {"cell_id": "default"}) == 1.0
            assert get_sample_value(registry, "rollout_engine_alive", {"cell_id": "default", "engine_index": "0"}) == 1.0
            assert get_sample_value(registry, "rollout_engine_alive", {"cell_id": "default", "engine_index": "1"}) == 1.0
        finally:
            await agent.shutdown()


class TestPartialEngineDeathMetrics:
    @pytest.mark.anyio
    async def test_dead_engine_reflected_in_metrics(self) -> None:
        agent, _ = _make_agent([True, False, True])

        try:
            await asyncio.sleep(0.15)

            registry = agent._metrics_exporter.registry
            assert get_sample_value(registry, "rollout_cell_alive", {"cell_id": "default"}) == 0.0
            assert get_sample_value(registry, "rollout_engine_alive", {"cell_id": "default", "engine_index": "0"}) == 1.0
            assert get_sample_value(registry, "rollout_engine_alive", {"cell_id": "default", "engine_index": "1"}) == 0.0
            assert get_sample_value(registry, "rollout_engine_alive", {"cell_id": "default", "engine_index": "2"}) == 1.0
        finally:
            await agent.shutdown()


class TestPauseDoesNotProduceFalseUpdates:
    @pytest.mark.anyio
    async def test_pause_preserves_last_metrics(self) -> None:
        agent, engines = _make_agent([True, True])

        try:
            # Step 1: wait for initial healthy check
            await asyncio.sleep(0.15)
            registry = agent._metrics_exporter.registry
            assert get_sample_value(registry, "rollout_cell_alive", {"cell_id": "default"}) == 1.0

            # Step 2: pause, then kill an engine
            agent.pause()
            engines[1].alive = False

            # Step 3: wait another cycle — metrics should NOT update
            await asyncio.sleep(0.15)
            assert get_sample_value(registry, "rollout_cell_alive", {"cell_id": "default"}) == 1.0
        finally:
            await agent.shutdown()


class TestResumeRestoresChecks:
    @pytest.mark.anyio
    async def test_resume_updates_metrics_after_pause(self) -> None:
        agent, engines = _make_agent([True, True])

        try:
            await asyncio.sleep(0.15)

            # Step 1: pause and kill engine
            agent.pause()
            engines[0].alive = False
            await asyncio.sleep(0.15)

            # Step 2: resume — metrics should update
            agent.resume()
            await asyncio.sleep(0.15)

            registry = agent._metrics_exporter.registry
            assert get_sample_value(registry, "rollout_cell_alive", {"cell_id": "default"}) == 0.0
        finally:
            await agent.shutdown()


class TestLifecycle:
    @pytest.mark.anyio
    async def test_address_available_immediately_and_shutdown_stops_loop(self) -> None:
        agent, _ = _make_agent([True])

        try:
            assert "http://localhost:" in agent.address
        finally:
            await agent.shutdown()

        assert agent._health_loop_task is not None
        assert agent._health_loop_task.done()


class TestHealthLoopSurvivesException:
    @pytest.mark.anyio
    async def test_exception_in_check_health_does_not_crash_loop(self) -> None:
        """If one cell's check_health raises, the loop continues checking other cells."""
        healthy_cell = MockRolloutCellAgent(cell_id="healthy", engine_alive=[True])
        broken_cell = _BrokenCheckHealthCell(cell_id="broken")
        agent = _make_multicell_agent({"healthy": healthy_cell, "broken": broken_cell})

        try:
            # Step 1: wait for multiple cycles
            await asyncio.sleep(0.15)

            # Step 2: healthy cell's metrics should still be updated
            registry = agent._metrics_exporter.registry
            assert get_sample_value(registry, "rollout_cell_alive", {"cell_id": "healthy"}) == 1.0

            # Step 3: broken cell should have no metrics (never successfully checked)
            assert get_sample_value(registry, "rollout_cell_alive", {"cell_id": "broken"}) is None

            # Step 4: loop is still running
            assert not agent._health_loop_task.done()
        finally:
            await agent.shutdown()


class _BrokenCheckHealthCell(MockRolloutCellAgent):
    def __init__(self, *, cell_id: str) -> None:
        super().__init__(cell_id=cell_id, engine_alive=[True])

    async def check_health(self) -> object:
        raise RuntimeError("simulated check_health failure")


class TestMultiCellMetricsAreIndependent:
    @pytest.mark.anyio
    async def test_per_cell_metrics_reflect_individual_health(self) -> None:
        cells = {
            "a0": MockRolloutCellAgent(cell_id="a0", engine_alive=[True, True]),
            "a1": MockRolloutCellAgent(cell_id="a1", engine_alive=[True, False]),
        }
        agent = _make_multicell_agent(cells)

        try:
            await asyncio.sleep(0.15)

            registry = agent._metrics_exporter.registry
            assert get_sample_value(registry, "rollout_cell_alive", {"cell_id": "a0"}) == 1.0
            assert get_sample_value(registry, "rollout_cell_alive", {"cell_id": "a1"}) == 0.0
            assert get_sample_value(registry, "rollout_engine_alive", {"cell_id": "a1", "engine_index": "0"}) == 1.0
            assert get_sample_value(registry, "rollout_engine_alive", {"cell_id": "a1", "engine_index": "1"}) == 0.0
        finally:
            await agent.shutdown()


class TestShutdownImmediately:
    @pytest.mark.anyio
    async def test_shutdown_immediately_after_construction_is_safe(self) -> None:
        agent, _ = _make_agent([True], check_interval=10.0)

        await agent.shutdown()


class TestBuildCells:
    def test_builds_default_cell_from_rollout_manager(self) -> None:
        rollout_manager = MagicMock()
        rollout_manager.all_rollout_engines = [MagicMock(), MagicMock(), MagicMock()]

        cells = FtRolloutAgent._build_cells(rollout_manager, health_checker=AsyncMock())

        assert list(cells.keys()) == ["default"]
        assert len(cells["default"]._engines) == 3

    def test_cell_id_is_default(self) -> None:
        rollout_manager = MagicMock()
        rollout_manager.all_rollout_engines = [MagicMock()]

        cells = FtRolloutAgent._build_cells(rollout_manager, health_checker=AsyncMock())
        assert cells["default"].cell_id == "default"
