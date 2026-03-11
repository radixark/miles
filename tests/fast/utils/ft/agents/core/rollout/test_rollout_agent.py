from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from miles.utils.ft.adapters.types import JobStatus
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


class TestGetStatus:
    @pytest.mark.anyio
    async def test_all_healthy_returns_running(self) -> None:
        cells = {
            "a0": MockRolloutCellAgent(cell_id="a0", engine_alive=[True, True]),
            "a1": MockRolloutCellAgent(cell_id="a1", engine_alive=[True]),
        }
        agent = _make_multicell_agent(cells)

        try:
            await asyncio.sleep(0.15)
            assert await agent.get_status() == JobStatus.RUNNING
        finally:
            await agent.shutdown()

    @pytest.mark.anyio
    async def test_any_dead_returns_failed(self) -> None:
        cells = {
            "a0": MockRolloutCellAgent(cell_id="a0", engine_alive=[True, True]),
            "a1": MockRolloutCellAgent(cell_id="a1", engine_alive=[True, False]),
        }
        agent = _make_multicell_agent(cells)

        try:
            await asyncio.sleep(0.15)
            assert await agent.get_status() == JobStatus.FAILED
        finally:
            await agent.shutdown()


class TestGetCellStatus:
    @pytest.mark.anyio
    async def test_delegates_to_rollout_manager(self) -> None:
        rm = AsyncMock()
        rm.all_rollout_engines = []
        rm.get_cell_status.side_effect = [JobStatus.RUNNING, JobStatus.FAILED]
        with patch.object(FtRolloutAgent, '_build_cells', return_value={}):
            agent = FtRolloutAgent(rm, health_checker=AsyncMock(), check_interval=10.0)

        try:
            assert await agent.get_cell_status("a0") == JobStatus.RUNNING
            assert await agent.get_cell_status("a1") == JobStatus.FAILED
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


class TestStopCell:
    @pytest.mark.anyio
    async def test_delegates_to_rollout_manager(self) -> None:
        rm = AsyncMock()
        rm.all_rollout_engines = [MockEngine(True)]
        agent = FtRolloutAgent(rm, health_checker=mock_health_checker, check_interval=10.0)

        try:
            await agent.stop_cell("a0")
            rm.stop_cell.assert_awaited_once_with("a0")
        finally:
            await agent.shutdown()


class TestStartCell:
    @pytest.mark.anyio
    async def test_delegates_to_rollout_manager(self) -> None:
        rm = AsyncMock()
        rm.all_rollout_engines = [MockEngine(True)]
        rm.start_cell.return_value = 2
        agent = FtRolloutAgent(rm, health_checker=mock_health_checker, check_interval=10.0)

        try:
            result = await agent.start_cell("a0")
            assert result == 2
            rm.start_cell.assert_awaited_once_with("a0")
        finally:
            await agent.shutdown()


class TestStopAll:
    @pytest.mark.anyio
    async def test_calls_stop_on_every_cell(self) -> None:
        cells = {
            "a0": _TrackingStopCell(cell_id="a0", engine_alive=[True]),
            "a1": _TrackingStopCell(cell_id="a1", engine_alive=[True, True]),
        }
        agent = _make_multicell_agent(cells, check_interval=10.0)

        try:
            await agent.stop_all()
            assert cells["a0"].stop_called
            assert cells["a1"].stop_called
        finally:
            await agent.shutdown()


class _TrackingStopCell(MockRolloutCellAgent):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.stop_called = False

    async def stop(self) -> None:
        self.stop_called = True


class TestRebuild:
    @pytest.mark.anyio
    async def test_raises_not_implemented(self) -> None:
        agent, _ = _make_agent([True], check_interval=10.0)

        try:
            with pytest.raises(NotImplementedError):
                await agent.rebuild()
        finally:
            await agent.shutdown()


class TestGetStatusBeforeAnyHealthCheck:
    @pytest.mark.anyio
    async def test_returns_failed_when_no_check_has_run(self) -> None:
        """Before any health check, is_healthy() returns False, so get_status should be FAILED."""
        agent, _ = _make_agent([True, True], check_interval=10.0)

        try:
            assert await agent.get_status() == JobStatus.FAILED
        finally:
            await agent.shutdown()


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


class TestUpdateCellEngines:
    @pytest.mark.anyio
    async def test_delegates_to_cell_update_engines(self) -> None:
        agent, _ = _make_agent([True, True], check_interval=10.0)

        try:
            agent.update_cell_engines("default", ["new0", "new1", "new2"])

            cell = agent.get_cell_agent("default")
            assert cell.get_engine_count() == 3
            assert cell.is_healthy() is False
        finally:
            await agent.shutdown()

    @pytest.mark.anyio
    async def test_raises_key_error_for_unknown_cell(self) -> None:
        agent, _ = _make_agent([True], check_interval=10.0)

        try:
            with pytest.raises(KeyError):
                agent.update_cell_engines("nonexistent", ["e0"])
        finally:
            await agent.shutdown()


class TestPublicApi:
    @pytest.mark.anyio
    async def test_get_cell_ids(self) -> None:
        cells = {
            "a0": MockRolloutCellAgent(cell_id="a0", engine_alive=[True]),
            "a1": MockRolloutCellAgent(cell_id="a1", engine_alive=[True, True]),
        }
        agent = _make_multicell_agent(cells, check_interval=10.0)

        try:
            assert sorted(agent.get_cell_ids()) == ["a0", "a1"]
        finally:
            await agent.shutdown()

    @pytest.mark.anyio
    async def test_get_cell_agent(self) -> None:
        cell = MockRolloutCellAgent(cell_id="a0", engine_alive=[True])
        agent = _make_multicell_agent({"a0": cell}, check_interval=10.0)

        try:
            assert agent.get_cell_agent("a0") is cell
        finally:
            await agent.shutdown()


class TestBuildCells:
    def test_builds_default_cell_from_rollout_manager(self) -> None:
        rollout_manager = MagicMock()
        rollout_manager.all_rollout_engines = [MagicMock(), MagicMock(), MagicMock()]

        cells = FtRolloutAgent._build_cells(rollout_manager, health_checker=AsyncMock())

        assert list(cells.keys()) == ["default"]
        assert cells["default"].get_engine_count() == 3

    def test_cell_id_is_default(self) -> None:
        rollout_manager = MagicMock()
        rollout_manager.all_rollout_engines = [MagicMock()]

        cells = FtRolloutAgent._build_cells(rollout_manager, health_checker=AsyncMock())
        assert cells["default"].cell_id == "default"
