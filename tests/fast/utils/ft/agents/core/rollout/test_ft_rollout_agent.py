from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.agents.core.rollout.ft_rollout_agent import FtRolloutAgent
from tests.fast.utils.ft.agents.core.rollout.conftest import MockRolloutCellAgent
from tests.fast.utils.ft.utils.metric_injectors import get_sample_value


def _make_agent(
    cells: dict[str, MockRolloutCellAgent],
    check_interval: float = 0.05,
) -> FtRolloutAgent:
    return FtRolloutAgent(
        cells=cells,
        check_interval=check_interval,
    )


class TestHealthCheckLoopUpdatesMetrics:
    @pytest.mark.anyio
    async def test_metrics_reflect_healthy_state(self) -> None:
        cells = {
            "a0": MockRolloutCellAgent(cell_id="a0", engine_alive=[True, True]),
        }
        agent = _make_agent(cells)
        await agent.start()

        try:
            # Step 1: wait for at least one health check cycle
            await asyncio.sleep(0.15)

            # Step 2: verify metrics
            registry = agent._metrics_exporter.registry
            assert get_sample_value(registry, "rollout_cell_alive", {"cell_id": "a0"}) == 1.0
            assert get_sample_value(registry, "rollout_engine_alive", {"cell_id": "a0", "engine_index": "0"}) == 1.0
            assert get_sample_value(registry, "rollout_engine_alive", {"cell_id": "a0", "engine_index": "1"}) == 1.0
        finally:
            await agent.shutdown()


class TestPartialEngineDeathMetrics:
    @pytest.mark.anyio
    async def test_dead_engine_reflected_in_metrics(self) -> None:
        cells = {
            "a0": MockRolloutCellAgent(cell_id="a0", engine_alive=[True, False, True]),
        }
        agent = _make_agent(cells)
        await agent.start()

        try:
            await asyncio.sleep(0.15)

            registry = agent._metrics_exporter.registry
            assert get_sample_value(registry, "rollout_cell_alive", {"cell_id": "a0"}) == 0.0
            assert get_sample_value(registry, "rollout_engine_alive", {"cell_id": "a0", "engine_index": "0"}) == 1.0
            assert get_sample_value(registry, "rollout_engine_alive", {"cell_id": "a0", "engine_index": "1"}) == 0.0
            assert get_sample_value(registry, "rollout_engine_alive", {"cell_id": "a0", "engine_index": "2"}) == 1.0
        finally:
            await agent.shutdown()


class TestPauseDoesNotProduceFalseUpdates:
    @pytest.mark.anyio
    async def test_pause_preserves_last_metrics(self) -> None:
        cell = MockRolloutCellAgent(cell_id="a0", engine_alive=[True, True])
        agent = _make_agent({"a0": cell})
        await agent.start()

        try:
            # Step 1: wait for initial healthy check
            await asyncio.sleep(0.15)
            registry = agent._metrics_exporter.registry
            assert get_sample_value(registry, "rollout_cell_alive", {"cell_id": "a0"}) == 1.0

            # Step 2: pause, then kill an engine
            agent.pause()
            cell._engine_alive[1] = False

            # Step 3: wait another cycle — metrics should NOT update
            await asyncio.sleep(0.15)
            assert get_sample_value(registry, "rollout_cell_alive", {"cell_id": "a0"}) == 1.0
        finally:
            await agent.shutdown()


class TestResumeRestoresChecks:
    @pytest.mark.anyio
    async def test_resume_updates_metrics_after_pause(self) -> None:
        cell = MockRolloutCellAgent(cell_id="a0", engine_alive=[True, True])
        agent = _make_agent({"a0": cell})
        await agent.start()

        try:
            await asyncio.sleep(0.15)

            # Step 1: pause and kill engine
            agent.pause()
            cell._engine_alive[0] = False
            await asyncio.sleep(0.15)

            # Step 2: resume — metrics should update
            agent.resume()
            await asyncio.sleep(0.15)

            registry = agent._metrics_exporter.registry
            assert get_sample_value(registry, "rollout_cell_alive", {"cell_id": "a0"}) == 0.0
        finally:
            await agent.shutdown()


class TestGetStatus:
    @pytest.mark.anyio
    async def test_all_healthy_returns_running(self) -> None:
        cells = {
            "a0": MockRolloutCellAgent(cell_id="a0", engine_alive=[True, True]),
            "a1": MockRolloutCellAgent(cell_id="a1", engine_alive=[True]),
        }
        agent = _make_agent(cells)
        await agent.start()

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
        agent = _make_agent(cells)
        await agent.start()

        try:
            await asyncio.sleep(0.15)
            assert await agent.get_status() == JobStatus.FAILED
        finally:
            await agent.shutdown()


class TestGetCellStatusPerCellIsolation:
    @pytest.mark.anyio
    async def test_per_cell_status_is_independent(self) -> None:
        cells = {
            "a0": MockRolloutCellAgent(cell_id="a0", engine_alive=[True, True]),
            "a1": MockRolloutCellAgent(cell_id="a1", engine_alive=[True, False]),
        }
        agent = _make_agent(cells)
        await agent.start()

        try:
            await asyncio.sleep(0.15)
            assert await agent.get_cell_status("a0") == JobStatus.RUNNING
            assert await agent.get_cell_status("a1") == JobStatus.FAILED
        finally:
            await agent.shutdown()


class TestLifecycle:
    @pytest.mark.anyio
    async def test_start_makes_address_available_and_shutdown_stops_loop(self) -> None:
        agent = _make_agent({"a0": MockRolloutCellAgent(cell_id="a0", engine_alive=[True])})
        await agent.start()

        try:
            assert "http://localhost:" in agent.address
        finally:
            await agent.shutdown()

        assert agent._health_loop_task is not None
        assert agent._health_loop_task.done()


class TestRegisterWithController:
    @pytest.mark.anyio
    async def test_calls_add_scrape_target_with_correct_address(self) -> None:
        agent = _make_agent({"a0": MockRolloutCellAgent(cell_id="a0", engine_alive=[True])})

        try:
            controller_handle = MagicMock()
            controller_handle.add_scrape_target = MagicMock()
            controller_handle.add_scrape_target.remote = AsyncMock()

            await agent.register_with_controller(controller_handle)

            controller_handle.add_scrape_target.remote.assert_called_once_with(
                target_id="rollout-ft-agent",
                address=agent.address,
            )
        finally:
            agent._metrics_exporter.shutdown()


class TestHealthLoopSurvivesException:
    @pytest.mark.anyio
    async def test_exception_in_check_health_does_not_crash_loop(self) -> None:
        """If one cell's check_health raises, the loop continues checking other cells."""
        healthy_cell = MockRolloutCellAgent(cell_id="healthy", engine_alive=[True])
        broken_cell = _BrokenCheckHealthCell(cell_id="broken")
        agent = _make_agent({"healthy": healthy_cell, "broken": broken_cell})
        await agent.start()

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
    async def test_delegates_to_cell(self) -> None:
        cell = MockRolloutCellAgent(cell_id="a0", engine_alive=[True, True])
        agent = FtRolloutAgent(cells={"a0": cell}, check_interval=10.0)

        try:
            await agent.stop_cell("a0")
        finally:
            agent._metrics_exporter.shutdown()

    @pytest.mark.anyio
    async def test_raises_key_error_for_unknown_cell(self) -> None:
        agent = FtRolloutAgent(
            cells={"a0": MockRolloutCellAgent(cell_id="a0", engine_alive=[True])},
            check_interval=10.0,
        )

        try:
            with pytest.raises(KeyError):
                await agent.stop_cell("nonexistent")
        finally:
            agent._metrics_exporter.shutdown()


class TestStartCell:
    @pytest.mark.anyio
    async def test_delegates_to_cell_and_returns_count(self) -> None:
        cell = MockRolloutCellAgent(cell_id="a0", engine_alive=[True, False, True])
        agent = FtRolloutAgent(cells={"a0": cell}, check_interval=10.0)

        try:
            result = await agent.start_cell("a0")
            assert result == 2
        finally:
            agent._metrics_exporter.shutdown()

    @pytest.mark.anyio
    async def test_raises_key_error_for_unknown_cell(self) -> None:
        agent = FtRolloutAgent(
            cells={"a0": MockRolloutCellAgent(cell_id="a0", engine_alive=[True])},
            check_interval=10.0,
        )

        try:
            with pytest.raises(KeyError):
                await agent.start_cell("nonexistent")
        finally:
            agent._metrics_exporter.shutdown()


class TestStopAll:
    @pytest.mark.anyio
    async def test_calls_stop_on_every_cell(self) -> None:
        cells = {
            "a0": _TrackingStopCell(cell_id="a0", engine_alive=[True]),
            "a1": _TrackingStopCell(cell_id="a1", engine_alive=[True, True]),
        }
        agent = FtRolloutAgent(cells=cells, check_interval=10.0)

        try:
            await agent.stop_all()
            assert cells["a0"].stop_called
            assert cells["a1"].stop_called
        finally:
            agent._metrics_exporter.shutdown()


class _TrackingStopCell(MockRolloutCellAgent):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.stop_called = False

    async def stop(self) -> None:
        self.stop_called = True


class TestRebuild:
    @pytest.mark.anyio
    async def test_raises_not_implemented(self) -> None:
        agent = FtRolloutAgent(
            cells={"a0": MockRolloutCellAgent(cell_id="a0", engine_alive=[True])},
            check_interval=10.0,
        )

        try:
            with pytest.raises(NotImplementedError):
                await agent.rebuild()
        finally:
            agent._metrics_exporter.shutdown()


class TestGetStatusBeforeAnyHealthCheck:
    @pytest.mark.anyio
    async def test_returns_failed_when_no_check_has_run(self) -> None:
        """Before any health check, is_healthy() returns False, so get_status should be FAILED."""
        agent = FtRolloutAgent(
            cells={"a0": MockRolloutCellAgent(cell_id="a0", engine_alive=[True, True])},
            check_interval=10.0,
        )

        try:
            assert await agent.get_status() == JobStatus.FAILED
        finally:
            agent._metrics_exporter.shutdown()


class TestMultiCellMetricsAreIndependent:
    @pytest.mark.anyio
    async def test_per_cell_metrics_reflect_individual_health(self) -> None:
        cells = {
            "a0": MockRolloutCellAgent(cell_id="a0", engine_alive=[True, True]),
            "a1": MockRolloutCellAgent(cell_id="a1", engine_alive=[True, False]),
        }
        agent = _make_agent(cells)
        await agent.start()

        try:
            await asyncio.sleep(0.15)

            registry = agent._metrics_exporter.registry
            assert get_sample_value(registry, "rollout_cell_alive", {"cell_id": "a0"}) == 1.0
            assert get_sample_value(registry, "rollout_cell_alive", {"cell_id": "a1"}) == 0.0
            assert get_sample_value(registry, "rollout_engine_alive", {"cell_id": "a1", "engine_index": "0"}) == 1.0
            assert get_sample_value(registry, "rollout_engine_alive", {"cell_id": "a1", "engine_index": "1"}) == 0.0
        finally:
            await agent.shutdown()


class TestShutdownWithoutStart:
    @pytest.mark.anyio
    async def test_shutdown_without_start_is_safe(self) -> None:
        agent = FtRolloutAgent(
            cells={"a0": MockRolloutCellAgent(cell_id="a0", engine_alive=[True])},
            check_interval=10.0,
        )

        await agent.shutdown()


class TestUpdateCellEngines:
    def test_delegates_to_cell_update_engines(self) -> None:
        cell = MockRolloutCellAgent(cell_id="a0", engine_alive=[True, True])
        agent = FtRolloutAgent(cells={"a0": cell}, check_interval=10.0)

        try:
            agent.update_cell_engines("a0", ["new0", "new1", "new2"])

            assert cell.get_engine_count() == 3
            assert cell.is_healthy() is False
        finally:
            agent._metrics_exporter.shutdown()

    def test_raises_key_error_for_unknown_cell(self) -> None:
        agent = FtRolloutAgent(
            cells={"a0": MockRolloutCellAgent(cell_id="a0", engine_alive=[True])},
            check_interval=10.0,
        )

        try:
            with pytest.raises(KeyError):
                agent.update_cell_engines("nonexistent", ["e0"])
        finally:
            agent._metrics_exporter.shutdown()


class TestPublicApi:
    def test_get_cell_ids(self) -> None:
        cells = {
            "a0": MockRolloutCellAgent(cell_id="a0", engine_alive=[True]),
            "a1": MockRolloutCellAgent(cell_id="a1", engine_alive=[True, True]),
        }
        agent = FtRolloutAgent(cells=cells, check_interval=10.0)

        try:
            assert sorted(agent.get_cell_ids()) == ["a0", "a1"]
        finally:
            agent._metrics_exporter.shutdown()

    def test_get_cell_agent(self) -> None:
        cell = MockRolloutCellAgent(cell_id="a0", engine_alive=[True])
        agent = FtRolloutAgent(cells={"a0": cell}, check_interval=10.0)

        try:
            assert agent.get_cell_agent("a0") is cell
        finally:
            agent._metrics_exporter.shutdown()
