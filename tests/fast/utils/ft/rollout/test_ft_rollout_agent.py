from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.rollout.ft_rollout_agent import FtRolloutAgent
from tests.fast.utils.ft.rollout.conftest import MockRolloutAtomAgent
from tests.fast.utils.ft.utils.metric_injectors import get_sample_value


def _make_agent(
    atoms: dict[str, MockRolloutAtomAgent],
    check_interval: float = 0.05,
) -> FtRolloutAgent:
    return FtRolloutAgent(
        atoms=atoms,
        check_interval=check_interval,
        metrics_port=0,
    )


class TestHealthCheckLoopUpdatesMetrics:
    @pytest.mark.anyio
    async def test_metrics_reflect_healthy_state(self) -> None:
        atoms = {
            "a0": MockRolloutAtomAgent(atom_id="a0", engine_alive=[True, True]),
        }
        agent = _make_agent(atoms)
        await agent.start()

        try:
            # Step 1: wait for at least one health check cycle
            await asyncio.sleep(0.15)

            # Step 2: verify metrics
            assert get_sample_value(agent._registry, "rollout_atom_alive", {"atom_id": "a0"}) == 1.0
            assert get_sample_value(agent._registry, "rollout_engine_alive", {"atom_id": "a0", "engine_index": "0"}) == 1.0
            assert get_sample_value(agent._registry, "rollout_engine_alive", {"atom_id": "a0", "engine_index": "1"}) == 1.0
        finally:
            await agent.shutdown()


class TestPartialEngineDeathMetrics:
    @pytest.mark.anyio
    async def test_dead_engine_reflected_in_metrics(self) -> None:
        atoms = {
            "a0": MockRolloutAtomAgent(atom_id="a0", engine_alive=[True, False, True]),
        }
        agent = _make_agent(atoms)
        await agent.start()

        try:
            await asyncio.sleep(0.15)

            assert get_sample_value(agent._registry, "rollout_atom_alive", {"atom_id": "a0"}) == 0.0
            assert get_sample_value(agent._registry, "rollout_engine_alive", {"atom_id": "a0", "engine_index": "0"}) == 1.0
            assert get_sample_value(agent._registry, "rollout_engine_alive", {"atom_id": "a0", "engine_index": "1"}) == 0.0
            assert get_sample_value(agent._registry, "rollout_engine_alive", {"atom_id": "a0", "engine_index": "2"}) == 1.0
        finally:
            await agent.shutdown()


class TestPauseDoesNotProduceFalseUpdates:
    @pytest.mark.anyio
    async def test_pause_preserves_last_metrics(self) -> None:
        atom = MockRolloutAtomAgent(atom_id="a0", engine_alive=[True, True])
        agent = _make_agent({"a0": atom})
        await agent.start()

        try:
            # Step 1: wait for initial healthy check
            await asyncio.sleep(0.15)
            assert get_sample_value(agent._registry, "rollout_atom_alive", {"atom_id": "a0"}) == 1.0

            # Step 2: pause, then kill an engine
            agent.pause()
            atom._engine_alive[1] = False

            # Step 3: wait another cycle — metrics should NOT update
            await asyncio.sleep(0.15)
            assert get_sample_value(agent._registry, "rollout_atom_alive", {"atom_id": "a0"}) == 1.0
        finally:
            await agent.shutdown()


class TestResumeRestoresChecks:
    @pytest.mark.anyio
    async def test_resume_updates_metrics_after_pause(self) -> None:
        atom = MockRolloutAtomAgent(atom_id="a0", engine_alive=[True, True])
        agent = _make_agent({"a0": atom})
        await agent.start()

        try:
            await asyncio.sleep(0.15)

            # Step 1: pause and kill engine
            agent.pause()
            atom._engine_alive[0] = False
            await asyncio.sleep(0.15)

            # Step 2: resume — metrics should update
            agent.resume()
            await asyncio.sleep(0.15)

            assert get_sample_value(agent._registry, "rollout_atom_alive", {"atom_id": "a0"}) == 0.0
        finally:
            await agent.shutdown()


class TestGetStatus:
    @pytest.mark.anyio
    async def test_all_healthy_returns_running(self) -> None:
        atoms = {
            "a0": MockRolloutAtomAgent(atom_id="a0", engine_alive=[True, True]),
            "a1": MockRolloutAtomAgent(atom_id="a1", engine_alive=[True]),
        }
        agent = _make_agent(atoms)
        await agent.start()

        try:
            await asyncio.sleep(0.15)
            assert await agent.get_status() == JobStatus.RUNNING
        finally:
            await agent.shutdown()

    @pytest.mark.anyio
    async def test_any_dead_returns_failed(self) -> None:
        atoms = {
            "a0": MockRolloutAtomAgent(atom_id="a0", engine_alive=[True, True]),
            "a1": MockRolloutAtomAgent(atom_id="a1", engine_alive=[True, False]),
        }
        agent = _make_agent(atoms)
        await agent.start()

        try:
            await asyncio.sleep(0.15)
            assert await agent.get_status() == JobStatus.FAILED
        finally:
            await agent.shutdown()


class TestGetAtomStatusPerAtomIsolation:
    @pytest.mark.anyio
    async def test_per_atom_status_is_independent(self) -> None:
        atoms = {
            "a0": MockRolloutAtomAgent(atom_id="a0", engine_alive=[True, True]),
            "a1": MockRolloutAtomAgent(atom_id="a1", engine_alive=[True, False]),
        }
        agent = _make_agent(atoms)
        await agent.start()

        try:
            await asyncio.sleep(0.15)
            assert await agent.get_atom_status("a0") == JobStatus.RUNNING
            assert await agent.get_atom_status("a1") == JobStatus.FAILED
        finally:
            await agent.shutdown()


class TestLifecycle:
    @pytest.mark.anyio
    async def test_start_makes_address_available_and_shutdown_stops_loop(self) -> None:
        agent = _make_agent({"a0": MockRolloutAtomAgent(atom_id="a0", engine_alive=[True])})
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
        agent = _make_agent({"a0": MockRolloutAtomAgent(atom_id="a0", engine_alive=[True])})
        await agent.start()

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
            await agent.shutdown()


class TestHealthLoopSurvivesException:
    @pytest.mark.anyio
    async def test_exception_in_check_health_does_not_crash_loop(self) -> None:
        """If one atom's check_health raises, the loop continues checking other atoms."""
        healthy_atom = MockRolloutAtomAgent(atom_id="healthy", engine_alive=[True])
        broken_atom = _BrokenCheckHealthAtom(atom_id="broken")
        agent = _make_agent({"healthy": healthy_atom, "broken": broken_atom})
        await agent.start()

        try:
            # Step 1: wait for multiple cycles
            await asyncio.sleep(0.15)

            # Step 2: healthy atom's metrics should still be updated
            assert get_sample_value(agent._registry, "rollout_atom_alive", {"atom_id": "healthy"}) == 1.0

            # Step 3: broken atom should have no metrics (never successfully checked)
            assert get_sample_value(agent._registry, "rollout_atom_alive", {"atom_id": "broken"}) is None

            # Step 4: loop is still running
            assert not agent._health_loop_task.done()
        finally:
            await agent.shutdown()


class _BrokenCheckHealthAtom(MockRolloutAtomAgent):
    def __init__(self, *, atom_id: str) -> None:
        super().__init__(atom_id=atom_id, engine_alive=[True])

    async def check_health(self) -> object:
        raise RuntimeError("simulated check_health failure")


class TestStopAtom:
    @pytest.mark.anyio
    async def test_delegates_to_atom(self) -> None:
        atom = MockRolloutAtomAgent(atom_id="a0", engine_alive=[True, True])
        agent = FtRolloutAgent(atoms={"a0": atom}, check_interval=10.0, metrics_port=0)

        await agent.stop_atom("a0")

    @pytest.mark.anyio
    async def test_raises_key_error_for_unknown_atom(self) -> None:
        agent = FtRolloutAgent(
            atoms={"a0": MockRolloutAtomAgent(atom_id="a0", engine_alive=[True])},
            check_interval=10.0,
            metrics_port=0,
        )

        with pytest.raises(KeyError):
            await agent.stop_atom("nonexistent")


class TestStartAtom:
    @pytest.mark.anyio
    async def test_delegates_to_atom_and_returns_count(self) -> None:
        atom = MockRolloutAtomAgent(atom_id="a0", engine_alive=[True, False, True])
        agent = FtRolloutAgent(atoms={"a0": atom}, check_interval=10.0, metrics_port=0)

        result = await agent.start_atom("a0")

        assert result == 2

    @pytest.mark.anyio
    async def test_raises_key_error_for_unknown_atom(self) -> None:
        agent = FtRolloutAgent(
            atoms={"a0": MockRolloutAtomAgent(atom_id="a0", engine_alive=[True])},
            check_interval=10.0,
            metrics_port=0,
        )

        with pytest.raises(KeyError):
            await agent.start_atom("nonexistent")


class TestStopAll:
    @pytest.mark.anyio
    async def test_calls_stop_on_every_atom(self) -> None:
        atoms = {
            "a0": _TrackingStopAtom(atom_id="a0", engine_alive=[True]),
            "a1": _TrackingStopAtom(atom_id="a1", engine_alive=[True, True]),
        }
        agent = FtRolloutAgent(atoms=atoms, check_interval=10.0, metrics_port=0)

        await agent.stop_all()

        assert atoms["a0"].stop_called
        assert atoms["a1"].stop_called


class _TrackingStopAtom(MockRolloutAtomAgent):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.stop_called = False

    async def stop(self) -> None:
        self.stop_called = True


class TestRebuild:
    @pytest.mark.anyio
    async def test_raises_not_implemented(self) -> None:
        agent = FtRolloutAgent(
            atoms={"a0": MockRolloutAtomAgent(atom_id="a0", engine_alive=[True])},
            check_interval=10.0,
            metrics_port=0,
        )

        with pytest.raises(NotImplementedError):
            await agent.rebuild()


class TestGetStatusBeforeAnyHealthCheck:
    @pytest.mark.anyio
    async def test_returns_failed_when_no_check_has_run(self) -> None:
        """Before any health check, is_healthy() returns False, so get_status should be FAILED."""
        agent = FtRolloutAgent(
            atoms={"a0": MockRolloutAtomAgent(atom_id="a0", engine_alive=[True, True])},
            check_interval=10.0,
            metrics_port=0,
        )

        assert await agent.get_status() == JobStatus.FAILED


class TestMultiAtomMetricsAreIndependent:
    @pytest.mark.anyio
    async def test_per_atom_metrics_reflect_individual_health(self) -> None:
        atoms = {
            "a0": MockRolloutAtomAgent(atom_id="a0", engine_alive=[True, True]),
            "a1": MockRolloutAtomAgent(atom_id="a1", engine_alive=[True, False]),
        }
        agent = _make_agent(atoms)
        await agent.start()

        try:
            await asyncio.sleep(0.15)

            assert get_sample_value(agent._registry, "rollout_atom_alive", {"atom_id": "a0"}) == 1.0
            assert get_sample_value(agent._registry, "rollout_atom_alive", {"atom_id": "a1"}) == 0.0
            assert get_sample_value(agent._registry, "rollout_engine_alive", {"atom_id": "a1", "engine_index": "0"}) == 1.0
            assert get_sample_value(agent._registry, "rollout_engine_alive", {"atom_id": "a1", "engine_index": "1"}) == 0.0
        finally:
            await agent.shutdown()


class TestShutdownWithoutStart:
    @pytest.mark.anyio
    async def test_shutdown_without_start_is_safe(self) -> None:
        agent = FtRolloutAgent(
            atoms={"a0": MockRolloutAtomAgent(atom_id="a0", engine_alive=[True])},
            check_interval=10.0,
            metrics_port=0,
        )

        await agent.shutdown()


class TestPublicApi:
    def test_get_atom_ids(self) -> None:
        atoms = {
            "a0": MockRolloutAtomAgent(atom_id="a0", engine_alive=[True]),
            "a1": MockRolloutAtomAgent(atom_id="a1", engine_alive=[True, True]),
        }
        agent = FtRolloutAgent(atoms=atoms, check_interval=10.0, metrics_port=0)

        assert sorted(agent.get_atom_ids()) == ["a0", "a1"]

    def test_get_atom_agent(self) -> None:
        atom = MockRolloutAtomAgent(atom_id="a0", engine_alive=[True])
        agent = FtRolloutAgent(atoms={"a0": atom}, check_interval=10.0, metrics_port=0)

        assert agent.get_atom_agent("a0") is atom
