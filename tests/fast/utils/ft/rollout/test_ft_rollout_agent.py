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
