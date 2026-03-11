from __future__ import annotations

import pytest

from miles.utils.ft.rollout.atom_agent import AtomHealthResult, RolloutAtomAgent
from tests.fast.utils.ft.rollout.conftest import MockRolloutAtomAgent


class TestCheckHealth:
    @pytest.mark.anyio
    async def test_all_engines_alive_returns_healthy(self) -> None:
        agent = MockRolloutAtomAgent(atom_id="a0", engine_alive=[True, True, True])

        result = await agent.check_health()

        assert result.is_healthy is True
        assert result.alive_engines == 3
        assert result.total_engines == 3
        assert result.dead_engine_indices == []
        assert result.atom_id == "a0"

    @pytest.mark.anyio
    async def test_one_engine_dead_returns_unhealthy(self) -> None:
        agent = MockRolloutAtomAgent(atom_id="a1", engine_alive=[True, False, True])

        result = await agent.check_health()

        assert result.is_healthy is False
        assert result.alive_engines == 2
        assert result.dead_engine_indices == [1]

    @pytest.mark.anyio
    async def test_all_engines_dead_returns_unhealthy(self) -> None:
        agent = MockRolloutAtomAgent(atom_id="a2", engine_alive=[False, False, False])

        result = await agent.check_health()

        assert result.is_healthy is False
        assert result.alive_engines == 0
        assert result.dead_engine_indices == [0, 1, 2]


class TestIsHealthy:
    def test_returns_false_before_any_check(self) -> None:
        agent = MockRolloutAtomAgent(atom_id="a0", engine_alive=[True, True])

        assert agent.is_healthy() is False

    @pytest.mark.anyio
    async def test_returns_true_after_healthy_check(self) -> None:
        agent = MockRolloutAtomAgent(atom_id="a0", engine_alive=[True, True])

        await agent.check_health()

        assert agent.is_healthy() is True


class TestNodeTracking:
    def test_set_and_get_node_ids(self) -> None:
        agent = MockRolloutAtomAgent(atom_id="a0", engine_alive=[True])

        agent.set_node_ids({"node-1", "node-2"})

        assert agent.get_node_ids() == {"node-1", "node-2"}

    def test_set_node_ids_returns_copy(self) -> None:
        agent = MockRolloutAtomAgent(atom_id="a0", engine_alive=[True])
        agent.set_node_ids({"node-1"})

        returned = agent.get_node_ids()
        returned.add("node-extra")

        assert agent.get_node_ids() == {"node-1"}


class TestEngineCounts:
    def test_get_engine_count(self) -> None:
        agent = MockRolloutAtomAgent(atom_id="a0", engine_alive=[True, False, True])

        assert agent.get_engine_count() == 3

    def test_get_alive_engine_count_before_check(self) -> None:
        agent = MockRolloutAtomAgent(atom_id="a0", engine_alive=[True, True])

        assert agent.get_alive_engine_count() == 0

    @pytest.mark.anyio
    async def test_get_alive_engine_count_after_check(self) -> None:
        agent = MockRolloutAtomAgent(atom_id="a0", engine_alive=[True, False, True])

        await agent.check_health()

        assert agent.get_alive_engine_count() == 2


class TestConsecutiveChecksUpdateResult:
    @pytest.mark.anyio
    async def test_result_updates_after_state_change(self) -> None:
        agent = MockRolloutAtomAgent(atom_id="a0", engine_alive=[True, True, True])

        # Step 1: all alive
        result1 = await agent.check_health()
        assert result1.is_healthy is True

        # Step 2: simulate engine 2 dying
        agent._engine_alive[2] = False
        result2 = await agent.check_health()

        assert result2.is_healthy is False
        assert result2.alive_engines == 2
        assert result2.dead_engine_indices == [2]
        assert result2.checked_at >= result1.checked_at
