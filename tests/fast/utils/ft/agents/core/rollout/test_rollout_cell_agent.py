from __future__ import annotations

import pytest

from tests.fast.utils.ft.agents.core.rollout.conftest import MockRolloutCellAgent


class TestCheckHealth:
    @pytest.mark.anyio
    async def test_healthy_cell_returns_true(self) -> None:
        agent = MockRolloutCellAgent(cell_id="a0", healthy=True)

        result = await agent.check_health()

        assert result is True

    @pytest.mark.anyio
    async def test_unhealthy_cell_returns_false(self) -> None:
        agent = MockRolloutCellAgent(cell_id="a1", healthy=False)

        result = await agent.check_health()

        assert result is False


class TestConsecutiveChecks:
    @pytest.mark.anyio
    async def test_result_updates_after_state_change(self) -> None:
        agent = MockRolloutCellAgent(cell_id="a0", healthy=True)

        # Step 1: healthy
        assert await agent.check_health() is True

        # Step 2: simulate cell becoming unhealthy
        agent._health_checker.healthy = False
        assert await agent.check_health() is False


class TestCellId:
    def test_cell_id_property(self) -> None:
        agent = MockRolloutCellAgent(cell_id="my-cell")

        assert agent.cell_id == "my-cell"
