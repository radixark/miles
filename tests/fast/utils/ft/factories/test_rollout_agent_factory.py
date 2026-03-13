"""Tests for rollout agent factory multi-cell interface.

Previously build_rollout_agent accepted an opaque rollout_manager object
and hardcoded a single "default" cell_id internally, so non-default cells
were never monitored. Now the factory requires explicit cell_ids and
get_engines so that all cells are properly wired.
"""

from __future__ import annotations

import pytest

from miles.utils.ft.agents.core.rollout.rollout_agent import FtRolloutAgent


class TestFtRolloutAgentInterface:
    def test_requires_cell_ids_and_get_engines(self) -> None:
        """FtRolloutAgent no longer accepts an opaque rollout_manager —
        callers must provide explicit cell_ids and get_engines."""
        with pytest.raises(TypeError):
            FtRolloutAgent(object(), health_checker=lambda e: None, check_interval=1.0)  # type: ignore[call-arg]

    @pytest.mark.anyio
    async def test_single_cell_construction(self) -> None:
        engines = [object()]
        agent = FtRolloutAgent(
            cell_ids=["default"],
            get_engines=lambda cid: engines,
            health_checker=lambda e: None,  # type: ignore[return-value,arg-type]
            check_interval=100.0,
        )
        try:
            assert agent._health_checker.cell_ids == ["default"]
        finally:
            await agent.shutdown()

    @pytest.mark.anyio
    async def test_multi_cell_construction(self) -> None:
        agent = FtRolloutAgent(
            cell_ids=["0", "1", "2"],
            get_engines=lambda cid: [object()],
            health_checker=lambda e: None,  # type: ignore[return-value,arg-type]
            check_interval=100.0,
        )
        try:
            assert sorted(agent._health_checker.cell_ids) == ["0", "1", "2"]
        finally:
            await agent.shutdown()
