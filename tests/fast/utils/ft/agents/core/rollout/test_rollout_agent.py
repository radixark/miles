from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from miles.utils.ft.agents.core.rollout.rollout_agent import FtRolloutAgent
from tests.fast.utils.ft.agents.core.rollout.conftest import MockEngine
from tests.fast.utils.ft.utils.metric_injectors import get_sample_value


async def _mock_health_checker(engine: object) -> None:
    if not engine.alive:  # type: ignore[attr-defined]
        raise ConnectionError("engine dead")


def _make_agent(
    engine_alive: list[bool],
    check_interval: float = 0.05,
) -> tuple[FtRolloutAgent, list[MockEngine]]:
    engines = [MockEngine(a) for a in engine_alive]
    rm = MagicMock()
    rm.all_rollout_engines = engines
    return FtRolloutAgent(rm, health_checker=_mock_health_checker, check_interval=check_interval), engines


class TestIntegration:
    """Verify FtRolloutAgent wires RolloutHealthChecker → RolloutMetricsExporter."""

    @pytest.mark.anyio
    async def test_healthy_engines_appear_in_prometheus(self) -> None:
        agent, _ = _make_agent([True, True])

        try:
            await asyncio.sleep(0.15)

            registry = agent._metrics_exporter.registry
            assert get_sample_value(registry, "rollout_cell_alive", {"cell_id": "default"}) == 1.0
        finally:
            await agent.shutdown()

    @pytest.mark.anyio
    async def test_dead_engine_appears_in_prometheus(self) -> None:
        agent, _ = _make_agent([False, True])

        try:
            await asyncio.sleep(0.15)

            registry = agent._metrics_exporter.registry
            assert get_sample_value(registry, "rollout_cell_alive", {"cell_id": "default"}) == 0.0
        finally:
            await agent.shutdown()


class TestLifecycle:
    @pytest.mark.anyio
    async def test_address_available_immediately(self) -> None:
        agent, _ = _make_agent([True])

        try:
            assert agent.address.startswith("http://")
            assert int(agent.address.rsplit(":", 1)[-1]) > 0
        finally:
            await agent.shutdown()

    @pytest.mark.anyio
    async def test_shutdown_is_safe(self) -> None:
        agent, _ = _make_agent([True], check_interval=10.0)
        await agent.shutdown()
