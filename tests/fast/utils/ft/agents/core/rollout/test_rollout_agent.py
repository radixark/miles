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
    cell_id: str = "default",
) -> tuple[FtRolloutAgent, list[MockEngine]]:
    engines = [MockEngine(a) for a in engine_alive]
    return FtRolloutAgent(
        cell_ids=[cell_id],
        get_engines=lambda _cid: engines,
        health_checker=_mock_health_checker,
        check_interval=check_interval,
    ), engines


def _make_multi_cell_agent(
    cells: dict[str, list[bool]],
    check_interval: float = 0.05,
) -> tuple[FtRolloutAgent, dict[str, list[MockEngine]]]:
    engines_map: dict[str, list[MockEngine]] = {
        cid: [MockEngine(a) for a in alive_flags]
        for cid, alive_flags in cells.items()
    }
    return FtRolloutAgent(
        cell_ids=list(cells.keys()),
        get_engines=lambda cid: engines_map[cid],
        health_checker=_mock_health_checker,
        check_interval=check_interval,
    ), engines_map


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


class TestMultiCell:
    """Previously FtRolloutAgent hardcoded a single 'default' cell_id,
    so non-default cells were never monitored. Now the agent accepts
    explicit cell_ids and produces per-cell metrics."""

    @pytest.mark.anyio
    async def test_multi_cell_exports_per_cell_metrics(self) -> None:
        agent, _ = _make_multi_cell_agent({"0": [True], "1": [True]})

        try:
            await asyncio.sleep(0.15)
            registry = agent._metrics_exporter.registry
            assert get_sample_value(registry, "rollout_cell_alive", {"cell_id": "0"}) == 1.0
            assert get_sample_value(registry, "rollout_cell_alive", {"cell_id": "1"}) == 1.0
        finally:
            await agent.shutdown()

    @pytest.mark.anyio
    async def test_multi_cell_dead_cell_reported_separately(self) -> None:
        agent, _ = _make_multi_cell_agent({"0": [True], "1": [False]})

        try:
            await asyncio.sleep(0.15)
            registry = agent._metrics_exporter.registry
            assert get_sample_value(registry, "rollout_cell_alive", {"cell_id": "0"}) == 1.0
            assert get_sample_value(registry, "rollout_cell_alive", {"cell_id": "1"}) == 0.0
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
