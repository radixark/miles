"""Shared fixtures and helpers for MilesTestbed-based semi-e2e tests."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from miles.utils.ft.controller.types import ControllerMode, ControllerStatus
from tests.fast.utils.ft.integration.conftest import FAST_TIMEOUT, RayNodeInfo
from tests.fast.utils.ft.testbed.config import TestbedConfig, TestbedNodeConfig
from tests.fast.utils.ft.testbed.train import MilesTestbed

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.anyio,
]


@pytest.fixture
async def make_testbed(
    local_ray_nodes: list[RayNodeInfo],
) -> AsyncIterator[Callable[..., MilesTestbed]]:
    """Factory fixture for creating MilesTestbed instances with custom configs.

    Usage::

        tb = await make_testbed(
            training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
            detectors=[TrainingCrashDetector()],
        )
    """
    created: list[MilesTestbed] = []

    async def _factory(**kwargs: Any) -> MilesTestbed:
        config = TestbedConfig(**kwargs)
        tb = await MilesTestbed.launch(config=config, ray_nodes=local_ray_nodes)
        created.append(tb)
        return tb

    yield _factory

    for tb in created:
        await tb.shutdown()


async def assert_no_recovery_triggered(
    testbed: MilesTestbed,
    observation_ticks: int = 20,
    timeout: float = 30.0,
) -> ControllerStatus:
    """Assert the controller stays in MONITORING for the given number of ticks."""
    initial = await testbed.get_status()
    target_ticks = initial.tick_count + observation_ticks
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        status = await testbed.get_status()
        assert (
            status.mode == ControllerMode.MONITORING
        ), f"Unexpected recovery at tick {status.tick_count}"
        if status.tick_count >= target_ticks:
            return status
        await asyncio.sleep(0.3)

    raise TimeoutError(
        f"Controller did not reach {observation_ticks} additional ticks within {timeout}s"
    )
