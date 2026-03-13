"""Scenario: multi-cell rollout crash with staggered failures -> independent recovery."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.protocol import FaultTestProtocol

from miles.utils.ft.controller.types import ControllerStatus


async def scenario_multi_cell_crash(
    env: FaultTestProtocol,
    *,
    crash_fns: list[Callable[[], Awaitable[None]]],
    all_rollout_subsystems: list[str],
    stable_iterations: int = 3,
    stable_timeout: float = 60.0,
    stagger_delay: float = 30.0,
    detection_timeout: float = 300.0,
    recovery_timeout: float = 600.0,
) -> ControllerStatus:
    """Crash multiple rollout cells with stagger, wait for all to recover.

    Args:
        crash_fns: One async callable per cell to crash. The number of fns
            determines how many cells are crashed (must be < len(all_rollout_subsystems)).
        all_rollout_subsystems: Names of all rollout subsystems (including the
            ones not being crashed).
        stagger_delay: Seconds between successive crash injections.
    """
    if stable_iterations > 0:
        await env.wait_for_training_stable(n_iterations=stable_iterations, timeout=stable_timeout)

    # Step 1: crash cells with stagger delay
    for i, crash_fn in enumerate(crash_fns):
        if i > 0:
            await asyncio.sleep(stagger_delay)
        await crash_fn()

    # Step 2: wait for all rollout subsystems to return to DetectingAnomalySt
    status = await env.wait_for_all_subsystems_detecting(timeout=recovery_timeout)

    # Step 3: verify all subsystems recovered
    assert all(
        state == "DetectingAnomalySt" for state in status.subsystem_states.values()
    ), f"Not all subsystems in DetectingAnomalySt: {status.subsystem_states}"

    return status
