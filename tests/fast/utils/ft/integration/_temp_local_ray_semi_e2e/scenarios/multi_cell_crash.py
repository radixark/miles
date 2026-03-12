from __future__ import annotations

import asyncio
from collections.abc import Callable, Awaitable

import ray
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.polling import (
    get_status,
    wait_for_all_subsystems_detecting,
    wait_for_training_stable,
)

from miles.utils.ft.controller.types import ControllerStatus


async def scenario_multi_cell_crash(
    handle: ray.actor.ActorHandle,
    crash_fns: list[Callable[[], Awaitable[None]]],
    all_rollout_subsystems: list[str],
    *,
    stable_iterations: int = 3,
    stable_timeout: float = 60.0,
    stagger_delay: float = 30.0,
    detection_timeout: float = 120.0,
    recovery_timeout: float = 600.0,
) -> ControllerStatus:
    """Crash multiple rollout cells (staggered) → independent recovery.

    crash_fns: list of async callables, one per cell to crash (NOT all cells).
    all_rollout_subsystems: names of all rollout subsystems (including the surviving one).

    Crashes len(crash_fns) cells with stagger_delay between each, then waits
    for all subsystems to return to DetectingAnomalySt.
    """
    if stable_iterations > 0:
        await wait_for_training_stable(
            handle=handle,
            n_iterations=stable_iterations,
            timeout=stable_timeout,
        )

    status = get_status(handle)
    for name in all_rollout_subsystems:
        assert status.subsystem_states.get(name) == "DetectingAnomalySt", (
            f"Precondition failed: {name} in {status.subsystem_states.get(name)}"
        )

    # Step: crash cells with stagger
    for i, crash_fn in enumerate(crash_fns):
        if i > 0:
            await asyncio.sleep(stagger_delay)
        await crash_fn()

    # Step: wait for at least len(crash_fns) subsystems to enter RecoveringSt
    expected_recovering = len(crash_fns)
    deadline = asyncio.get_event_loop().time() + detection_timeout
    while asyncio.get_event_loop().time() < deadline:
        status = get_status(handle)
        recovering_count = sum(
            1 for name in all_rollout_subsystems
            if status.subsystem_states.get(name) == "RecoveringSt"
        )
        if recovering_count >= expected_recovering:
            break
        await asyncio.sleep(1.0)
    else:
        status = get_status(handle)
        raise AssertionError(
            f"Expected {expected_recovering} recovering subsystems, "
            f"states={status.subsystem_states}"
        )

    # Step: wait for all subsystems (including training) to return to detecting
    status = await wait_for_all_subsystems_detecting(
        handle=handle,
        timeout=recovery_timeout,
    )

    assert status.subsystem_states.get("training") == "DetectingAnomalySt", (
        f"Training affected: {status.subsystem_states.get('training')}"
    )

    return status
