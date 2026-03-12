"""Scenario: rollout repeated crash -> escalation -> node evicted."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from miles.utils.ft.controller.types import ControllerStatus

from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.protocol import (
    FaultTestProtocol,
)


async def scenario_rollout_repeated_crash(
    env: FaultTestProtocol,
    *,
    start_killing_fn: Callable[[], Awaitable[asyncio.Task[None]]],
    stop_killing_fn: Callable[[], Awaitable[None]],
    check_eviction_fn: Callable[[], Awaitable[bool]],
    target_subsystem: str,
    detection_timeout: float = 180.0,
    eviction_timeout: float = 900.0,
    eviction_poll_interval: float = 10.0,
    recovery_timeout: float = 420.0,
) -> ControllerStatus:
    """Repeatedly kill rollout -> L1 restarts fail -> escalate to node eviction.

    Args:
        start_killing_fn: Starts a background task that continuously kills
            the rollout process. Returns the task handle.
        stop_killing_fn: Stops the background killing task.
        check_eviction_fn: Returns True if the target node has been evicted.
        target_subsystem: Rollout subsystem name (e.g. "rollout_default").
    """
    try:
        # Step 1: start continuous killing
        await start_killing_fn()

        # Step 2: wait for subsystem to enter recovery
        await env.wait_for_subsystem_state(
            name=target_subsystem, state="RecoveringSt", timeout=detection_timeout
        )

        # Step 3: wait for node eviction (L1 retries exhaust -> escalate)
        deadline = asyncio.get_event_loop().time() + eviction_timeout
        while asyncio.get_event_loop().time() < deadline:
            if await check_eviction_fn():
                break
            await asyncio.sleep(eviction_poll_interval)
        else:
            evicted = await check_eviction_fn()
            assert evicted, (
                f"Node not evicted after {eviction_timeout}s of repeated rollout crashes"
            )

    finally:
        # Step 4: stop background killing
        await stop_killing_fn()

    # Step 5: wait for subsystem to recover (rebuilt on spare node)
    status = await env.wait_for_subsystem_state(
        name=target_subsystem, state="DetectingAnomalySt", timeout=recovery_timeout
    )

    return status
