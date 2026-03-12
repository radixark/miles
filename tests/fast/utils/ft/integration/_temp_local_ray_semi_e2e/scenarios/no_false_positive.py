from __future__ import annotations

import asyncio
import time

import ray
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.polling import get_status

from miles.utils.ft.controller.types import ControllerMode, ControllerStatus


async def scenario_no_false_positive(
    handle: ray.actor.ActorHandle,
    *,
    observation_iterations: int | None = None,
    observation_ticks: int | None = None,
    timeout: float = 60.0,
    poll_interval: float = 0.5,
) -> ControllerStatus:
    """Let the controller run with no injected faults.

    Verify it stays in MONITORING mode and never enters recovery.
    Exactly one of observation_iterations (E2E) or observation_ticks (local_ray)
    must be provided.
    """
    assert (observation_iterations is None) != (
        observation_ticks is None
    ), "Exactly one of observation_iterations or observation_ticks must be provided"

    initial = get_status(handle)

    if observation_iterations is not None:
        baseline = initial.latest_iteration or 0

        def _reached(s: ControllerStatus) -> bool:
            return (s.latest_iteration or 0) - baseline >= observation_iterations

    else:
        assert observation_ticks is not None
        target_ticks = initial.tick_count + observation_ticks

        def _reached(s: ControllerStatus) -> bool:
            return s.tick_count >= target_ticks

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        status = get_status(handle)
        assert (
            status.mode == ControllerMode.MONITORING
        ), f"Unexpected mode transition to {status.mode} at tick {status.tick_count}"
        if _reached(status):
            return status
        await asyncio.sleep(poll_interval)

    raise TimeoutError(f"Controller did not reach observation target within {timeout}s")
