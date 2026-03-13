"""Scenario: healthy training produces no false-positive recovery."""

from __future__ import annotations

import asyncio
import time

from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.protocol import FaultTestProtocol

from miles.utils.ft.controller.types import ControllerMode, ControllerStatus


async def scenario_no_false_positive(
    env: FaultTestProtocol,
    *,
    observation_iterations: int = 10,
    timeout: float = 120.0,
    poll_interval: float = 5.0,
) -> ControllerStatus:
    """Observe healthy training for N iterations, assert no recovery triggered.

    Returns the final status so callers can add extra assertions
    (e.g. all subsystems in DetectingAnomalySt).
    """
    # Step 1: wait for training to produce enough iterations
    await env.wait_for_training_stable(n_iterations=observation_iterations, timeout=timeout)

    # Step 2: verify controller is still in MONITORING with no active recovery
    deadline = time.monotonic() + poll_interval * 3
    while time.monotonic() < deadline:
        status = await env.get_status()
        assert status.mode == ControllerMode.MONITORING, f"False positive: mode={status.mode} during healthy training"
        assert not status.recovery_in_progress, "False positive: recovery_in_progress during healthy training"
        await asyncio.sleep(poll_interval)

    return await env.get_status()
