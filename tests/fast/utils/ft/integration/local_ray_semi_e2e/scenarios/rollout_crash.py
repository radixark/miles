"""Scenario: rollout engine crash -> Level 1 restart -> recovery."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from miles.utils.ft.controller.types import ControllerStatus

from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.protocol import (
    FaultTestProtocol,
)


async def scenario_rollout_crash(
    env: FaultTestProtocol,
    *,
    crash_fn: Callable[[], Awaitable[None]],
    target_subsystem: str,
    stable_iterations: int = 3,
    stable_timeout: float = 300.0,
    detection_timeout: float = 180.0,
    recovery_timeout: float = 420.0,
) -> ControllerStatus:
    """Crash rollout engine -> detect -> recover -> DetectingAnomaly.

    Args:
        crash_fn: Async callable that kills the rollout engine process.
            E2E: E2eFaultInjector.crash_rollout_on_node(node_id).
            Semi-E2E: testbed.kill_sglang_cell(cell_id).
        target_subsystem: Name of the rollout subsystem to watch (e.g. "rollout_default").
    """
    # Step 1: wait for training to stabilize
    await env.wait_for_training_stable(
        n_iterations=stable_iterations, timeout=stable_timeout
    )

    # Step 2: kill rollout engine
    await crash_fn()

    # Step 3: wait for subsystem to enter recovery
    await env.wait_for_subsystem_state(
        name=target_subsystem, state="RecoveringSt", timeout=detection_timeout
    )

    # Step 4: wait for recovery to complete
    status = await env.wait_for_subsystem_state(
        name=target_subsystem, state="DetectingAnomalySt", timeout=recovery_timeout
    )

    return status
