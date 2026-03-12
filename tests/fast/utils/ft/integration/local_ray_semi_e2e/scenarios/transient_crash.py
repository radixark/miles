"""Scenario: single crash -> auto-recovery -> training resumes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from miles.utils.ft.controller.types import ControllerMode, ControllerStatus

from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.protocol import (
    FaultInjectionProtocol,
    FaultTestProtocol,
)


async def scenario_transient_crash(
    env: FaultTestProtocol,
    injector: FaultInjectionProtocol,
    *,
    stable_iterations: int = 3,
    stable_timeout: float = 60.0,
    recovery_timeout: float = 120.0,
    post_recovery_iterations: int = 5,
    post_recovery_timeout: float = 120.0,
    crash_fn: Callable[[], Awaitable[None]] | None = None,
) -> ControllerStatus:
    """Single crash -> detection -> recovery -> training resumes.

    Args:
        crash_fn: Optional custom crash callable. When provided, used instead
            of injector.crash_training(). This allows reusing the same scenario
            for different crash types (e.g. Python exception via flag file).
    """
    # Step 1: wait for training to stabilize
    await env.wait_for_training_stable(
        n_iterations=stable_iterations, timeout=stable_timeout
    )

    # Step 2: inject fault
    if crash_fn is not None:
        await crash_fn()
    else:
        await injector.crash_training()

    # Step 3: wait for recovery cycle to complete (leave MONITORING, return to MONITORING)
    status = await env.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING, timeout=recovery_timeout
    )
    assert status.mode == ControllerMode.MONITORING

    # Step 4: verify training resumes after recovery
    await env.wait_for_training_stable(
        n_iterations=post_recovery_iterations, timeout=post_recovery_timeout
    )

    return status
