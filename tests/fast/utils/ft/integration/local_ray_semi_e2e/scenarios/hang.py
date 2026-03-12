"""Scenario: hang detection -> recovery -> training resumes."""

from __future__ import annotations

from miles.utils.ft.controller.types import ControllerMode, ControllerStatus

from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.protocol import (
    FaultInjectionProtocol,
    FaultTestProtocol,
)


async def scenario_hang_detection_and_recovery(
    env: FaultTestProtocol,
    injector: FaultInjectionProtocol,
    *,
    hang_timeout: float = 720.0,
    recovery_timeout: float = 720.0,
    max_detection_seconds: float = 660.0,
    post_recovery_iterations: int = 5,
    post_recovery_timeout: float = 300.0,
) -> ControllerStatus:
    """Inject hang -> detection -> recovery -> training resumes.

    The caller is responsible for waiting for training to stabilize
    before calling this function, since hang detection timeouts vary
    significantly between E2E (minutes) and semi-E2E (seconds).
    """
    # Step 1: inject hang (SIGSTOP in E2E, iteration stall in semi-E2E)
    await injector.inject_hang()

    # Step 2: wait for recovery cycle to complete
    status = await env.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING, timeout=recovery_timeout
    )
    assert status.mode == ControllerMode.MONITORING

    # Step 3: verify training resumes after recovery
    await env.wait_for_training_stable(
        n_iterations=post_recovery_iterations, timeout=post_recovery_timeout
    )

    return status
