"""Scenario: hang detection -> recovery -> training resumes."""

from __future__ import annotations

import time

from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.protocol import (
    FaultInjectionProtocol,
    FaultTestProtocol,
)

from miles.utils.ft.controller.types import ControllerMode, ControllerStatus


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
    # Step 1: record pre-hang baseline
    pre_status = await env.get_status()
    pre_iteration = pre_status.latest_iteration

    # Step 2: inject hang (SIGSTOP in E2E, iteration stall in semi-E2E)
    t0 = time.monotonic()
    await injector.inject_hang()

    # Step 3: wait for recovery cycle to complete
    status = await env.wait_for_mode_transition(target_mode=ControllerMode.MONITORING, timeout=recovery_timeout)
    elapsed = time.monotonic() - t0
    assert status.mode == ControllerMode.MONITORING

    # Step 4: verify detection happened within the expected time budget
    assert (
        elapsed < max_detection_seconds
    ), f"Hang detection+recovery took {elapsed:.1f}s, exceeds {max_detection_seconds}s"

    # Step 5: verify training resumes after recovery
    await env.wait_for_training_stable(n_iterations=post_recovery_iterations, timeout=post_recovery_timeout)

    post_status = await env.get_status()
    assert pre_iteration is not None and post_status.latest_iteration is not None
    assert (
        post_status.latest_iteration > pre_iteration
    ), f"Training not advancing: pre={pre_iteration} post={post_status.latest_iteration}"

    return post_status
