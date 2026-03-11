from __future__ import annotations

import time

import ray
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.polling import (
    assert_phase_path_contains,
    get_status,
    wait_for_mode,
    wait_for_mode_transition,
    wait_for_training_stable,
)
from tests.fast.utils.ft.utils.fault_injection import FaultInjectionProtocol

from miles.utils.ft.controller.types import ControllerMode, ControllerStatus


async def scenario_hang_detection(
    handle: ray.actor.ActorHandle,
    injector: FaultInjectionProtocol,
    *,
    hang_timeout: float = 30.0,
) -> ControllerStatus:
    """Inject hang → controller detects stale iteration → ENTER_RECOVERY.

    Returns the ControllerStatus after recovery is triggered.
    """
    pre_status = get_status(handle)
    assert pre_status.mode == ControllerMode.MONITORING

    await injector.inject_hang()

    status = await wait_for_mode(
        handle=handle,
        target_mode=ControllerMode.RECOVERY,
        timeout=hang_timeout,
    )

    assert status.mode == ControllerMode.RECOVERY
    return status


async def scenario_hang_detection_and_recovery(
    handle: ray.actor.ActorHandle,
    injector: FaultInjectionProtocol,
    *,
    hang_timeout: float = 30.0,
    recovery_timeout: float = 120.0,
    post_recovery_iterations: int = 5,
    post_recovery_timeout: float = 300.0,
    max_detection_seconds: float | None = None,
) -> ControllerStatus:
    """Inject hang → detect → full recovery → verify training resumes."""
    t_start = time.monotonic()

    await scenario_hang_detection(
        handle=handle,
        injector=injector,
        hang_timeout=hang_timeout,
    )

    if max_detection_seconds is not None:
        elapsed = time.monotonic() - t_start
        assert (
            elapsed < max_detection_seconds
        ), f"Hang detection took {elapsed:.0f}s, expected < {max_detection_seconds}s"

    status = await wait_for_mode_transition(
        handle=handle,
        target_mode=ControllerMode.MONITORING,
        timeout=recovery_timeout,
    )

    assert_phase_path_contains(
        status,
        [
            "RealtimeChecksSt",
            "StoppingAndRestartingSt",
            "MonitoringProgressSt",
        ],
    )

    if post_recovery_iterations > 0:
        await wait_for_training_stable(
            handle=handle,
            n_iterations=post_recovery_iterations,
            timeout=post_recovery_timeout,
        )

    return get_status(handle)
