from __future__ import annotations

import ray
from tests.fast.utils.ft.utils.fault_injection import FaultInjectionProtocol
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.polling import (
    assert_phase_path_contains,
    get_status,
    wait_for_mode_transition,
    wait_for_training_stable,
)

from miles.utils.ft.models.recovery import ControllerMode, ControllerStatus


async def scenario_transient_crash(
    handle: ray.actor.ActorHandle,
    injector: FaultInjectionProtocol,
    *,
    stable_iterations: int = 3,
    stable_timeout: float = 60.0,
    recovery_timeout: float = 120.0,
    post_recovery_iterations: int = 0,
    post_recovery_timeout: float = 60.0,
) -> ControllerStatus:
    """Single crash → auto-recovery → training resumes.

    Returns the final ControllerStatus after recovery completes.
    """
    if stable_iterations > 0:
        await wait_for_training_stable(
            handle=handle,
            n_iterations=stable_iterations,
            timeout=stable_timeout,
        )

    pre_status = get_status(handle)
    assert pre_status.mode == ControllerMode.MONITORING

    await injector.crash_training()

    status = await wait_for_mode_transition(
        handle=handle,
        target_mode=ControllerMode.MONITORING,
        timeout=recovery_timeout,
    )

    assert status.mode == ControllerMode.MONITORING
    assert status.bad_nodes == []

    if post_recovery_iterations > 0:
        await wait_for_training_stable(
            handle=handle,
            n_iterations=post_recovery_iterations,
            timeout=post_recovery_timeout,
        )

    final = get_status(handle)
    assert_phase_path_contains(
        final,
        [
            "RealtimeChecks",
            "StoppingAndRestarting",
            "MonitoringProgress",
        ],
    )

    return final
