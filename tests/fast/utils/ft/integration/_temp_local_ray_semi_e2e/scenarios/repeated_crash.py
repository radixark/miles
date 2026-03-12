from __future__ import annotations

import ray
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.polling import (
    wait_for_mode_transition,
    wait_for_recovery_complete,
    wait_for_recovery_phase,
    wait_for_training_stable,
)
from tests.fast.utils.ft.utils.fault_injection import FaultInjectionProtocol

from miles.utils.ft.controller.types import ControllerMode, ControllerStatus


async def scenario_repeated_crash(
    handle: ray.actor.ActorHandle,
    injector: FaultInjectionProtocol,
    *,
    stable_iterations: int = 3,
    stable_timeout: float = 60.0,
    recovery_timeout: float = 120.0,
) -> ControllerStatus:
    """Two consecutive crashes → escalation to DIAGNOSING → recovery.

    Returns the final ControllerStatus after the second recovery completes.
    """
    if stable_iterations > 0:
        await wait_for_training_stable(
            handle=handle,
            n_iterations=stable_iterations,
            timeout=stable_timeout,
        )

    # First crash → recovery → back to MONITORING
    await injector.crash_training()
    await wait_for_mode_transition(
        handle=handle,
        target_mode=ControllerMode.MONITORING,
        timeout=recovery_timeout,
    )

    # Second crash → should escalate to DIAGNOSING
    await injector.crash_training()
    await wait_for_recovery_phase(
        handle=handle,
        phase="StopTimeDiagnosticsSt",
        timeout=recovery_timeout,
    )

    # Wait for full recovery
    final = await wait_for_recovery_complete(
        handle=handle,
        timeout=recovery_timeout,
    )
    assert final.mode == ControllerMode.MONITORING

    return final
