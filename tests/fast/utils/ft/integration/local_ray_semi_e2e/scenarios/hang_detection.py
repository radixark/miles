from __future__ import annotations

import ray

from miles.utils.ft.models.recovery import ControllerMode, ControllerStatus

from tests.fast.utils.ft.helpers.fault_injection import FaultInjectionProtocol
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.polling import (
    get_status,
    wait_for_mode,
)


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
