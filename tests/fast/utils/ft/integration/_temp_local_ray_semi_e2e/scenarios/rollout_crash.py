from __future__ import annotations

import ray
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.polling import (
    get_status,
    wait_for_all_subsystems_detecting,
    wait_for_subsystem_state,
    wait_for_training_stable,
)
from tests.fast.utils.ft.utils.fault_injection import FaultInjectionProtocol

from miles.utils.ft.controller.types import ControllerStatus


async def scenario_rollout_crash(
    handle: ray.actor.ActorHandle,
    injector: FaultInjectionProtocol,
    target_subsystem: str,
    target_node: str,
    *,
    stable_iterations: int = 3,
    stable_timeout: float = 60.0,
    detection_timeout: float = 180.0,
    recovery_timeout: float = 420.0,
) -> ControllerStatus:
    """Single rollout crash → recovery → verify subsystem and training unaffected.

    Returns the final ControllerStatus after recovery completes.
    """
    if stable_iterations > 0:
        await wait_for_training_stable(
            handle=handle,
            n_iterations=stable_iterations,
            timeout=stable_timeout,
        )

    status = get_status(handle)
    assert status.subsystem_states.get(target_subsystem) == "DetectingAnomalySt", (
        f"Precondition failed: {target_subsystem} in {status.subsystem_states.get(target_subsystem)}"
    )

    await injector.crash_rollout_on_node(target_node)

    await wait_for_subsystem_state(
        handle=handle,
        subsystem_name=target_subsystem,
        target_state="RecoveringSt",
        timeout=detection_timeout,
    )

    await wait_for_subsystem_state(
        handle=handle,
        subsystem_name=target_subsystem,
        target_state="DetectingAnomalySt",
        timeout=recovery_timeout,
    )

    status = get_status(handle)
    assert status.subsystem_states.get("training") == "DetectingAnomalySt", (
        f"Training affected: {status.subsystem_states.get('training')}"
    )

    return status
