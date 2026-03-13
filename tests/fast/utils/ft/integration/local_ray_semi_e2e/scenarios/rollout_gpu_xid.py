"""Scenario: GPU XID on rollout node -> recovery -> node evicted."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.protocol import FaultTestProtocol

from miles.utils.ft.controller.types import ControllerMode, ControllerStatus


async def scenario_rollout_gpu_xid(
    env: FaultTestProtocol,
    *,
    inject_xid_fn: Callable[[], Awaitable[None]],
    target_subsystem: str,
    detection_timeout: float = 180.0,
    recovery_timeout: float = 420.0,
) -> ControllerStatus:
    """Inject GPU XID on rollout node -> detect -> recover.

    The caller should assert eviction/bad-node status after this returns,
    since the eviction check mechanism differs between E2E (K8s labels) and
    semi-E2E (testbed.node_manager.was_ever_marked_bad).

    Args:
        inject_xid_fn: Async callable that triggers GPU XID on the target node.
            E2E: ray.get(injector.trigger_gpu_xid.remote()).
            Semi-E2E: testbed.inject_gpu_xid(node_id).
        target_subsystem: Rollout subsystem name (e.g. "rollout_default").
    """
    # Step 1: inject GPU XID
    await inject_xid_fn()

    # Step 2: wait for subsystem to enter recovery
    await env.wait_for_subsystem_state(name=target_subsystem, state="RecoveringSt", timeout=detection_timeout)

    # Step 3: wait for recovery to complete
    status = await env.wait_for_subsystem_state(
        name=target_subsystem, state="DetectingAnomalySt", timeout=recovery_timeout
    )

    # Step 4: verify subsystem recovered and controller is back to monitoring
    assert (
        status.subsystem_states[target_subsystem] == "DetectingAnomalySt"
    ), f"{target_subsystem} not in DetectingAnomalySt: {status.subsystem_states[target_subsystem]}"
    assert (
        status.mode == ControllerMode.MONITORING
    ), f"Controller not in MONITORING after GPU XID recovery: {status.mode}"

    return status
