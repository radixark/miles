"""E2E: Repeated crash → reattempt fails → DIAGNOSING."""

from __future__ import annotations

import asyncio

import ray

from miles.utils.ft.models import ControllerMode, RecoveryPhase
from tests.e2e.ft.conftest import (
    FaultInjectorFactory,
    assert_phase_path_contains,
    find_training_pid,
    wait_for_mode_transition,
    wait_for_recovery_complete,
    wait_for_recovery_phase,
    wait_for_training_pid,
    wait_for_training_stable,
)

async def test_repeated_crash_enters_diagnosing(
    ft_controller_handle: ray.actor.ActorHandle,
    fault_injector: FaultInjectorFactory,
    target_node: str,
) -> None:
    # Step 1: Wait for stable baseline
    await wait_for_training_stable(
        handle=ft_controller_handle,
        n_iterations=3,
        timeout=180.0,
    )

    injector = fault_injector.deploy_to(node_id=target_node)

    # Step 2: First kill
    first_pid = find_training_pid(injector)
    ray.get(injector.kill_process.remote(pid=first_pid, sig=9))

    await wait_for_mode_transition(
        handle=ft_controller_handle,
        target_mode=ControllerMode.MONITORING,
        timeout=180.0,
    )

    # Step 3: Second kill during MONITORING phase
    await asyncio.sleep(5.0)
    second_pid = await wait_for_training_pid(injector)
    ray.get(injector.kill_process.remote(pid=second_pid, sig=9))

    # Step 4: Verify escalation to DIAGNOSING → NOTIFY
    status = await wait_for_recovery_phase(
        handle=ft_controller_handle,
        phase=RecoveryPhase.DIAGNOSING,
        timeout=180.0,
    )
    assert status.recovery_phase == RecoveryPhase.DIAGNOSING

    final_status = await wait_for_recovery_complete(
        handle=ft_controller_handle,
        timeout=300.0,
    )
    assert final_status.mode == ControllerMode.MONITORING

    assert_phase_path_contains(final_status, [
        RecoveryPhase.DIAGNOSING,
        RecoveryPhase.NOTIFY,
        RecoveryPhase.DONE,
    ])
