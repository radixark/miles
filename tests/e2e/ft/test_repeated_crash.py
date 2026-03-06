"""E2E: Repeated crash → reattempt fails → DIAGNOSING."""

from __future__ import annotations

import asyncio

import pytest
import ray

from miles.utils.ft.models import ControllerMode, RecoveryPhase
from tests.e2e.ft.conftest import (
    FaultInjectorFactory,
    assert_phase_path_contains,
    get_status,
    wait_for_mode_transition,
    wait_for_recovery_complete,
    wait_for_recovery_phase,
    wait_for_training_stable,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.timeout(600),
]


async def test_repeated_crash_enters_diagnosing(
    ft_controller_handle: ray.actor.ActorHandle,
    fault_injector: FaultInjectorFactory,
    target_node: str,
) -> None:
    # Step 1: Wait for stable baseline
    await wait_for_training_stable(
        handle=ft_controller_handle,
        n_iterations=5,
        timeout=300.0,
    )

    injector = fault_injector.deploy_to(node_id=target_node)

    # Step 2: First kill
    procs = ray.get(injector.find_training_processes.remote())
    assert len(procs) > 0
    ray.get(injector.kill_process.remote(pid=procs[0]["pid"], sig=9))

    await wait_for_mode_transition(
        handle=ft_controller_handle,
        target_mode=ControllerMode.MONITORING,
        timeout=180.0,
    )

    # Step 3: Second kill during MONITORING phase
    await asyncio.sleep(5.0)
    for _ in range(10):
        procs = ray.get(injector.find_training_processes.remote())
        if procs:
            break
        await asyncio.sleep(3.0)
    assert len(procs) > 0, "No training processes found for second kill"
    ray.get(injector.kill_process.remote(pid=procs[0]["pid"], sig=9))

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
