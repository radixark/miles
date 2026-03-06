"""E2E: Transient crash — single kill → auto-recovery."""

from __future__ import annotations

import time

import pytest
import ray
from miles.utils.ft.models import ControllerMode, RecoveryPhase
from tests.e2e.ft.conftest import (
    FaultInjectorFactory,
    assert_phase_path_contains,
    get_status,
    wait_for_recovery_complete,
    wait_for_training_stable,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.timeout(600),
]


async def test_transient_crash_auto_recovery(
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
    pre_status = get_status(ft_controller_handle)
    assert pre_status.mode == ControllerMode.MONITORING

    # Step 2: Kill one training process
    injector = fault_injector.deploy_to(node_id=target_node)
    procs = ray.get(injector.find_training_processes.remote())
    assert len(procs) > 0, f"No training processes found on node {target_node}"

    target_pid = procs[0]["pid"]
    t_inject = time.monotonic()
    ray.get(injector.kill_process.remote(pid=target_pid, sig=9))

    # Step 3: Wait for auto-recovery
    status = await wait_for_recovery_complete(
        handle=ft_controller_handle,
        timeout=300.0,
    )

    t_recover = time.monotonic() - t_inject
    assert status.mode == ControllerMode.MONITORING

    # Step 4: Verify training resumes and no nodes marked bad
    await wait_for_training_stable(
        handle=ft_controller_handle,
        n_iterations=10,
        timeout=300.0,
    )

    final_status = get_status(ft_controller_handle)
    assert (
        final_status.bad_nodes == []
    ), f"Expected no bad nodes for transient crash, got: {final_status.bad_nodes}"

    assert_phase_path_contains(final_status, [
        RecoveryPhase.CHECK_ALERTS,
        RecoveryPhase.REATTEMPTING,
        RecoveryPhase.MONITORING,
        RecoveryPhase.DONE,
    ])

    assert t_recover < 300.0, f"Recovery took too long: {t_recover:.1f}s"
