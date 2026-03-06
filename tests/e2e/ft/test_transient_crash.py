"""E2E: Transient crash — single kill → auto-recovery."""

from __future__ import annotations

import time

import ray
from miles.utils.ft.models import ControllerMode, RecoveryPhase
from tests.e2e.ft.conftest import (
    FaultInjectorFactory,
    assert_phase_path_contains,
    find_training_pid,
    get_status,
    wait_for_mode_transition,
    wait_for_training_stable,
)

async def test_transient_crash_auto_recovery(
    ft_controller_handle: ray.actor.ActorHandle,
    fault_injector: FaultInjectorFactory,
    target_node: str,
) -> None:
    await wait_for_training_stable(
        handle=ft_controller_handle,
        n_iterations=3,
        timeout=180.0,
    )
    pre_status = get_status(ft_controller_handle)
    assert pre_status.mode == ControllerMode.MONITORING

    injector = fault_injector.deploy_to(node_id=target_node)
    target_pid = find_training_pid(injector, node_id=target_node)
    t_inject = time.monotonic()
    ray.get(injector.kill_process.remote(pid=target_pid, sig=9))

    status = await wait_for_mode_transition(
        handle=ft_controller_handle,
        target_mode=ControllerMode.MONITORING,
        timeout=300.0,
    )

    t_recover = time.monotonic() - t_inject
    assert status.mode == ControllerMode.MONITORING

    await wait_for_training_stable(
        handle=ft_controller_handle,
        n_iterations=5,
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
