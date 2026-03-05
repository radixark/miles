"""E2E Scenario 1: Transient crash — single kill → auto-recovery.

The most common recovery path (~22.7% of failures per ByteRobust):
  1. Kill one training process
  2. Controller detects crash → ENTER_RECOVERY
  3. CHECK_ALERTS finds no hardware issues → REATTEMPTING
  4. Training restarts and stabilizes → MONITORING → DONE
  5. No nodes marked as bad (transient fault)
"""

from __future__ import annotations

import time

import pytest
import ray
from miles.utils.ft.models import ControllerMode
from tests.e2e.ft.conftest import FaultInjectorFactory, FtSystem, wait_for_recovery_complete, wait_for_training_stable

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.timeout(600),
]


async def test_transient_crash_auto_recovery(
    ft_system: FtSystem,
    fault_injector: FaultInjectorFactory,
    target_node: str,
) -> None:
    controller = ft_system.controller

    # Wait for training to be stable before injection
    await wait_for_training_stable(
        controller=controller,
        mini_wandb=ft_system.mini_wandb,
        n_iterations=5,
        timeout=300.0,
    )
    pre_status = controller.get_status()
    assert pre_status.mode == ControllerMode.MONITORING

    # Deploy fault injector and kill one training process
    injector = fault_injector.deploy_to(node_id=target_node)
    procs = ray.get(injector.find_training_processes.remote())
    assert len(procs) > 0, f"No training processes found on node {target_node}"

    target_pid = procs[0]["pid"]
    t_inject = time.monotonic()
    ray.get(injector.kill_process.remote(pid=target_pid, sig=9))

    # Wait for recovery to complete
    status = await wait_for_recovery_complete(
        controller=controller,
        timeout=300.0,
    )

    t_recover = time.monotonic() - t_inject
    assert status.mode == ControllerMode.MONITORING

    # Training should be running again and stable
    await wait_for_training_stable(
        controller=controller,
        mini_wandb=ft_system.mini_wandb,
        n_iterations=10,
        timeout=300.0,
    )

    # No nodes should be marked as bad (transient fault)
    final_status = controller.get_status()
    assert (
        final_status.bad_nodes == []
    ), f"Expected no bad nodes for transient crash, got: {final_status.bad_nodes}"

    # Sanity check: recovery time should be reasonable (< 5 min)
    assert t_recover < 300.0, f"Recovery took too long: {t_recover:.1f}s"
