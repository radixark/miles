"""E2E: Python exception in training → auto-recovery.

Injects a FaultInjectionError via flag file into the training process.
Unlike test_transient_crash (SIGKILL → instant death), this test verifies
that a normal Python exception (CUDA OOM, data error, shape mismatch, etc.)
also triggers proper recovery through the FT controller.
"""

from __future__ import annotations

import time

import ray
from tests.e2e.ft.conftest import (
    E2eFaultInjector,
    FaultInjectorFactory,
    wait_for_mode_transition,
    wait_for_training_stable,
)

from miles.utils.ft.controller.types import ControllerMode


async def test_python_exception_auto_recovery(
    ft_controller_handle: ray.actor.ActorHandle,
    fault_injector: FaultInjectorFactory,
    target_node: str,
) -> None:
    # Step 1: Deploy fault injector to target node
    injector = E2eFaultInjector(
        injector_handle=fault_injector.deploy_to(node_id=target_node),
        target_node=target_node,
    )

    # Step 2: Wait for training to be stable
    await wait_for_training_stable(
        ft_controller_handle,
        n_iterations=3,
        timeout=180.0,
    )

    # Step 3: Inject Python exception via flag file
    t0 = time.monotonic()
    await injector.inject_python_exception()

    # Step 4: Wait for recovery (MONITORING → something else → back to MONITORING)
    status = await wait_for_mode_transition(
        ft_controller_handle,
        target_mode=ControllerMode.MONITORING,
        timeout=300.0,
    )
    assert status.mode == ControllerMode.MONITORING
    assert status.recovery is None

    # Step 5: Verify training resumes after recovery
    await wait_for_training_stable(
        ft_controller_handle,
        n_iterations=5,
        timeout=300.0,
    )

    assert time.monotonic() - t0 < 300.0
