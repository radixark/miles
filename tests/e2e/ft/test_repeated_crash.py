"""E2E Scenario 2: Repeated crash → reattempt fails → DIAGNOSING.

Validates the escalation path:
  1. Kill training process (first time)
  2. Controller reattempts → enters MONITORING
  3. Kill training process again during MONITORING
  4. Controller detects repeated failure → enters DIAGNOSING
  5. With StubDiagnosticScheduler → all diagnostics pass → NOTIFY_HUMAN
"""
from __future__ import annotations

import asyncio

import pytest
import ray

from tests.e2e.ft.conftest import (
    FaultInjectorFactory,
    FtSystem,
    wait_for_recovery_complete,
    wait_for_recovery_phase,
    wait_for_training_stable,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.timeout(600),
]


@pytest.mark.asyncio
async def test_repeated_crash_enters_diagnosing(
    ft_system: FtSystem,
    fault_injector: FaultInjectorFactory,
    target_node: str,
) -> None:
    controller = ft_system.controller

    await wait_for_training_stable(
        controller=controller,
        mini_wandb=ft_system.mini_wandb,
        n_iterations=5,
        timeout=300.0,
    )

    injector = fault_injector.deploy_to(node_id=target_node)

    # First kill
    procs = ray.get(injector.find_training_processes.remote())
    assert len(procs) > 0
    ray.get(injector.kill_process.remote(pid=procs[0]["pid"], sig=9))

    # Wait for Controller to enter MONITORING (reattempt succeeded, now watching)
    await wait_for_recovery_phase(
        controller=controller,
        phase="monitoring",
        timeout=180.0,
    )

    # Second kill during MONITORING phase
    await asyncio.sleep(5.0)
    procs = ray.get(injector.find_training_processes.remote())
    if procs:
        ray.get(injector.kill_process.remote(pid=procs[0]["pid"], sig=9))

    # Wait for Controller to enter DIAGNOSING
    status = await wait_for_recovery_phase(
        controller=controller,
        phase="diagnosing",
        timeout=180.0,
    )
    assert status["recovery_phase"] == "diagnosing"

    # Wait for recovery to complete (with stub scheduler → NOTIFY → DONE)
    final_status = await wait_for_recovery_complete(
        controller=controller,
        timeout=300.0,
    )
    assert final_status["mode"] == "monitoring"
