"""E2E Scenario 4: MFU decline detection via GPU stress.

Validates MfuDeclineDetector when GPU compute is contended:
  1. Wait for baseline MFU to stabilize
  2. Start GPU stress workload on target node
  3. MFU drops → MfuDeclineDetector triggers
  4. Outcome depends on GPU temperature correlation:
     - MARK_BAD_AND_RESTART: target node identified via temperature
     - NOTIFY_HUMAN: temperature correlation insufficient (acceptable)
  5. Cleanup GPU stress

See 9-testing.md §5.5: GPU stress may not always cause temperature rise,
so both MARK_BAD and NOTIFY outcomes are acceptable.
"""
from __future__ import annotations

import asyncio
import logging
import time

import pytest
import ray

from tests.e2e.ft.conftest import (
    FaultInjectorFactory,
    FtSystem,
    wait_for_recovery_complete,
    wait_for_training_stable,
)

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.timeout(900),
]


@pytest.mark.asyncio
async def test_mfu_decline_detection(
    ft_system: FtSystem,
    fault_injector: FaultInjectorFactory,
    target_node: str,
) -> None:
    controller = ft_system.controller

    # Wait for MFU baseline to stabilize
    await wait_for_training_stable(
        controller=controller,
        mini_wandb=ft_system.mini_wandb,
        n_iterations=20,
        timeout=600.0,
    )

    # Deploy fault injector and start GPU stress
    injector = fault_injector.deploy_to(node_id=target_node)
    stress_pid = ray.get(injector.start_gpu_stress.remote())
    logger.info("gpu_stress_started pid=%d node=%s", stress_pid, target_node)

    try:
        # Wait for MfuDeclineDetector to trigger
        # consecutive_decline_threshold × iteration_time + buffer
        timeout = 600.0
        poll_interval = 10.0
        deadline = time.monotonic() + timeout
        detected = False

        while time.monotonic() < deadline:
            status = controller.get_status()
            if status["mode"] == "recovery":
                detected = True
                logger.info("mfu_decline_detected status=%s", status)
                break
            await asyncio.sleep(poll_interval)

        assert detected, (
            f"MfuDeclineDetector did not trigger within {timeout}s"
        )

        # Both outcomes are acceptable
        final_status = await wait_for_recovery_complete(
            controller=controller,
            timeout=300.0,
        )
        assert final_status["mode"] == "monitoring"

        bad_nodes = await ft_system.node_manager.get_bad_nodes()
        evicted = target_node in bad_nodes or any(
            target_node in str(n) for n in bad_nodes
        )

        if evicted:
            logger.info("mfu_decline_evicted node=%s (temperature correlated)", target_node)
        else:
            logger.info("mfu_decline_notified (no temperature correlation)")

    finally:
        ray.get(injector.stop_gpu_stress.remote(pid=stress_pid))

    # If node was evicted, verify training recovers on remaining nodes
    if evicted:
        await wait_for_training_stable(
            controller=controller,
            mini_wandb=ft_system.mini_wandb,
            n_iterations=10,
            timeout=300.0,
        )
