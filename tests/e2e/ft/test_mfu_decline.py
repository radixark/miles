"""E2E: MFU decline detection via GPU stress.

GPU stress may not always cause temperature rise, so both outcomes
(eviction vs notify) are valid.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest
import ray
from miles.utils.ft.models import ControllerMode, RecoveryPhase
from miles.utils.ft.platform.k8s_node_manager import K8sNodeManager
from tests.e2e.ft.conftest import (
    FaultInjectorFactory,
    assert_phase_path_contains,
    get_status,
    wait_for_recovery_complete,
    wait_for_training_stable,
)

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.timeout(900),
]


async def test_mfu_decline_detection(
    ft_controller_handle: ray.actor.ActorHandle,
    fault_injector: FaultInjectorFactory,
    target_node: str,
    _cleanup_node_manager: K8sNodeManager,
) -> None:
    # Step 1: Wait for baseline MFU to stabilize
    await wait_for_training_stable(
        handle=ft_controller_handle,
        n_iterations=20,
        timeout=600.0,
    )

    # Step 2: Start GPU stress workload
    injector = fault_injector.deploy_to(node_id=target_node)
    stress_pid = ray.get(injector.start_gpu_stress.remote())
    logger.info("gpu_stress_started pid=%d node=%s", stress_pid, target_node)

    try:
        # Step 3: Wait for MfuDeclineDetector to trigger
        timeout = 600.0
        poll_interval = 10.0
        deadline = time.monotonic() + timeout
        detected = False

        while time.monotonic() < deadline:
            status = get_status(ft_controller_handle)
            if status.mode == ControllerMode.RECOVERY:
                detected = True
                logger.info("mfu_decline_detected status=%s", status)
                break
            await asyncio.sleep(poll_interval)

        assert detected, f"MfuDeclineDetector did not trigger within {timeout}s"

        final_status = await wait_for_recovery_complete(
            handle=ft_controller_handle,
            timeout=300.0,
        )
        assert final_status.mode == ControllerMode.MONITORING
        assert final_status.phase_history is not None

        bad_nodes = await _cleanup_node_manager.get_bad_nodes()
        evicted = target_node in bad_nodes or any(target_node in str(n) for n in bad_nodes)

    finally:
        ray.get(injector.stop_gpu_stress.remote(pid=stress_pid))

    # Step 4: Verify outcome (eviction or notify, both valid)
    if evicted:
        logger.info("mfu_decline_evicted node=%s (temperature correlated)", target_node)
        assert_phase_path_contains(final_status, [
            RecoveryPhase.EVICT_AND_RESTART,
            RecoveryPhase.DONE,
        ])

        await wait_for_training_stable(
            handle=ft_controller_handle,
            n_iterations=10,
            timeout=300.0,
        )
    else:
        logger.info("mfu_decline_notified (no temperature correlation)")
        assert_phase_path_contains(final_status, [
            RecoveryPhase.NOTIFY,
            RecoveryPhase.DONE,
        ])

        post_bad = await _cleanup_node_manager.get_bad_nodes()
        assert target_node not in post_bad, (
            f"Notify path should not mark nodes bad, but {target_node} found in {post_bad}"
        )
