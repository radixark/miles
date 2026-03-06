"""E2E: Training hang via SIGSTOP → detection → recovery.

Validates hang detection when iteration stops progressing:
  1. Send SIGSTOP to freeze a training process
  2. HangDetector triggers after iteration stalls for N minutes
  3. Controller enters ENTER_RECOVERY → restarts training
  4. Training resumes from checkpoint and stabilizes

This is the slowest E2E test due to the hang detection timeout (~5-10 min).
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.timeout(900),
]


async def test_hang_detection_and_recovery(
    ft_controller_handle: ray.actor.ActorHandle,
    fault_injector: FaultInjectorFactory,
    target_node: str,
) -> None:
    await wait_for_training_stable(
        handle=ft_controller_handle,
        n_iterations=5,
        timeout=300.0,
    )

    injector = fault_injector.deploy_to(node_id=target_node)

    procs = ray.get(injector.find_training_processes.remote())
    assert len(procs) > 0, f"No training processes found on {target_node}"

    target_pid = procs[0]["pid"]
    t_inject = time.monotonic()
    ray.get(injector.stop_process.remote(pid=target_pid))
    logger.info("SIGSTOP sent to pid=%d on node=%s", target_pid, target_node)

    status = await wait_for_recovery_complete(
        handle=ft_controller_handle,
        timeout=720.0,
        poll_interval=10.0,
    )

    t_detect = time.monotonic() - t_inject
    logger.info("hang_detected_and_recovered t_detect=%.1fs", t_detect)

    assert status.mode == ControllerMode.MONITORING

    final_status = get_status(ft_controller_handle)
    assert_phase_path_contains(final_status, [
        RecoveryPhase.CHECK_ALERTS,
        RecoveryPhase.REATTEMPTING,
        RecoveryPhase.MONITORING,
        RecoveryPhase.DONE,
    ])

    await wait_for_training_stable(
        handle=ft_controller_handle,
        n_iterations=10,
        timeout=300.0,
    )
