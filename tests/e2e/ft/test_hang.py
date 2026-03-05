"""E2E Scenario 3: Training hang via SIGSTOP → detection → recovery.

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
from miles.utils.ft.models import ControllerMode
from tests.e2e.ft.conftest import FaultInjectorFactory, FtSystem, wait_for_recovery_complete, wait_for_training_stable

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.timeout(900),
]


async def test_hang_detection_and_recovery(
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

    # Find and freeze a training process
    procs = ray.get(injector.find_training_processes.remote())
    assert len(procs) > 0, f"No training processes found on {target_node}"

    target_pid = procs[0]["pid"]
    t_inject = time.monotonic()
    ray.get(injector.stop_process.remote(pid=target_pid))
    logger.info("SIGSTOP sent to pid=%d on node=%s", target_pid, target_node)

    # Wait for HangDetector to trigger ENTER_RECOVERY
    # HangDetector default threshold is ~5 min, add buffer
    status = await wait_for_recovery_complete(
        controller=controller,
        timeout=720.0,
        poll_interval=10.0,
    )

    t_detect = time.monotonic() - t_inject
    logger.info("hang_detected_and_recovered t_detect=%.1fs", t_detect)

    assert status.mode == ControllerMode.MONITORING

    # Wait for training to stabilize post-recovery
    await wait_for_training_stable(
        controller=controller,
        mini_wandb=ft_system.mini_wandb,
        n_iterations=10,
        timeout=300.0,
    )
