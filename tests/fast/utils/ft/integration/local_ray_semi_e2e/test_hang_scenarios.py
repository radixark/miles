"""Semi-E2E: hang scenarios — real process hang detection with alive-but-stalled workers."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import pytest

from miles.utils.ft.controller.types import ControllerMode
from tests.fast.utils.ft.integration.conftest import (
    FAST_TIMEOUT,
    LONG_RECOVERY_TIMEOUT,
)
from tests.fast.utils.ft.testbed import MilesTestbed, TestbedNodeConfig
from tests.fast.utils.ft.utils.controller_fakes import FastHangDetector

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.anyio,
]


# ------------------------------------------------------------------
# 1. test_hang_detected_workers_alive_but_stalled
# ------------------------------------------------------------------


async def test_hang_detected_workers_alive_but_stalled(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Workers respond to ping but stop advancing iterations -> HangDetector fires -> RECOVERY -> resume.

    This exercises real Ray actor hang detection: unlike old E2EEnv where "hang"
    was a flag on fake TrainingWorkerActor with state_actor, MilesTestbed workers
    are real Ray actor processes that stay alive but stop doing work.
    """
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[FastHangDetector(timeout_seconds=3.0)],
    )

    # Step 1: confirm training is progressing normally
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 2: capture baseline iteration count before injecting hang
    status_before = await testbed.get_status()
    iteration_before: int | None = status_before.latest_iteration
    assert iteration_before is not None and iteration_before > 0

    # Step 3: inject hang — workers stay alive but stop advancing iterations
    await testbed.inject_hang()

    # Step 4: verify workers are still alive (respond to ping) but iterations are frozen
    await asyncio.sleep(1.0)
    alive = await testbed.train_group.all_alive()
    assert alive, "Workers should still be alive (responding to ping) after inject_hang"

    status_after_hang = await testbed.get_status()
    iteration_after_hang: int | None = status_after_hang.latest_iteration
    assert iteration_after_hang is not None
    assert iteration_after_hang <= iteration_before + 1, (
        f"Iterations should be frozen after hang: before={iteration_before}, "
        f"after={iteration_after_hang}"
    )

    # Step 5: wait for FastHangDetector to fire (3s timeout + margin) -> RECOVERY
    status = await testbed.wait_for_mode(
        mode=ControllerMode.RECOVERY,
        timeout=FAST_TIMEOUT,
    )
    assert status.mode == ControllerMode.RECOVERY

    # Step 6: wait for full recovery to complete -> back to MONITORING
    final = await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=LONG_RECOVERY_TIMEOUT,
    )
    assert final.mode == ControllerMode.MONITORING

    # Step 7: verify training resumes with new workers advancing iterations
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)
