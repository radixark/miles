"""E2E negative test: no false positives during normal training."""

from __future__ import annotations

import asyncio
import logging

import ray
from miles.utils.ft.models import ControllerMode
from tests.e2e.ft.conftest import get_iteration_count, get_status

logger = logging.getLogger(__name__)

# 50 iterations provides ~95% confidence to detect a per-iteration false-positive
# rate >= 6%, but cannot reliably detect rates around 1%.  Increase to ~300 for
# 95% confidence at the 1% level.  Kept low to bound E2E test runtime.
_TARGET_ITERATIONS = 50
_POLL_INTERVAL = 5.0


async def test_no_false_positive_during_normal_training(
    ft_controller_handle: ray.actor.ActorHandle,
) -> None:
    """Controller should not trigger recovery when training runs normally."""
    # Step 1: Run 50 iterations with no fault injection
    baseline = get_iteration_count(ft_controller_handle)
    recovery_triggered = False

    while True:
        # Step 2: Check controller never enters RECOVERY
        status = get_status(ft_controller_handle)

        if status.mode == ControllerMode.RECOVERY:
            recovery_triggered = True
            logger.error(
                "false_positive_detected status=%s iteration=%d",
                status,
                get_iteration_count(ft_controller_handle),
            )
            break

        current = get_iteration_count(ft_controller_handle)
        progress = current - baseline
        if progress >= _TARGET_ITERATIONS:
            logger.info(
                "no_false_positive iterations=%d/%d",
                progress, _TARGET_ITERATIONS,
            )
            break

        await asyncio.sleep(_POLL_INTERVAL)

    assert not recovery_triggered, (
        f"Controller entered recovery during normal training at iteration "
        f"{get_iteration_count(ft_controller_handle)}: {get_status(ft_controller_handle)}"
    )
