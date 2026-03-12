"""Semi-E2E: hang detection — stale iteration, monitoring timeout, full recovery cycle."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

import pytest

from miles.utils.ft.controller.detectors.core.training_crash import TrainingCrashDetector
from miles.utils.ft.controller.types import ControllerMode
from tests.fast.utils.ft.integration.conftest import (
    FAST_TIMEOUT,
    LONG_RECOVERY_TIMEOUT,
    RECOVERY_TIMEOUT,
)
from tests.fast.utils.ft.testbed import MilesTestbed, TestbedNodeConfig
from tests.fast.utils.ft.utils.controller_fakes import FastHangDetector

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.anyio,
]


# ------------------------------------------------------------------
# 1. test_stale_iteration_triggers_recovery
# ------------------------------------------------------------------


async def test_stale_iteration_triggers_recovery(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Worker iteration stalls -> FastHangDetector fires -> RECOVERY mode."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[FastHangDetector(timeout_seconds=3.0)],
    )

    # Step 1: inject hang — workers stop advancing iterations
    await testbed.inject_hang()

    # Step 2: wait for FastHangDetector to fire (3s timeout + margin)
    await asyncio.sleep(4.0)

    # Step 3: verify controller entered RECOVERY
    status = await testbed.wait_for_mode(
        mode=ControllerMode.RECOVERY,
        timeout=FAST_TIMEOUT,
    )
    assert status.mode == ControllerMode.RECOVERY


# ------------------------------------------------------------------
# 2. test_crash_during_hung_monitoring_escalates
# ------------------------------------------------------------------


async def test_crash_during_hung_monitoring_escalates(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Crash -> MonitoringProgress -> crash again -> escalates to StopTimeDiagnostics.

    Uses slow step_interval so MonitoringProgress lasts long enough to inject a second crash.
    """
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector()],
        step_interval=2.0,
    )

    # Step 1: crash -> enters recovery -> wait for MonitoringProgress phase
    await testbed.crash_training()
    await testbed.wait_for_recovery_phase(
        phase="MonitoringProgressSt",
        timeout=RECOVERY_TIMEOUT,
    )

    # Step 2: crash during MonitoringProgress -> recovery completes (may escalate)
    await testbed.crash_training()
    await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=RECOVERY_TIMEOUT,
    )


# ------------------------------------------------------------------
# 3. test_hang_full_recovery_and_resume
# ------------------------------------------------------------------


async def test_hang_full_recovery_and_resume(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Hang -> detect -> full recovery -> training resumes with new workers.

    After inject_hang(), workers stop advancing. FastHangDetector fires.
    Controller enters recovery, stops training, starts new workers (not hung).
    New workers advance normally -> MonitoringProgress succeeds -> MONITORING.
    """
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[FastHangDetector(timeout_seconds=3.0)],
    )

    # Step 1: inject hang
    await testbed.inject_hang()

    # Step 2: wait for detection and recovery to complete
    status = await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=LONG_RECOVERY_TIMEOUT,
    )
    assert status.mode == ControllerMode.MONITORING

    # Step 3: verify training resumes with 3 more iterations
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)


# ------------------------------------------------------------------
# 4. test_monitoring_progress_timeout
# ------------------------------------------------------------------


async def test_monitoring_progress_timeout(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """step_interval=999 + monitoring_timeout_seconds=5 -> StopTimeDiagnostics.

    After crash, recovery restarts training but workers advance every 999s,
    so MonitoringProgress sees no progress and times out.
    """
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector()],
        step_interval=999.0,
        monitoring_timeout_seconds=5,
        monitoring_success_iterations=100,
    )

    # Step 1: crash -> recovery -> MonitoringProgress but never progresses
    await testbed.crash_training()

    # Step 2: wait for StopTimeDiagnostics phase due to monitoring timeout
    deadline = time.monotonic() + RECOVERY_TIMEOUT
    while time.monotonic() < deadline:
        status = await testbed.get_status()
        if status.recovery is not None and status.recovery.phase == "StopTimeDiagnosticsSt":
            break
        await asyncio.sleep(0.5)
    else:
        raise TimeoutError(
            f"MonitoringProgress did not timeout to StopTimeDiagnostics within {RECOVERY_TIMEOUT}s"
        )


# ------------------------------------------------------------------
# 5. test_monitoring_success_iterations_zero
# ------------------------------------------------------------------


async def test_monitoring_success_iterations_zero(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """monitoring_success_iterations=0 -> recovery completes almost instantly with new run_id."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector()],
        monitoring_success_iterations=0,
    )

    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)
    old_run_id = (await testbed.get_status()).active_run_id

    # Step 1: crash -> recovery completes almost instantly (0 iterations needed)
    await testbed.crash_training()

    # Step 2: wait for new run_id and MONITORING mode
    deadline = time.monotonic() + RECOVERY_TIMEOUT
    while time.monotonic() < deadline:
        status = await testbed.get_status()
        if status.active_run_id != old_run_id and status.mode == ControllerMode.MONITORING:
            break
        await asyncio.sleep(0.5)
    else:
        raise TimeoutError(
            f"Recovery did not complete within {RECOVERY_TIMEOUT}s: "
            f"run_id changed={status.active_run_id != old_run_id}, mode={status.mode}"
        )

    assert status.mode == ControllerMode.MONITORING

    # Step 3: should have completed without StopTimeDiagnostics (immediate success)
    assert status.recovery is None or status.recovery.phase != "StopTimeDiagnosticsSt"


# ------------------------------------------------------------------
# 6. test_monitoring_timeout_zero
# ------------------------------------------------------------------


async def test_monitoring_timeout_zero(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """monitoring_timeout_seconds=0, step_interval=999 -> immediate StopTimeDiagnostics.

    MonitoringProgress times out immediately because timeout is 0 and workers
    never advance (step_interval=999s).
    """
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector()],
        step_interval=999.0,
        monitoring_timeout_seconds=0,
        monitoring_success_iterations=999,
    )

    # Step 1: crash -> recovery
    await testbed.crash_training()

    # Step 2: MonitoringProgress should time out immediately -> StopTimeDiagnostics
    deadline = time.monotonic() + RECOVERY_TIMEOUT
    while time.monotonic() < deadline:
        status = await testbed.get_status()
        if status.recovery is not None and status.recovery.phase == "StopTimeDiagnosticsSt":
            break
        await asyncio.sleep(0.5)
    else:
        raise TimeoutError(
            f"monitoring_timeout_seconds=0 did not trigger StopTimeDiagnostics within {RECOVERY_TIMEOUT}s"
        )
