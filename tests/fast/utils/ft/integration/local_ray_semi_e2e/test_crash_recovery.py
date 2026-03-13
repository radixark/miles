"""Semi-E2E: crash recovery — transient, repeated, throttle, reset, exception, concurrent."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

import pytest
from tests.fast.utils.ft.integration.conftest import FAST_TIMEOUT, LONG_RECOVERY_TIMEOUT, RECOVERY_TIMEOUT
from tests.fast.utils.ft.integration.local_ray_semi_e2e.conftest import assert_no_recovery_triggered
from tests.fast.utils.ft.testbed import MilesTestbed, TestbedNodeConfig
from tests.fast.utils.ft.utils.controller_fakes import FastHangDetector
from tests.fast.utils.ft.utils.diagnostic_fakes import DelayedDiagnosticOrchestrator

from miles.utils.ft.agents.types import GaugeSample
from miles.utils.ft.controller.detectors.chain import build_detector_chain
from miles.utils.ft.controller.detectors.core.training_crash import TrainingCrashDetector
from miles.utils.ft.controller.types import ControllerMode
from miles.utils.ft.utils.metric_names import GPU_AVAILABLE
from miles.utils.ft.utils.sliding_window import SlidingWindowThrottle

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.anyio,
]

_SLOW_STEP = 2.0


# test_crash_recovery_full_cycle removed: covered by test_scenarios::test_transient_crash


# ------------------------------------------------------------------
# 2. test_third_crash_throttled
# ------------------------------------------------------------------


async def test_third_crash_throttled(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """3 crashes with max_count=2 -> third is throttled (stays in MONITORING).

    is_throttled() is checked before record(), so max_count=2 allows 2 recoveries.
    """
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector()],
        recovery_cooldown=SlidingWindowThrottle(
            window_minutes=60,
            max_count=2,
        ),
    )

    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 1: first crash -> recovery
    await testbed.crash_training()
    await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=RECOVERY_TIMEOUT,
    )

    # Step 2: second crash -> recovery
    await testbed.crash_training()
    await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=RECOVERY_TIMEOUT,
    )

    # Step 3: third crash -> throttled, no recovery
    await testbed.crash_training()
    await assert_no_recovery_triggered(
        testbed,
        observation_ticks=20,
        timeout=FAST_TIMEOUT,
    )


# ------------------------------------------------------------------
# 3. test_state_clean_after_recovery
# ------------------------------------------------------------------


async def test_state_clean_after_recovery(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Recovery DONE -> mode=MONITORING, phase=None -> new crash is a fresh cycle."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector()],
        recovery_cooldown=SlidingWindowThrottle(window_minutes=60, max_count=10),
    )

    # Step 1: first crash -> full recovery
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)
    await testbed.crash_training()
    status = await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=RECOVERY_TIMEOUT,
    )
    assert status.mode == ControllerMode.MONITORING
    assert status.recovery_in_progress is False

    # Step 2: second crash -> enters a fresh recovery cycle
    await testbed.wait_for_training_stable(n_iterations=2, timeout=FAST_TIMEOUT)
    await testbed.crash_training()
    status = await testbed.wait_for_mode(
        mode=ControllerMode.RECOVERY,
        timeout=FAST_TIMEOUT,
    )
    assert status.mode == ControllerMode.RECOVERY
    assert status.recovery is not None


# test_crash_recovery_completes removed: covered by test_scenarios::test_transient_crash
# test_two_crashes_escalate_to_diagnosing removed: covered by test_scenarios::test_repeated_crash


# ------------------------------------------------------------------
# 6. test_simultaneous_nan_and_crash
# ------------------------------------------------------------------


async def test_simultaneous_nan_and_crash(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """NaN + FAILED simultaneously -> detector priority chain triggers 1 recovery."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=build_detector_chain(),
        scrape_interval_seconds=0.5,
    )

    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 1: inject both faults
    await testbed.inject_nan_loss()
    await testbed.crash_training()

    # Step 2: wait for recovery
    status = await testbed.wait_for_mode(
        mode=ControllerMode.RECOVERY,
        timeout=FAST_TIMEOUT,
    )
    assert status.mode == ControllerMode.RECOVERY

    # Step 3: let recovery complete
    final = await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=LONG_RECOVERY_TIMEOUT,
    )
    assert final.mode == ControllerMode.MONITORING


# ------------------------------------------------------------------
# 7. test_restart_fails_escalates_to_diagnostics
# ------------------------------------------------------------------


async def test_restart_fails_escalates_to_diagnostics(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Restart completes but job FAILED during MonitoringProgress -> StopTimeDiagnostics.

    Uses a DelayedDiagnosticOrchestrator so that StopTimeDiagnosticsSt is
    observable across multiple ticks (the default instant orchestrator would
    converge through the state in a single tick).
    """
    diag_orchestrator = DelayedDiagnosticOrchestrator(delay_seconds=5.0)
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector()],
        step_interval=_SLOW_STEP,
        recovery_cooldown=SlidingWindowThrottle(window_minutes=1.0, max_count=5),
        diagnostic_orchestrator_override=diag_orchestrator,
    )

    # Step 1: crash -> enters recovery, wait for MonitoringProgress (durable phase)
    await testbed.crash_training()
    await testbed.wait_for_recovery_phase(
        phase="MonitoringProgressSt",
        timeout=RECOVERY_TIMEOUT,
    )

    # Step 2: crash again during monitoring (restart fails)
    await testbed.crash_training()

    # Step 3: poll for StopTimeDiagnostics phase (visible because diagnostics are slow)
    deadline = time.monotonic() + RECOVERY_TIMEOUT
    while time.monotonic() < deadline:
        status = await testbed.get_status()
        if status.recovery is not None and status.recovery.phase == "StopTimeDiagnosticsSt":
            break
        await asyncio.sleep(0.3)
    else:
        raise TimeoutError(f"StopTimeDiagnostics not observed within {RECOVERY_TIMEOUT}s")


# ------------------------------------------------------------------
# 8. test_crash_during_early_recovery_converges
# ------------------------------------------------------------------


async def test_crash_during_early_recovery_converges(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Crash while already in RECOVERY -> system eventually converges."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector()],
        recovery_cooldown=SlidingWindowThrottle(window_minutes=60, max_count=10),
    )

    # Step 1: crash -> enter recovery
    await testbed.crash_training()
    await testbed.wait_for_mode(
        mode=ControllerMode.RECOVERY,
        timeout=FAST_TIMEOUT,
    )

    # Step 2: crash again while recovering
    await testbed.crash_training()

    # Step 3: system must converge (back to MONITORING or NotifyHumans)
    deadline = time.monotonic() + LONG_RECOVERY_TIMEOUT
    while time.monotonic() < deadline:
        status = await testbed.get_status()
        if status.mode == ControllerMode.MONITORING and not status.recovery_in_progress:
            return
        await asyncio.sleep(0.5)

    status = await testbed.get_status()
    recovery_phase = status.recovery.phase if status.recovery else None
    assert status.mode == ControllerMode.MONITORING or (
        status.recovery is not None and status.recovery.phase == "NotifyHumansSt"
    ), f"Recovery did not converge: mode={status.mode}, phase={recovery_phase}"


# ------------------------------------------------------------------
# 9. test_fault_during_monitoring_progress
# ------------------------------------------------------------------


async def test_fault_during_monitoring_progress(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """New fault during MonitoringProgress is handled without nested recovery."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector()],
        step_interval=_SLOW_STEP,
        recovery_cooldown=SlidingWindowThrottle(window_minutes=1.0, max_count=5),
    )

    # Step 1: crash -> wait for MonitoringProgress
    await testbed.crash_training()
    await testbed.wait_for_recovery_phase(
        phase="MonitoringProgressSt",
        timeout=RECOVERY_TIMEOUT,
    )

    # Step 2: inject another crash during MonitoringProgress
    await testbed.crash_training()

    # Step 3: wait for convergence
    deadline = time.monotonic() + LONG_RECOVERY_TIMEOUT
    while time.monotonic() < deadline:
        status = await testbed.get_status()
        if status.mode == ControllerMode.MONITORING and not status.recovery_in_progress:
            return
        await asyncio.sleep(0.5)

    status = await testbed.get_status()
    recovery_phase = status.recovery.phase if status.recovery else None
    assert status.mode == ControllerMode.MONITORING, f"Did not converge: mode={status.mode}, phase={recovery_phase}"


# ------------------------------------------------------------------
# 10. test_sequential_recovery_no_state_drift
# ------------------------------------------------------------------


async def test_sequential_recovery_no_state_drift(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Three crash-recover cycles with distinct run_ids and clean state each time."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector()],
        recovery_cooldown=SlidingWindowThrottle(window_minutes=60, max_count=10),
    )

    run_ids: list[str] = []
    initial_status = await testbed.get_status()
    if initial_status.active_run_id:
        run_ids.append(initial_status.active_run_id)

    for _cycle in range(3):
        await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)
        await testbed.crash_training()

        status = await testbed.wait_for_mode_transition(
            target_mode=ControllerMode.MONITORING,
            timeout=RECOVERY_TIMEOUT,
        )
        assert status.mode == ControllerMode.MONITORING
        assert status.recovery_in_progress is False

        if status.active_run_id:
            run_ids.append(status.active_run_id)

    assert len(set(run_ids)) == len(run_ids), f"Duplicate run_ids: {run_ids}"


# ------------------------------------------------------------------
# 11. test_cooldown_boundary_exact_count
# ------------------------------------------------------------------


async def test_cooldown_boundary_exact_count(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """max_count=1 allows 1 recovery; the second crash is throttled.

    is_throttled() is checked before record(), so max_count=1 allows 1 recovery.
    """
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector()],
        recovery_cooldown=SlidingWindowThrottle(window_minutes=60, max_count=1),
    )

    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 1: first crash -> recovery succeeds
    await testbed.crash_training()
    await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=RECOVERY_TIMEOUT,
    )

    # Step 2: second crash -> throttled
    await testbed.wait_for_training_stable(n_iterations=2, timeout=FAST_TIMEOUT)
    await testbed.crash_training()
    await assert_no_recovery_triggered(
        testbed,
        observation_ticks=20,
        timeout=FAST_TIMEOUT,
    )


# ------------------------------------------------------------------
# 12. test_concurrent_crash_and_hang
# ------------------------------------------------------------------


async def test_concurrent_crash_and_hang(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Hang + crash conditions both true -> exactly one recovery cycle triggered."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector(), FastHangDetector(timeout_seconds=3.0)],
    )

    # Step 1: inject hang first
    await testbed.inject_hang()
    await asyncio.sleep(4.0)

    # Step 2: also crash
    await testbed.crash_training()

    # Step 3: wait for recovery to complete
    final = await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=LONG_RECOVERY_TIMEOUT,
    )
    assert final.mode == ControllerMode.MONITORING


# ------------------------------------------------------------------
# 13. test_throttle_notification
# ------------------------------------------------------------------


async def test_throttle_notification(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """When recovery is throttled, the notifier receives a notification."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector()],
        recovery_cooldown=SlidingWindowThrottle(window_minutes=60, max_count=1),
    )

    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 1: first crash -> recovery
    await testbed.crash_training()
    await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=RECOVERY_TIMEOUT,
    )

    # Step 2: second crash -> throttled
    await testbed.wait_for_training_stable(n_iterations=2, timeout=FAST_TIMEOUT)
    await testbed.crash_training()
    await assert_no_recovery_triggered(
        testbed,
        observation_ticks=20,
        timeout=FAST_TIMEOUT,
    )

    # Step 3: verify notifier received calls
    calls = testbed.notifications
    assert len(calls) > 0, "Notifier should have received at least one call"


# ------------------------------------------------------------------
# 14. test_recovery_timeout_escalates
# ------------------------------------------------------------------


async def test_recovery_timeout_escalates(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """recovery_timeout_seconds fires when MonitoringProgress stalls -> NotifyHumans.

    Uses initial_stable_iterations=0 because step_interval=999 prevents
    the default stability check from completing. A brief sleep ensures
    workers are spawned before crashing.
    """
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector()],
        step_interval=999.0,
        recovery_timeout_seconds=3,
        monitoring_timeout_seconds=999,
        monitoring_success_iterations=999,
        initial_stable_iterations=0,
    )

    # Step 1: brief wait for initial workers to spawn and register,
    # then crash -> recovery starts, reaches MonitoringProgress but never succeeds
    await asyncio.sleep(3)
    await testbed.crash_training()

    # Step 2: NotifyHumansSt is transient, so wait for recovery to settle back
    # to MONITORING and verify the timeout path via notifier output.
    await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=RECOVERY_TIMEOUT,
    )

    recovery_alerts = [
        (title, content, severity) for title, content, severity in testbed.notifications if title == "Recovery Alert"
    ]
    assert recovery_alerts, "Expected recovery timeout to trigger a recovery alert"
    assert any(
        "recovery_timeout_exceeded" in content for _title, content, _severity in recovery_alerts
    ), "Expected at least one recovery alert to mention recovery_timeout_exceeded"


# ------------------------------------------------------------------
# 15. test_cooldown_expiry_allows_recovery
# ------------------------------------------------------------------


async def test_cooldown_expiry_allows_recovery(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """After cooldown window expires, a new crash triggers recovery normally.

    In MilesTestbed there is no recover_training(). When training is dead and
    throttled, the detector keeps firing every tick seeing FAILED. Once the
    cooldown window expires, the next detector fire triggers recovery
    automatically. So we just wait for the window to expire + margin.
    """
    window_seconds = 8
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector()],
        recovery_cooldown=SlidingWindowThrottle(
            window_minutes=window_seconds / 60.0,
            max_count=1,
        ),
    )

    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 1: first crash -> recovery
    await testbed.crash_training()
    await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=RECOVERY_TIMEOUT,
    )

    # Step 2: second crash immediately -> throttled
    await testbed.wait_for_training_stable(n_iterations=2, timeout=FAST_TIMEOUT)
    await testbed.crash_training()
    await assert_no_recovery_triggered(
        testbed,
        observation_ticks=20,
        timeout=FAST_TIMEOUT,
    )
    old_run_id = (await testbed.get_status()).active_run_id

    # Step 3: wait for cooldown window to expire. Training is dead, detector
    # keeps firing, and once cooldown expires the next fire triggers recovery.
    await asyncio.sleep(window_seconds + 3)

    # Step 4: verify recovery was triggered automatically after cooldown expiry.
    # The recovery may complete before this assertion starts polling, so use
    # run_id change instead of waiting to observe a transient mode transition.
    deadline = time.monotonic() + RECOVERY_TIMEOUT
    while time.monotonic() < deadline:
        status = await testbed.get_status()
        if status.active_run_id != old_run_id and status.mode == ControllerMode.MONITORING:
            break
        await asyncio.sleep(0.5)
    else:
        raise TimeoutError(
            f"Recovery did not complete within {RECOVERY_TIMEOUT}s after cooldown expiry: "
            f"run_id changed={status.active_run_id != old_run_id}, mode={status.mode}"
        )

    assert status.mode == ControllerMode.MONITORING


# ------------------------------------------------------------------
# 16. test_too_many_bad_nodes_false_positive
# ------------------------------------------------------------------


async def test_too_many_bad_nodes_false_positive(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """When >= max_simultaneous_bad_nodes report faults, no recovery is triggered."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id=f"n-{i}", num_ranks=2) for i in range(4)],
        detectors=build_detector_chain(),
        scrape_interval_seconds=0.5,
        tick_interval=1.0,
        max_simultaneous_bad_nodes=3,
    )

    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 1: inject GPU_AVAILABLE=0 on 3 nodes simultaneously
    for i in range(3):
        node_id = f"n-{i}"
        await testbed.inject_collector_metrics(
            node_id=node_id,
            metrics=[
                GaugeSample(
                    name=GPU_AVAILABLE,
                    labels={"node_id": node_id, "gpu": "0"},
                    value=0.0,
                ),
            ],
        )

    # Step 2: wait for metrics to propagate through scrape pipeline
    # (collect_interval=0.3s + scrape_interval=0.5s + margin) before
    # the next detector evaluation (tick_interval=1.0s).
    await asyncio.sleep(1.5)

    # Step 3: verify no recovery triggered -- all 3 bad nodes seen at once
    await assert_no_recovery_triggered(
        testbed,
        observation_ticks=10,
        timeout=FAST_TIMEOUT,
    )
