"""Semi-E2E: hardware faults — GPU lost, NaN loss, XID, disk space, MFU, fault during recovery."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

import pytest
from tests.fast.utils.ft.integration.conftest import FAST_TIMEOUT, LONG_RECOVERY_TIMEOUT, RECOVERY_TIMEOUT
from tests.fast.utils.ft.integration.local_ray_semi_e2e.conftest import assert_no_recovery_triggered
from tests.fast.utils.ft.testbed import MilesTestbed, TestbedNodeConfig

from miles.utils.ft.agents.types import GaugeSample
from miles.utils.ft.controller.detectors.chain import build_detector_chain
from miles.utils.ft.controller.detectors.core.gpu_fault import GpuFaultDetector
from miles.utils.ft.controller.detectors.core.mfu_decline import MfuDeclineDetector, MfuDeclineDetectorConfig
from miles.utils.ft.controller.detectors.core.training_crash import TrainingCrashDetector
from miles.utils.ft.controller.types import ControllerMode
from miles.utils.ft.utils.metric_names import (
    GPU_AVAILABLE,
    NODE_FILESYSTEM_AVAIL_BYTES,
    XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL,
)
from miles.utils.ft.utils.sliding_window import SlidingWindowThrottle

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.anyio,
]


# ------------------------------------------------------------------
# 1. test_gpu_lost_triggers_recovery
# ------------------------------------------------------------------


async def test_gpu_lost_triggers_recovery(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """GPU_AVAILABLE=0 on a node triggers GpuFaultDetector, recovery completes back to MONITORING."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=build_detector_chain(),
        scrape_interval_seconds=0.5,
    )

    # Step 1: verify training is stable
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)
    old_run_id = (await testbed.get_status()).active_run_id

    # Step 2: inject GPU unavailable metric
    await testbed.inject_collector_metrics(
        node_id="n-0",
        metrics=[
            GaugeSample(
                name=GPU_AVAILABLE,
                labels={"node_id": "n-0", "gpu": "0"},
                value=0.0,
            ),
        ],
    )

    # Step 3: wait for recovery to complete (run_id changes + back to MONITORING)
    deadline = time.monotonic() + LONG_RECOVERY_TIMEOUT
    while time.monotonic() < deadline:
        status = await testbed.get_status()
        if status.active_run_id != old_run_id and status.mode == ControllerMode.MONITORING:
            break
        await asyncio.sleep(0.5)
    else:
        raise TimeoutError(
            f"Recovery did not complete within {LONG_RECOVERY_TIMEOUT}s: "
            f"run_id changed={status.active_run_id != old_run_id}, mode={status.mode}"
        )

    assert status.mode == ControllerMode.MONITORING


# ------------------------------------------------------------------
# 2. test_nan_loss_triggers_recovery
# ------------------------------------------------------------------


async def test_nan_loss_triggers_recovery(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Injecting NaN loss triggers NanLossDetector and enters RECOVERY."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=build_detector_chain(),
        scrape_interval_seconds=0.5,
    )

    # Step 1: verify training is stable
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 2: inject NaN loss
    await testbed.inject_nan_loss()

    # Step 3: wait for recovery mode
    status = await testbed.wait_for_mode(
        mode=ControllerMode.RECOVERY,
        timeout=FAST_TIMEOUT,
    )
    assert status.mode == ControllerMode.RECOVERY


# ------------------------------------------------------------------
# 3. test_hardware_fault_during_recovery_escalates
# ------------------------------------------------------------------


async def test_hardware_fault_during_recovery_escalates(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Crash enters RECOVERY, then GPU_AVAILABLE=0 escalates to eviction; recovery completes."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=build_detector_chain(),
        scrape_interval_seconds=0.5,
    )

    # Step 1: verify training is stable
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 2: crash to enter recovery
    await testbed.crash_training()
    await testbed.wait_for_mode(
        mode=ControllerMode.RECOVERY,
        timeout=FAST_TIMEOUT,
    )

    # Step 3: while in recovery, inject hardware fault
    await testbed.inject_collector_metrics(
        node_id="n-0",
        metrics=[
            GaugeSample(
                name=GPU_AVAILABLE,
                labels={"node_id": "n-0", "gpu": "0"},
                value=0.0,
            ),
        ],
    )

    # Step 4: wait for recovery to complete back to MONITORING
    await testbed.wait_for_mode(
        mode=ControllerMode.MONITORING,
        timeout=LONG_RECOVERY_TIMEOUT,
    )


# ------------------------------------------------------------------
# 4. test_too_many_dynamic_bad_nodes_during_recovery_aborts
# ------------------------------------------------------------------


async def test_too_many_dynamic_bad_nodes_during_recovery_aborts(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """4 nodes, crash enters RECOVERY, GPU_AVAILABLE=0 on 3 nodes aborts recovery to MONITORING."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id=f"n-{i}", num_ranks=2) for i in range(4)],
        detectors=build_detector_chain(),
        scrape_interval_seconds=0.5,
        max_simultaneous_bad_nodes=3,
    )

    # Step 1: verify training is stable
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 2: crash to enter recovery
    await testbed.crash_training()
    await testbed.wait_for_mode(
        mode=ControllerMode.RECOVERY,
        timeout=FAST_TIMEOUT,
    )

    # Step 3: during recovery, inject GPU_AVAILABLE=0 on 3 nodes
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

    # Step 4: recovery should abort, returning to MONITORING
    deadline = time.monotonic() + RECOVERY_TIMEOUT
    while time.monotonic() < deadline:
        status = await testbed.get_status()
        if status.mode == ControllerMode.MONITORING and not status.recovery_in_progress:
            return
        await asyncio.sleep(0.5)

    status = await testbed.get_status()
    recovery_phase = status.recovery.phase if status.recovery else None
    assert (
        status.mode == ControllerMode.MONITORING
    ), f"Recovery did not abort: mode={status.mode}, phase={recovery_phase}"


# ------------------------------------------------------------------
# 5. test_critical_fault_during_recovery_merges_bad_nodes
# ------------------------------------------------------------------


async def test_critical_fault_during_recovery_merges_bad_nodes(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """GPU fault on node-0 enters RECOVERY, then GPU fault on node-1 merges; both evicted."""
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="n-0", num_ranks=2),
            TestbedNodeConfig(node_id="n-1", num_ranks=2),
            TestbedNodeConfig(node_id="n-2", num_ranks=2),
        ],
        detectors=build_detector_chain(),
        scrape_interval_seconds=0.5,
    )

    # Step 1: verify training is stable
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 2: inject GPU fault on node-0 to trigger recovery
    await testbed.inject_collector_metrics(
        node_id="n-0",
        metrics=[
            GaugeSample(
                name=GPU_AVAILABLE,
                labels={"node_id": "n-0", "gpu": "0"},
                value=0.0,
            ),
        ],
    )

    await testbed.wait_for_mode(
        mode=ControllerMode.RECOVERY,
        timeout=FAST_TIMEOUT,
    )

    # Step 3: during recovery, inject GPU fault on node-1
    await testbed.inject_collector_metrics(
        node_id="n-1",
        metrics=[
            GaugeSample(
                name=GPU_AVAILABLE,
                labels={"node_id": "n-1", "gpu": "0"},
                value=0.0,
            ),
        ],
    )

    # Step 4: wait for recovery to complete with both nodes evicted
    await testbed.wait_for_mode(
        mode=ControllerMode.MONITORING,
        timeout=LONG_RECOVERY_TIMEOUT,
    )


# ------------------------------------------------------------------
# 6. test_different_fault_types_share_throttle
# ------------------------------------------------------------------


async def test_different_fault_types_share_throttle(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Crash recovery then NaN injection shares the same cooldown window; NaN is throttled.

    is_throttled() is checked before record(), so max_count=1 allows exactly
    1 recovery before the next fault is throttled.
    """
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=build_detector_chain(),
        scrape_interval_seconds=0.5,
        recovery_cooldown=SlidingWindowThrottle(window_minutes=60, max_count=1),
    )

    # Step 1: verify training is stable
    await testbed.wait_for_training_stable(n_iterations=3, timeout=RECOVERY_TIMEOUT)

    # Step 2: crash triggers first recovery
    await testbed.crash_training()
    await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=LONG_RECOVERY_TIMEOUT,
    )

    # Step 3: NaN should be throttled (shared window, max_count=2)
    await testbed.wait_for_training_stable(n_iterations=2, timeout=RECOVERY_TIMEOUT)
    await testbed.inject_nan_loss()
    await assert_no_recovery_triggered(
        testbed,
        observation_ticks=20,
        timeout=FAST_TIMEOUT,
    )


# ------------------------------------------------------------------
# 7. test_nan_cleared_after_recovery
# ------------------------------------------------------------------


async def test_nan_cleared_after_recovery(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """NaN triggers recovery; after recovery completes, training resumes with clean metrics."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=build_detector_chain(),
        scrape_interval_seconds=0.5,
    )

    # Step 1: verify training is stable
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 2: inject NaN to trigger recovery
    await testbed.inject_nan_loss()
    await testbed.wait_for_mode(
        mode=ControllerMode.RECOVERY,
        timeout=FAST_TIMEOUT,
    )

    # Step 3: wait for recovery to complete
    final = await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=LONG_RECOVERY_TIMEOUT,
    )
    assert final.mode == ControllerMode.MONITORING

    # Step 4: training resumes normally (NaN state cleared by submit())
    await testbed.wait_for_training_stable(n_iterations=5, timeout=FAST_TIMEOUT)


# ------------------------------------------------------------------
# 8. test_disk_space_low_notifies_no_recovery
# ------------------------------------------------------------------


async def test_disk_space_low_notifies_no_recovery(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Low disk space triggers NOTIFY_HUMAN but no recovery."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=build_detector_chain(),
        scrape_interval_seconds=0.5,
    )

    # Step 1: verify training is stable
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 2: inject low disk space (500 MB < 1 GB threshold)
    await testbed.inject_collector_metrics(
        node_id="n-0",
        metrics=[
            GaugeSample(
                name=NODE_FILESYSTEM_AVAIL_BYTES,
                labels={"node_id": "n-0", "mountpoint": "/data"},
                value=500_000_000.0,
            ),
        ],
    )

    # Step 3: verify no recovery triggered
    await assert_no_recovery_triggered(
        testbed,
        observation_ticks=20,
        timeout=FAST_TIMEOUT,
    )

    # Step 4: notifier should have received a disk-related notification
    calls = testbed.notifications
    assert len(calls) > 0, "Notifier should have received disk space notification"


# ------------------------------------------------------------------
# 9. test_xid_non_auto_recoverable_triggers_eviction
# ------------------------------------------------------------------


async def test_xid_non_auto_recoverable_triggers_eviction(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL > 0 triggers GpuFaultDetector and eviction."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=build_detector_chain(),
        scrape_interval_seconds=0.5,
    )

    # Step 1: verify training is stable
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)
    old_run_id = (await testbed.get_status()).active_run_id

    # Step 2: inject non-auto-recoverable XID event
    await testbed.inject_collector_metrics(
        node_id="n-0",
        metrics=[
            GaugeSample(
                name=XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL,
                labels={"node_id": "n-0"},
                value=1.0,
            ),
        ],
    )

    # Step 3: wait for eviction + recovery (run_id changes) AND return to MONITORING
    deadline = time.monotonic() + LONG_RECOVERY_TIMEOUT
    while time.monotonic() < deadline:
        status = await testbed.get_status()
        if status.active_run_id != old_run_id and status.mode == ControllerMode.MONITORING:
            break
        await asyncio.sleep(0.5)
    else:
        raise TimeoutError(
            f"XID fault did not complete recovery within {LONG_RECOVERY_TIMEOUT}s: "
            f"run_id changed={status.active_run_id != old_run_id}, mode={status.mode}"
        )

    assert status.mode == ControllerMode.MONITORING


# ------------------------------------------------------------------
# 10. test_mfu_absolute_minimum_notifies_human
# ------------------------------------------------------------------


async def test_mfu_absolute_minimum_notifies_human(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """MFU below absolute minimum triggers NOTIFY_HUMAN but no recovery."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[
            TrainingCrashDetector(),
            MfuDeclineDetector(
                config=MfuDeclineDetectorConfig(
                    mfu_absolute_minimum=0.5,
                    consecutive_steps=2,
                    baseline_steps=2,
                ),
            ),
        ],
    )

    # Step 1: verify training is stable
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 2: inject very low MFU
    await testbed.inject_custom_metrics({"mfu": 0.01})

    # Step 3: verify no recovery triggered
    await assert_no_recovery_triggered(
        testbed,
        observation_ticks=20,
        timeout=FAST_TIMEOUT,
    )

    # Step 4: notifier should have received MFU alert
    calls = testbed.notifications
    assert len(calls) > 0, "Notifier should have received MFU alert"


# ------------------------------------------------------------------
# 11. test_non_critical_suppressed_during_recovery
# ------------------------------------------------------------------


async def test_non_critical_suppressed_during_recovery(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Crash enters RECOVERY, low disk injected during recovery does not interrupt; recovery completes."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=build_detector_chain(),
        scrape_interval_seconds=0.5,
    )

    # Step 1: verify training is stable
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 2: crash to enter recovery
    await testbed.crash_training()
    await testbed.wait_for_mode(
        mode=ControllerMode.RECOVERY,
        timeout=FAST_TIMEOUT,
    )

    # Step 3: while in recovery, inject disk space fault (non-critical)
    await testbed.inject_collector_metrics(
        node_id="n-0",
        metrics=[
            GaugeSample(
                name=NODE_FILESYSTEM_AVAIL_BYTES,
                labels={"node_id": "n-0", "mountpoint": "/data"},
                value=100_000_000.0,
            ),
        ],
    )

    # Step 4: recovery should complete normally (disk alert ignored)
    final = await testbed.wait_for_mode(
        mode=ControllerMode.MONITORING,
        timeout=LONG_RECOVERY_TIMEOUT,
    )
    assert final.mode == ControllerMode.MONITORING


# ------------------------------------------------------------------
# 12. test_crash_plus_hardware_found_in_realtime_checks
# ------------------------------------------------------------------


async def test_crash_plus_hardware_found_in_realtime_checks(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """GPU fault metric + crash: GpuFaultDetector discovers bad node during recovery's realtime checks."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector(), GpuFaultDetector()],
        scrape_interval_seconds=0.5,
    )

    # Step 1: verify training is stable
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 2: inject GPU fault metric first
    await testbed.inject_collector_metrics(
        node_id="n-0",
        metrics=[
            GaugeSample(
                name=GPU_AVAILABLE,
                labels={"node_id": "n-0", "gpu": "0"},
                value=0.0,
            ),
        ],
    )
    await asyncio.sleep(0.5 * 3)

    # Step 3: crash triggers TrainingCrashDetector; realtime checks find GPU fault
    old_run_id = (await testbed.get_status()).active_run_id
    await testbed.crash_training()

    deadline = time.monotonic() + LONG_RECOVERY_TIMEOUT
    while time.monotonic() < deadline:
        status = await testbed.get_status()
        if status.active_run_id != old_run_id and status.mode == ControllerMode.MONITORING:
            break
        await asyncio.sleep(0.5)
    else:
        raise TimeoutError(
            f"Recovery did not complete within {LONG_RECOVERY_TIMEOUT}s: "
            f"run_id changed={status.active_run_id != old_run_id}, mode={status.mode}"
        )

    assert status.mode == ControllerMode.MONITORING


# ------------------------------------------------------------------
# 13. test_max_bad_nodes_one_single_fault_is_false_positive
# ------------------------------------------------------------------


async def test_max_bad_nodes_one_single_fault_is_false_positive(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """2 GPU faults with max_simultaneous_bad_nodes=1: treated as false positive (2 > 1)."""
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="n-0", num_ranks=2),
            TestbedNodeConfig(node_id="n-1", num_ranks=2),
        ],
        detectors=build_detector_chain(),
        scrape_interval_seconds=0.5,
        max_simultaneous_bad_nodes=1,
    )

    # Step 1: verify training is stable
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 2: inject GPU fault on both nodes (2 > max_simultaneous_bad_nodes=1)
    for node_id in ("n-0", "n-1"):
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

    # Step 3: verify no recovery triggered
    await assert_no_recovery_triggered(
        testbed,
        observation_ticks=20,
        timeout=FAST_TIMEOUT,
    )


# ------------------------------------------------------------------
# 14. test_hardware_fault_cleared_no_retriggering
# ------------------------------------------------------------------


async def test_hardware_fault_cleared_no_retriggering(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """GPU fault triggers recovery, clearing fault during recovery prevents re-triggering."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=build_detector_chain(),
        scrape_interval_seconds=0.5,
    )

    # Step 1: verify training is stable
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 2: inject GPU fault to trigger eviction
    await testbed.inject_collector_metrics(
        node_id="n-0",
        metrics=[
            GaugeSample(
                name=GPU_AVAILABLE,
                labels={"node_id": "n-0", "gpu": "0"},
                value=0.0,
            ),
        ],
    )

    await testbed.wait_for_mode(
        mode=ControllerMode.RECOVERY,
        timeout=FAST_TIMEOUT,
    )

    # Step 3: clear fault metric (simulating node replacement)
    await testbed.inject_collector_metrics(
        node_id="n-0",
        metrics=[
            GaugeSample(
                name=GPU_AVAILABLE,
                labels={"node_id": "n-0", "gpu": "0"},
                value=1.0,
            ),
        ],
    )

    # Step 4: recovery completes
    final = await testbed.wait_for_mode(
        mode=ControllerMode.MONITORING,
        timeout=LONG_RECOVERY_TIMEOUT,
    )
    assert final.mode == ControllerMode.MONITORING

    # Step 5: verify no new recovery triggered after several ticks
    await assert_no_recovery_triggered(
        testbed,
        observation_ticks=20,
        timeout=FAST_TIMEOUT,
    )
