"""Semi-E2E: multi-node scenarios — registration, targeted eviction, grace period, stale run_id."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

import pytest
import ray

from miles.utils.ft.controller.detectors.core.training_crash import TrainingCrashDetector
from miles.utils.ft.controller.types import ControllerMode
from miles.utils.ft.utils.sliding_window import SlidingWindowThrottle
from tests.fast.utils.ft.integration.conftest import (
    FAST_TIMEOUT,
    LONG_RECOVERY_TIMEOUT,
    RECOVERY_TIMEOUT,
)
from tests.fast.utils.ft.testbed.config import TestbedNodeConfig
from tests.fast.utils.ft.testbed.train import MilesTestbed
from tests.fast.utils.ft.utils.diagnostic_fakes import DelayedDiagnosticOrchestrator

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.anyio,
]


# ------------------------------------------------------------------
# 1. test_multi_rank_registration_and_targeted_eviction
# ------------------------------------------------------------------


async def test_multi_rank_registration_and_targeted_eviction(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """2 nodes × 2 ranks. Crash during recovery escalates to StopTimeDiagnostics."""
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="n-0", num_ranks=2),
            TestbedNodeConfig(node_id="n-1", num_ranks=2),
        ],
        detectors=[TrainingCrashDetector()],
        step_interval=2.0,
        recovery_cooldown=SlidingWindowThrottle(
            window_minutes=1.0,
            max_count=2,
        ),
        diagnostic_orchestrator_override=DelayedDiagnosticOrchestrator(delay_seconds=5.0),
    )

    # Step 1: verify training is stable across both nodes
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 2: crash training → enters recovery, wait for MonitoringProgress
    await testbed.crash_training()
    await testbed.wait_for_recovery_phase(
        phase="MonitoringProgressSt",
        timeout=RECOVERY_TIMEOUT,
    )

    # Step 3: crash again during MonitoringProgress → escalates to StopTimeDiagnostics
    await testbed.crash_training()

    deadline = time.monotonic() + RECOVERY_TIMEOUT
    while time.monotonic() < deadline:
        status = await testbed.get_status()
        if status.recovery is not None and status.recovery.phase == "StopTimeDiagnosticsSt":
            break
        await asyncio.sleep(0.5)
    else:
        raise TimeoutError(
            f"StopTimeDiagnosticsSt not observed within {RECOVERY_TIMEOUT}s"
        )


# ------------------------------------------------------------------
# 2. test_parallel_rank_registration
# ------------------------------------------------------------------


async def test_parallel_rank_registration(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """2 nodes × 2 ranks register in parallel; training reaches stable iteration."""
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="n-0", num_ranks=2),
            TestbedNodeConfig(node_id="n-1", num_ranks=2),
        ],
        detectors=[TrainingCrashDetector()],
    )

    # Step 1: wait for training to stabilize (proves all 4 ranks registered)
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 2: verify status has an active run_id and iteration > 0
    status = await testbed.get_status()
    assert status.active_run_id is not None, "Expected active_run_id to be set"
    assert status.latest_iteration is not None and status.latest_iteration > 0, (
        f"Expected iteration > 0, got {status.latest_iteration}"
    )


# ------------------------------------------------------------------
# 3. test_detectors_suppressed_during_grace_period
# ------------------------------------------------------------------


async def test_detectors_suppressed_during_grace_period(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Crash during registration grace period is suppressed; recovery fires after grace ends.

    registration_grace_ticks=50 with tick_interval=0.1 gives a 5s grace window.
    Workers always register in MilesTestbed, so we crash immediately after launch.
    Detectors are suppressed during grace → no recovery. After grace ends, dead
    workers are detected → recovery starts.
    """
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="n-0", num_ranks=2),
        ],
        detectors=[TrainingCrashDetector()],
        tick_interval=0.1,
        registration_grace_ticks=50,
    )

    # Step 1: crash training immediately
    await testbed.crash_training()

    # Step 2: verify no recovery during the first ~3 seconds of grace period
    grace_observation_end = time.monotonic() + 3.0
    while time.monotonic() < grace_observation_end:
        status = await testbed.get_status()
        assert status.mode == ControllerMode.MONITORING, (
            f"Recovery triggered during grace period at tick {status.tick_count}"
        )
        await asyncio.sleep(0.3)

    # Step 3: after grace period ends (~5s), detector fires → recovery starts
    status = await testbed.wait_for_mode(
        mode=ControllerMode.RECOVERY,
        timeout=RECOVERY_TIMEOUT,
    )
    assert status.mode == ControllerMode.RECOVERY


# ------------------------------------------------------------------
# 4. test_stale_run_id_registration_rejected
# ------------------------------------------------------------------


async def test_stale_run_id_registration_rejected(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """After recovery produces a new run_id, registering with the old run_id is silently rejected."""
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="n-0", num_ranks=2),
        ],
        detectors=[TrainingCrashDetector()],
    )

    # Step 1: capture initial run_id
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)
    initial_status = await testbed.get_status()
    old_run_id = initial_status.active_run_id
    assert old_run_id is not None

    # Step 2: crash → recovery → new run_id
    await testbed.crash_training()
    final_status = await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=RECOVERY_TIMEOUT,
    )
    new_run_id = final_status.active_run_id
    assert new_run_id is not None
    assert new_run_id != old_run_id

    # Step 3: attempt to register with the stale run_id (silently rejected)
    ray.get(
        testbed.controller.register_training_rank.remote(
            run_id=old_run_id,
            rank=99,
            world_size=2,
            node_id="n-0",
            exporter_address="http://fake:9090",
            pid=99999,
        ),
        timeout=5,
    )

    # Step 4: verify training continues normally with the new run_id
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)
    status = await testbed.get_status()
    assert status.active_run_id == new_run_id


# ------------------------------------------------------------------
# 5. test_multi_node_crash_recovers_cleanly
# ------------------------------------------------------------------


async def test_multi_node_crash_recovers_cleanly(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """2 nodes × 2 ranks. Crash all → recovery → training resumes."""
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="n-0", num_ranks=2),
            TestbedNodeConfig(node_id="n-1", num_ranks=2),
        ],
        detectors=[TrainingCrashDetector()],
    )

    # Step 1: verify training is stable
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 2: crash all training
    await testbed.crash_training()

    # Step 3: wait for recovery to complete → back to MONITORING
    final_status = await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=RECOVERY_TIMEOUT,
    )
    assert final_status.mode == ControllerMode.MONITORING

    # Step 4: training resumes with new workers
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)


# ------------------------------------------------------------------
# 6. test_3_nodes_full_recovery_all_ranks_reregister
# ------------------------------------------------------------------


async def test_3_nodes_full_recovery_all_ranks_reregister(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """3 nodes × 2 ranks. Crash → recovery → all 6 ranks re-register and training resumes."""
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="n-0", num_ranks=2),
            TestbedNodeConfig(node_id="n-1", num_ranks=2),
            TestbedNodeConfig(node_id="n-2", num_ranks=2),
        ],
        detectors=[TrainingCrashDetector()],
    )

    # Step 1: verify training is stable across all 3 nodes
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 2: crash all training
    await testbed.crash_training()

    # Step 3: wait for recovery to complete
    final_status = await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=LONG_RECOVERY_TIMEOUT,
    )
    assert final_status.mode == ControllerMode.MONITORING

    # Step 4: all ranks re-registered, training resumes
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 5: verify iteration is advancing (proves all 6 ranks are active)
    status = await testbed.get_status()
    assert status.latest_iteration is not None and status.latest_iteration > 0
