"""Semi-E2E scenario tests -- thin wrappers calling shared scenario functions.

Previously semi-E2E tests had scenario logic inlined with no code sharing
with E2E tests. These tests use shared scenario functions via the
FaultTestProtocol, ensuring E2E and semi-E2E cover identical scenarios.

MilesTestbed satisfies both FaultTestProtocol and FaultInjectionProtocol
directly (structural typing), so no adapter is needed.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable

import pytest

from miles.utils.ft.controller.detectors.core.training_crash import TrainingCrashDetector
from miles.utils.ft.utils.sliding_window import SlidingWindowThrottle
from tests.fast.utils.ft.integration.conftest import (
    FAST_TIMEOUT,
    LONG_RECOVERY_TIMEOUT,
    RECOVERY_TIMEOUT,
)
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios import (
    FaultInjectionProtocol,
    FaultTestProtocol,
    scenario_hang_detection_and_recovery,
    scenario_multi_cell_crash,
    scenario_no_false_positive,
    scenario_repeated_crash,
    scenario_rollout_crash,
    scenario_rollout_gpu_xid,
    scenario_transient_crash,
)
from tests.fast.utils.ft.testbed import MilesTestbed, TestbedNodeConfig
from tests.fast.utils.ft.utils.controller_fakes import FastHangDetector
from tests.fast.utils.ft.utils.diagnostic_fakes import DelayedDiagnosticOrchestrator

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.anyio,
]


# ------------------------------------------------------------------
# Protocol compliance
# ------------------------------------------------------------------


def test_testbed_satisfies_fault_test_protocol() -> None:
    """MilesTestbed structurally satisfies FaultTestProtocol (runtime check)."""
    assert issubclass(MilesTestbed, FaultTestProtocol)


def test_testbed_satisfies_fault_injection_protocol() -> None:
    """MilesTestbed structurally satisfies FaultInjectionProtocol (runtime check)."""
    assert issubclass(MilesTestbed, FaultInjectionProtocol)


# ------------------------------------------------------------------
# Scenario: transient crash
# ------------------------------------------------------------------


async def test_transient_crash(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Single crash -> auto-recovery -> training resumes."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector()],
    )

    await scenario_transient_crash(
        env=testbed,
        injector=testbed,
        stable_iterations=3,
        stable_timeout=FAST_TIMEOUT,
        recovery_timeout=RECOVERY_TIMEOUT,
        post_recovery_iterations=3,
        post_recovery_timeout=FAST_TIMEOUT,
    )


# ------------------------------------------------------------------
# Scenario: repeated crash -> DIAGNOSING
# ------------------------------------------------------------------


async def test_repeated_crash(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Two rapid crashes -> escalates to StopTimeDiagnostics."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector()],
        step_interval=2.0,
        recovery_cooldown=SlidingWindowThrottle(window_minutes=1.0, max_count=5),
        diagnostic_orchestrator_override=DelayedDiagnosticOrchestrator(delay_seconds=5.0),
    )

    await scenario_repeated_crash(
        env=testbed,
        injector=testbed,
        stable_iterations=3,
        stable_timeout=FAST_TIMEOUT,
        recovery_timeout=RECOVERY_TIMEOUT,
    )


# ------------------------------------------------------------------
# Scenario: hang detection and recovery
# ------------------------------------------------------------------


async def test_hang_detection_and_recovery(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Hang -> detect -> full recovery -> training resumes."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[FastHangDetector(timeout_seconds=3.0)],
    )

    await scenario_hang_detection_and_recovery(
        env=testbed,
        injector=testbed,
        recovery_timeout=LONG_RECOVERY_TIMEOUT,
        post_recovery_iterations=3,
        post_recovery_timeout=FAST_TIMEOUT,
    )


# ------------------------------------------------------------------
# Scenario: no false positive
# ------------------------------------------------------------------


async def test_no_false_positive(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Healthy training with real workers never triggers spurious recovery."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector()],
    )

    await scenario_no_false_positive(
        env=testbed,
        observation_iterations=10,
        timeout=FAST_TIMEOUT,
        poll_interval=1.0,
    )


# ------------------------------------------------------------------
# Scenario: rollout crash -> L1 restart
# ------------------------------------------------------------------


async def test_rollout_crash(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Kill sglang engine -> RolloutCrashDetector -> L1 restart -> DetectingAnomaly."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        rollout_nodes=[TestbedNodeConfig(node_id="rollout-0")],
        rollout_num_cells=1,
        rollout_alive_threshold_seconds=2.0,
        rollout_monitoring_alive_duration_seconds=0,
    )

    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    await scenario_rollout_crash(
        env=testbed,
        crash_fn=functools.partial(testbed.kill_sglang_cell, "default"),
        target_subsystem="rollout_default",
        stable_iterations=0,
        detection_timeout=RECOVERY_TIMEOUT,
        recovery_timeout=LONG_RECOVERY_TIMEOUT,
    )


# ------------------------------------------------------------------
# Scenario: multi-cell crash -> independent recovery
# ------------------------------------------------------------------


async def test_multi_cell_crash(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Crash 2 of 3 rollout cells -> both recover independently, third unaffected."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        rollout_nodes=[TestbedNodeConfig(node_id="rollout-0")],
        rollout_num_cells=3,
        rollout_alive_threshold_seconds=2.0,
        rollout_monitoring_alive_duration_seconds=0,
    )

    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    crash_fns = [
        functools.partial(testbed.kill_sglang_cell, "0"),
        functools.partial(testbed.kill_sglang_cell, "1"),
    ]

    await scenario_multi_cell_crash(
        env=testbed,
        crash_fns=crash_fns,
        all_rollout_subsystems=["rollout_0", "rollout_1", "rollout_2"],
        stable_iterations=0,
        stagger_delay=2.0,
        detection_timeout=RECOVERY_TIMEOUT,
        recovery_timeout=LONG_RECOVERY_TIMEOUT,
    )


# ------------------------------------------------------------------
# Scenario: rollout GPU XID -> eviction
# ------------------------------------------------------------------


async def test_rollout_gpu_xid(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Inject GPU XID on rollout node -> recovery -> node marked bad."""
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="train-0", num_ranks=2),
            TestbedNodeConfig(node_id="train-1", num_ranks=2),
        ],
        rollout_nodes=[TestbedNodeConfig(node_id="rollout-0")],
        spare_nodes=["spare-0"],
        rollout_num_cells=1,
        monitoring_success_iterations=3,
        rollout_alive_threshold_seconds=2.0,
        rollout_monitoring_alive_duration_seconds=0,
    )

    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    await scenario_rollout_gpu_xid(
        env=testbed,
        inject_xid_fn=functools.partial(testbed.inject_gpu_xid, "rollout-0"),
        target_subsystem="rollout_default",
        detection_timeout=RECOVERY_TIMEOUT,
        recovery_timeout=LONG_RECOVERY_TIMEOUT,
    )

    # Step: verify node was marked bad (hardware fault -> eviction)
    assert testbed.node_manager.was_ever_marked_bad("rollout-0"), (
        "Rollout node rollout-0 was NOT evicted despite GPU XID"
    )
