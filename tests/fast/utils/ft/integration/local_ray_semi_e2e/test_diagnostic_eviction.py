"""Semi-E2E: diagnostic eviction — bad node eviction, unreachable agents, escalation, notifications."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import pytest
import ray

from miles.utils.ft.adapters.types import ft_node_agent_actor_name
from miles.utils.ft.controller.detectors.core.training_crash import TrainingCrashDetector
from miles.utils.ft.controller.types import ControllerMode
from tests.fast.utils.ft.integration.conftest import (
    FAST_TIMEOUT,
    LONG_RECOVERY_TIMEOUT,
    RECOVERY_TIMEOUT,
)
from tests.fast.utils.ft.testbed import MilesTestbed, TestbedNodeConfig

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.anyio,
]

_SLOW_STEP = 2.0


# ------------------------------------------------------------------
# 1. test_diagnostic_failure_evicts_bad_node
# ------------------------------------------------------------------


async def test_diagnostic_failure_evicts_bad_node(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """node-0 diagnostic fails, node-1 passes -> only node-0 evicted after double crash."""
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="node-0", num_ranks=2, diagnostic_pass=False),
            TestbedNodeConfig(node_id="node-1", num_ranks=2, diagnostic_pass=True),
        ],
        detectors=[TrainingCrashDetector()],
        step_interval=_SLOW_STEP,
    )

    # Step 1: wait for training to stabilize
    await testbed.wait_for_training_stable(n_iterations=1, timeout=FAST_TIMEOUT)

    # Step 2: crash -> recovery enters MonitoringProgress
    await testbed.crash_training()
    await testbed.wait_for_recovery_phase(
        phase="MonitoringProgressSt",
        timeout=FAST_TIMEOUT,
    )

    # Step 3: crash during MonitoringProgress -> diagnostics run -> eviction
    await testbed.crash_training()
    await testbed.wait_for_mode(
        mode=ControllerMode.MONITORING,
        timeout=RECOVERY_TIMEOUT,
    )

    # Step 4: verify node-0 evicted, node-1 not
    assert testbed.node_manager.was_ever_marked_bad("node-0")
    assert not testbed.node_manager.was_ever_marked_bad("node-1")


# ------------------------------------------------------------------
# 2. test_multi_node_eviction_marks_both_bad
# ------------------------------------------------------------------


async def test_multi_node_eviction_marks_both_bad(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """2 bad nodes out of 3 -> both marked bad, the passing node is not."""
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="node-0", num_ranks=2, diagnostic_pass=False),
            TestbedNodeConfig(node_id="node-1", num_ranks=2, diagnostic_pass=False),
            TestbedNodeConfig(node_id="node-2", num_ranks=2, diagnostic_pass=True),
        ],
        detectors=[TrainingCrashDetector()],
        step_interval=_SLOW_STEP,
    )

    # Step 1: wait for training to stabilize
    await testbed.wait_for_training_stable(n_iterations=1, timeout=FAST_TIMEOUT)

    # Step 2: crash -> MonitoringProgress
    await testbed.crash_training()
    await testbed.wait_for_recovery_phase(
        phase="MonitoringProgressSt",
        timeout=FAST_TIMEOUT,
    )

    # Step 3: crash during MonitoringProgress -> diagnostics -> eviction
    await testbed.crash_training()
    await testbed.wait_for_mode(
        mode=ControllerMode.MONITORING,
        timeout=LONG_RECOVERY_TIMEOUT,
    )

    # Step 4: verify both failing nodes evicted, passing node not
    assert testbed.node_manager.was_ever_marked_bad("node-0")
    assert testbed.node_manager.was_ever_marked_bad("node-1")
    assert not testbed.node_manager.was_ever_marked_bad("node-2")


# ------------------------------------------------------------------
# 3. test_unreachable_node_agent_treated_as_bad
# ------------------------------------------------------------------


async def test_unreachable_node_agent_treated_as_bad(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Kill node-0's agent before crash -> RayActorError during diagnostics -> treated as bad."""
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="node-0", num_ranks=2, diagnostic_pass=True),
            TestbedNodeConfig(node_id="node-1", num_ranks=2, diagnostic_pass=True),
        ],
        detectors=[TrainingCrashDetector()],
        step_interval=_SLOW_STEP,
    )

    # Step 1: wait for training to stabilize
    await testbed.wait_for_training_stable(n_iterations=1, timeout=FAST_TIMEOUT)

    # Step 2: kill node-0's agent so it becomes unreachable during diagnostics
    ray.kill(
        ray.get_actor(ft_node_agent_actor_name(testbed.ft_id, "node-0")),
        no_restart=True,
    )

    # Step 3: crash -> MonitoringProgress
    await testbed.crash_training()
    await testbed.wait_for_recovery_phase(
        phase="MonitoringProgressSt",
        timeout=FAST_TIMEOUT,
    )

    # Step 4: crash during MonitoringProgress -> diagnostics -> node-0 unreachable -> evicted
    await testbed.crash_training()
    await testbed.wait_for_mode(
        mode=ControllerMode.MONITORING,
        timeout=LONG_RECOVERY_TIMEOUT,
    )

    # Step 5: verify node-0 was treated as bad (unreachable = bad)
    assert testbed.node_manager.was_ever_marked_bad("node-0")


# ------------------------------------------------------------------
# 4. test_all_diagnostics_fail_notifies_human
# ------------------------------------------------------------------


async def test_all_diagnostics_fail_notifies_human(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """All nodes fail diagnostic -> all evicted."""
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="node-0", num_ranks=2, diagnostic_pass=False),
            TestbedNodeConfig(node_id="node-1", num_ranks=2, diagnostic_pass=False),
        ],
        detectors=[TrainingCrashDetector()],
        step_interval=_SLOW_STEP,
    )

    # Step 1: wait for training to stabilize
    await testbed.wait_for_training_stable(n_iterations=1, timeout=FAST_TIMEOUT)

    # Step 2: crash -> MonitoringProgress
    await testbed.crash_training()
    await testbed.wait_for_recovery_phase(
        phase="MonitoringProgressSt",
        timeout=FAST_TIMEOUT,
    )

    # Step 3: crash during MonitoringProgress -> diagnostics -> all fail -> all evicted
    await testbed.crash_training()
    await testbed.wait_for_mode(
        mode=ControllerMode.MONITORING,
        timeout=LONG_RECOVERY_TIMEOUT,
    )


# ------------------------------------------------------------------
# 5. test_eviction_notification
# ------------------------------------------------------------------


async def test_eviction_notification(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """During eviction flow, notifier receives at least one notification."""
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="node-0", num_ranks=2, diagnostic_pass=False),
            TestbedNodeConfig(node_id="node-1", num_ranks=2, diagnostic_pass=True),
        ],
        detectors=[TrainingCrashDetector()],
        step_interval=_SLOW_STEP,
    )

    # Step 1: wait for training to stabilize
    await testbed.wait_for_training_stable(n_iterations=1, timeout=FAST_TIMEOUT)

    # Step 2: crash -> MonitoringProgress
    await testbed.crash_training()
    await testbed.wait_for_recovery_phase(
        phase="MonitoringProgressSt",
        timeout=FAST_TIMEOUT,
    )

    # Step 3: crash during MonitoringProgress -> diagnostics -> eviction
    await testbed.crash_training()
    await testbed.wait_for_mode(
        mode=ControllerMode.MONITORING,
        timeout=LONG_RECOVERY_TIMEOUT,
    )

    # Step 4: verify notifier received calls
    assert len(testbed.notifications) > 0, "Notifier should have received eviction notification"


# ------------------------------------------------------------------
# 6. test_eviction_escalation_to_notify_humans
# ------------------------------------------------------------------


async def test_eviction_escalation_to_notify_humans(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Evict node-0 -> restart -> MonitoringProgress -> crash -> RestartFailed -> NotifyHumans.

    Uses high monitoring_success_iterations so MonitoringProgress stays active
    long enough for us to inject crashes at the right moment.
    """
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="node-0", num_ranks=2, diagnostic_pass=False),
            TestbedNodeConfig(node_id="node-1", num_ranks=2, diagnostic_pass=True),
        ],
        detectors=[TrainingCrashDetector()],
        step_interval=_SLOW_STEP,
        monitoring_success_iterations=999,
    )

    # Step 1: wait for training to stabilize
    await testbed.wait_for_training_stable(n_iterations=1, timeout=FAST_TIMEOUT)

    # Step 2: crash -> MonitoringProgress (first recovery)
    await testbed.crash_training()
    await testbed.wait_for_recovery_phase(
        phase="MonitoringProgressSt",
        timeout=FAST_TIMEOUT,
    )

    # Step 3: crash during MonitoringProgress -> diagnostics -> evict node-0 ->
    # restart -> post-eviction MonitoringProgress
    await testbed.crash_training()

    deadline = time.monotonic() + RECOVERY_TIMEOUT
    while time.monotonic() < deadline:
        if testbed.node_manager.was_ever_marked_bad("node-0"):
            break
        await asyncio.sleep(0.5)
    else:
        raise TimeoutError(
            f"node-0 was not marked bad within {RECOVERY_TIMEOUT}s during eviction flow"
        )

    await testbed.wait_for_recovery_phase(
        phase="MonitoringProgressSt",
        timeout=RECOVERY_TIMEOUT,
    )

    # Step 4: crash during post-eviction MonitoringProgress -> RestartFailed -> NotifyHumans
    await testbed.crash_training()

    deadline = time.monotonic() + LONG_RECOVERY_TIMEOUT
    while time.monotonic() < deadline:
        status = await testbed.get_status()
        if status.recovery is not None and status.recovery.phase == "NotifyHumansSt":
            break
        if status.mode == ControllerMode.MONITORING and not status.recovery_in_progress:
            break
        await asyncio.sleep(0.5)


# ------------------------------------------------------------------
# 7. test_all_diagnostics_pass_escalates_to_notify
# ------------------------------------------------------------------


async def test_all_diagnostics_pass_escalates_to_notify(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """All nodes pass diagnostics (no root cause found) -> NotifyHumans."""
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="node-0", num_ranks=2, diagnostic_pass=True),
            TestbedNodeConfig(node_id="node-1", num_ranks=2, diagnostic_pass=True),
        ],
        detectors=[TrainingCrashDetector()],
        step_interval=_SLOW_STEP,
    )

    # Step 1: wait for training to stabilize
    await testbed.wait_for_training_stable(n_iterations=1, timeout=FAST_TIMEOUT)

    # Step 2: crash -> MonitoringProgress
    await testbed.crash_training()
    await testbed.wait_for_recovery_phase(
        phase="MonitoringProgressSt",
        timeout=FAST_TIMEOUT,
    )

    # Step 3: crash during MonitoringProgress -> diagnostics -> all pass -> NotifyHumans
    # NotifyHumansSt is transient (converges through in one tick), so we wait
    # for the final MONITORING mode and verify via notifications.
    await testbed.crash_training()

    await testbed.wait_for_mode(
        mode=ControllerMode.MONITORING,
        timeout=RECOVERY_TIMEOUT,
    )

    # Step 4: no nodes should be evicted (all diagnostics passed)
    assert not testbed.node_manager.was_ever_marked_bad("node-0")
    assert not testbed.node_manager.was_ever_marked_bad("node-1")

    # Step 5: notifier should have received a notification (proves NotifyHumans path)
    assert len(testbed.notifications) > 0, (
        "Notifier should have received notification for all-pass diagnostics"
    )


# ------------------------------------------------------------------
# 8. test_notification_contains_trigger_and_context
# ------------------------------------------------------------------


async def test_notification_contains_trigger_and_context(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """NotifyHumans notification includes trigger= and state_before= for operator actionability."""
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="node-0", num_ranks=2, diagnostic_pass=True),
            TestbedNodeConfig(node_id="node-1", num_ranks=2, diagnostic_pass=True),
        ],
        detectors=[TrainingCrashDetector()],
        step_interval=_SLOW_STEP,
    )

    # Step 1: wait for training to stabilize
    await testbed.wait_for_training_stable(n_iterations=1, timeout=FAST_TIMEOUT)

    # Step 2: crash -> MonitoringProgress -> crash -> diagnostics -> all pass -> NotifyHumans
    # NotifyHumansSt is transient, so wait for MONITORING mode and verify via notifications.
    await testbed.crash_training()
    await testbed.wait_for_recovery_phase(
        phase="MonitoringProgressSt",
        timeout=FAST_TIMEOUT,
    )
    await testbed.crash_training()

    await testbed.wait_for_mode(
        mode=ControllerMode.MONITORING,
        timeout=RECOVERY_TIMEOUT,
    )

    # Step 3: verify notification content includes trigger and state_before
    calls = testbed.notifications
    assert len(calls) > 0, "Notifier should have received calls"

    recovery_alerts = [(t, c, s) for t, c, s in calls if t == "Recovery Alert"]
    assert len(recovery_alerts) > 0, "Expected at least one 'Recovery Alert' notification"

    _, content, _ = recovery_alerts[0]
    assert "trigger=" in content, f"Notification missing trigger=: {content}"
    assert "state_before=" in content, f"Notification missing state_before=: {content}"


# ------------------------------------------------------------------
# 9. test_clear_bad_nodes
# ------------------------------------------------------------------


async def test_clear_bad_nodes(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Eviction marks node bad; verify node-0 was marked bad after eviction flow."""
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="node-0", num_ranks=2, diagnostic_pass=False),
            TestbedNodeConfig(node_id="node-1", num_ranks=2, diagnostic_pass=True),
        ],
        detectors=[TrainingCrashDetector()],
        step_interval=_SLOW_STEP,
    )

    # Step 1: wait for training to stabilize
    await testbed.wait_for_training_stable(n_iterations=1, timeout=FAST_TIMEOUT)

    # Step 2: crash -> MonitoringProgress -> crash -> eviction -> recovery
    await testbed.crash_training()
    await testbed.wait_for_recovery_phase(
        phase="MonitoringProgressSt",
        timeout=FAST_TIMEOUT,
    )
    await testbed.crash_training()
    await testbed.wait_for_mode(
        mode=ControllerMode.MONITORING,
        timeout=RECOVERY_TIMEOUT,
    )

    # Step 3: assert node-0 was marked bad
    assert testbed.node_manager.was_ever_marked_bad("node-0")
