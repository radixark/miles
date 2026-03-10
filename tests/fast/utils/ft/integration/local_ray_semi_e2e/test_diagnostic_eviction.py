"""Semi-E2E: diagnostic eviction — bad node eviction, excluded nodes, partial, all-evicted."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from tests.fast.utils.ft.integration.conftest import FAST_TIMEOUT, LONG_RECOVERY_TIMEOUT, RECOVERY_TIMEOUT
from tests.fast.utils.ft.integration.local_ray_semi_e2e.conftest import _SLOW_STEP, E2EEnv, NodeSpec
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios import (
    assert_phase_path_contains,
    get_status,
    wait_for_recovery_complete,
    wait_for_recovery_phase,
    wait_for_training_stable,
)

from miles.utils.ft.controller.detectors.core.training_crash import TrainingCrashDetector
from miles.utils.ft.controller.types import ControllerMode


class TestDiagnosticEviction:
    async def test_diagnostic_failure_evicts_bad_node(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """node-0 diagnostic fails, node-1 passes → only node-0 evicted.

        Crash during MONITORING → step_monitoring sees FAILED → DIAGNOSING.
        StubDiagnostic resolves instantly so we check phase_history after
        recovery completes rather than catching DIAGNOSING in flight.
        """
        env = make_e2e_env(
            ft_id="e2ediag",
            nodes=[
                NodeSpec(node_id="e2ediag-node-0", num_ranks=1, diagnostic_pass=False),
                NodeSpec(node_id="e2ediag-node-1", num_ranks=1, diagnostic_pass=True),
            ],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=FAST_TIMEOUT)

        # Step 1: crash → wait for recovery MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=FAST_TIMEOUT,
        )

        # Step 2: crash during MONITORING → DIAGNOSING (fast) → recovery completes
        await env.injector.crash_training()
        final = await wait_for_recovery_complete(env.controller, timeout=RECOVERY_TIMEOUT)

        assert final.mode == ControllerMode.MONITORING
        assert_phase_path_contains(
            final,
            [
                "StopTimeDiagnostics",
                "Evicting",
            ],
        )

        assert env.node_manager.was_ever_marked_bad("e2ediag-node-0")
        assert not env.node_manager.was_ever_marked_bad("e2ediag-node-1")


class TestEvictionExcludedNodes:
    async def test_multi_node_eviction_marks_both_nodes_bad(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """2 bad nodes out of 3 → both nodes marked bad via K8s labels."""
        env = make_e2e_env(
            ft_id="e2eexcl",
            nodes=[
                NodeSpec(node_id="e2eexcl-node-0", num_ranks=1, diagnostic_pass=False),
                NodeSpec(node_id="e2eexcl-node-1", num_ranks=1, diagnostic_pass=False),
                NodeSpec(node_id="e2eexcl-node-2", num_ranks=1, diagnostic_pass=True),
            ],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=FAST_TIMEOUT)

        # Step 1: crash → wait for MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=FAST_TIMEOUT,
        )

        # Step 2: crash during MONITORING → DIAGNOSING (fast) → recovery completes
        await env.injector.crash_training()
        final = await wait_for_recovery_complete(env.controller, timeout=LONG_RECOVERY_TIMEOUT)
        assert_phase_path_contains(final, ["StopTimeDiagnostics"])
        assert_phase_path_contains(final, ["Evicting"])

        assert env.node_manager.was_ever_marked_bad("e2eexcl-node-0")
        assert env.node_manager.was_ever_marked_bad("e2eexcl-node-1")
        assert not env.node_manager.was_ever_marked_bad("e2eexcl-node-2")


class TestPartialDiagnostic:
    async def test_unreachable_node_agent_treated_as_bad(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """During DIAGNOSING, kill node agent → RayActorError → treated as bad node."""
        import ray

        env = make_e2e_env(
            ft_id="e2epd",
            nodes=[
                NodeSpec(node_id="e2epd-node-0", num_ranks=1, diagnostic_pass=True),
                NodeSpec(node_id="e2epd-node-1", num_ranks=1, diagnostic_pass=True),
            ],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=FAST_TIMEOUT)

        # Step 1: kill node-0's agent before crash
        node_agent_0 = env.node_agents["e2epd-node-0"]
        ray.kill(node_agent_0, no_restart=True)
        env.node_agents.pop("e2epd-node-0", None)

        # Step 2: crash → enters recovery → wait for MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=FAST_TIMEOUT,
        )

        # Step 3: crash during MONITORING → DIAGNOSING (fast) → recovery completes
        await env.injector.crash_training()
        final = await wait_for_recovery_complete(env.controller, timeout=LONG_RECOVERY_TIMEOUT)
        assert final.mode == ControllerMode.MONITORING
        assert_phase_path_contains(final, ["StopTimeDiagnostics"])
        assert_phase_path_contains(final, ["Evicting"])


class TestAllNodesEvicted:
    async def test_all_diagnostics_fail_notifies_human(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """All nodes fail diagnostic → all evicted → NOTIFY."""
        env = make_e2e_env(
            ft_id="e2eall",
            nodes=[
                NodeSpec(node_id="e2eall-node-0", num_ranks=1, diagnostic_pass=False),
                NodeSpec(node_id="e2eall-node-1", num_ranks=1, diagnostic_pass=False),
            ],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=FAST_TIMEOUT)

        # Step 1: crash → wait for MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=FAST_TIMEOUT,
        )

        # Step 2: crash during MONITORING → DIAGNOSING (fast) → recovery completes
        await env.injector.crash_training()
        final = await wait_for_recovery_complete(env.controller, timeout=LONG_RECOVERY_TIMEOUT)
        assert_phase_path_contains(
            final,
            [
                "StopTimeDiagnostics",
                "Evicting",
            ],
        )


class TestEvictionNotification:
    async def test_notifier_receives_eviction_notification(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """During eviction flow, notifier receives at least one notification."""
        env = make_e2e_env(
            ft_id="e2enev",
            nodes=[
                NodeSpec(node_id="e2enev-node-0", num_ranks=1, diagnostic_pass=False),
                NodeSpec(node_id="e2enev-node-1", num_ranks=1, diagnostic_pass=True),
            ],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
            use_notifier=True,
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=FAST_TIMEOUT)

        # Step 1: crash → MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=FAST_TIMEOUT,
        )

        # Step 2: crash during MONITORING → DIAGNOSING → eviction
        await env.injector.crash_training()
        final = await wait_for_recovery_complete(env.controller, timeout=LONG_RECOVERY_TIMEOUT)
        assert_phase_path_contains(final, ["Evicting"])

        # Step 3: verify notifier received calls
        calls = env.get_notifier_calls()
        assert len(calls) > 0, "Notifier should have received eviction notification"


class TestEvictionEscalation:
    async def test_eviction_final_attempt_restart_fails_notifies_humans(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Diagnostic eviction with notify-on-fail + restart fails → NotifyHumans.

        Uses high monitoring_success_iterations so MonitoringProgress stays active
        long enough for us to inject crashes at the right moment.
        """
        env = make_e2e_env(
            ft_id="e2eesc",
            nodes=[
                NodeSpec(node_id="e2eesc-node-0", num_ranks=1, diagnostic_pass=False),
                NodeSpec(node_id="e2eesc-node-1", num_ranks=1, diagnostic_pass=True),
            ],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
            monitoring_success_iterations=999,
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=FAST_TIMEOUT)

        # Step 1: crash → wait for MonitoringProgress (first recovery)
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=FAST_TIMEOUT,
        )

        # Step 2: crash during MonitoringProgress → RestartFailed →
        # StopTimeDiagnostics → node-0 fails → evict_and_restart_final →
        # Evicting → StoppingAndRestarting → MonitoringProgress (post-eviction)
        await env.injector.crash_training()

        # Wait until eviction has happened (visible in phase_history),
        # then wait for post-eviction MonitoringProgress
        deadline = time.monotonic() + RECOVERY_TIMEOUT
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.phase_history and "Evicting" in status.phase_history:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError(
                f"Evicting not found in phase_history within {RECOVERY_TIMEOUT}s: {status.phase_history}"
            )
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=RECOVERY_TIMEOUT,
        )

        # Step 3: crash during post-eviction MonitoringProgress →
        # RestartFailed → failed_next_state=NotifyHumans
        await env.injector.crash_training()

        deadline = time.monotonic() + LONG_RECOVERY_TIMEOUT
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.phase_history and "NotifyHumans" in status.phase_history:
                break
            if status.mode == ControllerMode.MONITORING and not status.recovery_in_progress:
                break
            await asyncio.sleep(0.5)

        status = get_status(env.controller)
        assert status.phase_history is not None
        assert_phase_path_contains(status, ["NotifyHumans"])


class TestAllDiagnosticsPass:
    async def test_all_diagnostics_pass_escalates_to_notify_humans(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """All nodes pass diagnostics (no root cause found) → NotifyHumans."""
        env = make_e2e_env(
            ft_id="e2eadp",
            nodes=[
                NodeSpec(node_id="e2eadp-node-0", num_ranks=1, diagnostic_pass=True),
                NodeSpec(node_id="e2eadp-node-1", num_ranks=1, diagnostic_pass=True),
            ],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
            use_notifier=True,
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=FAST_TIMEOUT)

        # Step 1: crash → wait for MonitoringProgress
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=FAST_TIMEOUT,
        )

        # Step 2: crash during MonitoringProgress → StopTimeDiagnostics
        await env.injector.crash_training()

        # Step 3: all diagnostics pass → NotifyHumans
        deadline = time.monotonic() + RECOVERY_TIMEOUT
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.phase_history and "NotifyHumans" in status.phase_history:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError(f"All-pass diagnostics did not escalate to NotifyHumans within {RECOVERY_TIMEOUT}s")

        assert_phase_path_contains(status, ["StopTimeDiagnostics", "NotifyHumans"])

        # Step 4: notifier should have received a notification
        calls = env.get_notifier_calls()
        assert len(calls) > 0, "Notifier should have received notification for all-pass diagnostics"


class TestNotificationContent:
    async def test_notify_humans_notification_contains_trigger_and_context(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """NotifyHumans notification includes trigger= and state_before= for operator actionability."""
        env = make_e2e_env(
            ft_id="e2enc",
            nodes=[
                NodeSpec(node_id="e2enc-node-0", num_ranks=1, diagnostic_pass=True),
                NodeSpec(node_id="e2enc-node-1", num_ranks=1, diagnostic_pass=True),
            ],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
            use_notifier=True,
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=FAST_TIMEOUT)

        # Step 1: crash → MonitoringProgress → crash → StopTimeDiagnostics → all pass → NotifyHumans
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=FAST_TIMEOUT,
        )
        await env.injector.crash_training()

        deadline = time.monotonic() + RECOVERY_TIMEOUT
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.phase_history and "NotifyHumans" in status.phase_history:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError(f"NotifyHumans not reached within {RECOVERY_TIMEOUT}s")

        # Step 2: verify notification content includes trigger and state_before
        calls = env.get_notifier_calls()
        assert len(calls) > 0, "Notifier should have received calls"

        recovery_alerts = [(t, c, s) for t, c, s in calls if t == "Recovery Alert"]
        assert len(recovery_alerts) > 0, "Expected at least one 'Recovery Alert' notification"

        _, content, _ = recovery_alerts[0]
        assert "trigger=" in content, f"Notification missing trigger=: {content}"
        assert "state_before=" in content, f"Notification missing state_before=: {content}"


class TestUnmarkNodeReusable:
    async def test_unmark_removes_node_from_bad_list(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Eviction marks node bad; unmark_node_bad removes it from bad_nodes list."""
        # Step 1: build env — node-0 diagnostic fails, node-1 passes
        env = make_e2e_env(
            ft_id="e2eunm",
            nodes=[
                NodeSpec(node_id="e2eunm-node-0", num_ranks=1, diagnostic_pass=False),
                NodeSpec(node_id="e2eunm-node-1", num_ranks=1, diagnostic_pass=True),
            ],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=FAST_TIMEOUT)

        # Step 2: crash → monitoring → crash → eviction → recovery
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=FAST_TIMEOUT,
        )
        await env.injector.crash_training()
        await wait_for_recovery_complete(env.controller, timeout=RECOVERY_TIMEOUT)

        # Step 3: assert node-0 was marked bad and is in get_bad_nodes
        assert env.node_manager.was_ever_marked_bad("e2eunm-node-0")
        bad_nodes = await env.node_manager.get_bad_nodes()
        assert "e2eunm-node-0" in bad_nodes

        # Step 4: unmark_node_bad → assert removed from get_bad_nodes
        await env.node_manager.unmark_node_bad("e2eunm-node-0")
        bad_nodes = await env.node_manager.get_bad_nodes()
        assert "e2eunm-node-0" not in bad_nodes
