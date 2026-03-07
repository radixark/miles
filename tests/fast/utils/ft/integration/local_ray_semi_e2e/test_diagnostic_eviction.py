"""Semi-E2E: diagnostic eviction — bad node eviction, excluded nodes, partial, all-evicted."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from miles.utils.ft.controller.detectors.training_crash import TrainingCrashDetector
from miles.utils.ft.controller.recovery.helpers import SlidingWindowThrottle
from miles.utils.ft.models.recovery import ControllerMode

from tests.fast.utils.ft.integration.local_ray_semi_e2e.conftest import (
    E2EEnv,
    NodeSpec,
    _SLOW_STEP,
)
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios import (
    assert_phase_path_contains,
    get_status,
    wait_for_recovery_complete,
    wait_for_recovery_phase,
    wait_for_training_stable,
)


class TestDiagnosticEviction:
    async def test_diagnostic_failure_evicts_bad_node(
        self, make_e2e_env: Callable[..., E2EEnv],
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

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=30.0)

        # Step 1: crash → wait for recovery MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=30.0,
        )

        # Step 2: crash during MONITORING → DIAGNOSING (fast) → recovery completes
        await env.injector.crash_training()
        final = await wait_for_recovery_complete(env.controller, timeout=60.0)

        assert final.mode == ControllerMode.MONITORING
        assert_phase_path_contains(final, [
            "StopTimeDiagnostics",
            "Evicting",
            "RecoveryDone",
        ])


class TestEvictionExcludedNodes:
    async def test_multi_node_eviction_passes_correct_excluded_ids(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """2 bad nodes out of 3 → excluded_node_ids contains both."""
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

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=30.0)

        # Step 1: crash → wait for MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=30.0,
        )

        # Step 2: crash during MONITORING → DIAGNOSING (fast) → recovery completes
        await env.injector.crash_training()
        final = await wait_for_recovery_complete(env.controller, timeout=90.0)
        assert_phase_path_contains(final, ["StopTimeDiagnostics"])
        assert_phase_path_contains(final, ["Evicting"])


class TestPartialDiagnostic:
    async def test_unreachable_node_agent_treated_as_bad(
        self, make_e2e_env: Callable[..., E2EEnv],
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

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=30.0)

        # Step 1: kill node-0's agent before crash
        node_agent_0 = env.node_agents["e2epd-node-0"]
        ray.kill(node_agent_0, no_restart=True)
        env.node_agents.pop("e2epd-node-0", None)

        # Step 2: crash → enters recovery → wait for MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=30.0,
        )

        # Step 3: crash during MONITORING → DIAGNOSING (fast) → recovery completes
        await env.injector.crash_training()
        final = await wait_for_recovery_complete(env.controller, timeout=90.0)
        assert final.mode == ControllerMode.MONITORING
        assert_phase_path_contains(final, ["StopTimeDiagnostics"])
        assert_phase_path_contains(final, ["Evicting"])


class TestAllNodesEvicted:
    async def test_all_diagnostics_fail_notifies_human(
        self, make_e2e_env: Callable[..., E2EEnv],
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

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=30.0)

        # Step 1: crash → wait for MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=30.0,
        )

        # Step 2: crash during MONITORING → DIAGNOSING (fast) → recovery completes
        await env.injector.crash_training()
        final = await wait_for_recovery_complete(env.controller, timeout=90.0)
        assert_phase_path_contains(final, [
            "StopTimeDiagnostics",
            "Evicting",
            "RecoveryDone",
        ])


class TestEvictionNotification:
    async def test_notifier_receives_eviction_notification(
        self, make_e2e_env: Callable[..., E2EEnv],
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

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=30.0)

        # Step 1: crash → MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller, phase="MonitoringProgress", timeout=30.0,
        )

        # Step 2: crash during MONITORING → DIAGNOSING → eviction
        await env.injector.crash_training()
        final = await wait_for_recovery_complete(env.controller, timeout=90.0)
        assert_phase_path_contains(final, ["Evicting"])

        # Step 3: verify notifier received calls
        calls = env.get_notifier_calls()
        assert len(calls) > 0, "Notifier should have received eviction notification"


class TestEvictionEscalation:
    async def test_eviction_final_attempt_restart_fails_notifies_humans(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Diagnostic eviction with notify-on-fail + restart fails → NotifyHumans."""
        env = make_e2e_env(
            ft_id="e2eesc",
            nodes=[
                NodeSpec(node_id="e2eesc-node-0", num_ranks=1, diagnostic_pass=False),
                NodeSpec(node_id="e2eesc-node-1", num_ranks=1, diagnostic_pass=True),
            ],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
            recovery_cooldown=SlidingWindowThrottle(window_minutes=1.0, max_count=10),
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=30.0)

        # Step 1: crash → wait for MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller, phase="MonitoringProgress", timeout=30.0,
        )

        # Step 2: crash during MONITORING → DIAGNOSING → eviction (notify on fail)
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller, phase="MonitoringProgress", timeout=90.0,
        )

        # Step 3: crash during post-eviction MONITORING → should go to NotifyHumans
        await env.injector.crash_training()

        deadline = time.monotonic() + 90.0
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
        self, make_e2e_env: Callable[..., E2EEnv],
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

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=30.0)

        # Step 1: crash → wait for MonitoringProgress
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller, phase="MonitoringProgress", timeout=30.0,
        )

        # Step 2: crash during MonitoringProgress → StopTimeDiagnostics
        await env.injector.crash_training()

        # Step 3: all diagnostics pass → NotifyHumans
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.phase_history and "NotifyHumans" in status.phase_history:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError("All-pass diagnostics did not escalate to NotifyHumans within 60s")

        assert_phase_path_contains(status, ["StopTimeDiagnostics", "NotifyHumans"])

        # Step 4: notifier should have received a notification
        calls = env.get_notifier_calls()
        assert len(calls) > 0, "Notifier should have received notification for all-pass diagnostics"


class TestNotificationContent:
    async def test_notify_humans_notification_contains_trigger_and_context(
        self, make_e2e_env: Callable[..., E2EEnv],
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

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=30.0)

        # Step 1: crash → MonitoringProgress → crash → StopTimeDiagnostics → all pass → NotifyHumans
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller, phase="MonitoringProgress", timeout=30.0,
        )
        await env.injector.crash_training()

        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.phase_history and "NotifyHumans" in status.phase_history:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError("NotifyHumans not reached within 60s")

        # Step 2: verify notification content includes trigger and state_before
        calls = env.get_notifier_calls()
        assert len(calls) > 0, "Notifier should have received calls"

        recovery_alerts = [(t, c, s) for t, c, s in calls if t == "Recovery Alert"]
        assert len(recovery_alerts) > 0, "Expected at least one 'Recovery Alert' notification"

        _, content, _ = recovery_alerts[0]
        assert "trigger=" in content, f"Notification missing trigger=: {content}"
        assert "state_before=" in content, f"Notification missing state_before=: {content}"
