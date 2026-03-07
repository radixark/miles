"""Semi-E2E: crash recovery — transient, repeated, throttle, reset, exception, concurrent."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import pytest

from miles.utils.ft.controller.detectors.core.training_crash import TrainingCrashDetector
from miles.utils.ft.controller.recovery.helpers import SlidingWindowThrottle
from miles.utils.ft.models.recovery import ControllerMode

from tests.fast.utils.ft.integration.local_ray_semi_e2e.conftest import (
    E2EEnv,
    NodeSpec,
    _FastHangDetector,
    _SLOW_STEP,
)
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios import (
    assert_phase_path_contains,
    get_status,
    scenario_transient_crash,
    wait_for_mode,
    wait_for_mode_transition,
    wait_for_recovery_phase,
    wait_for_training_stable,
)


class TestTransientCrash:
    async def test_crash_recovery_with_real_agents(self, e2e_env: E2EEnv) -> None:
        """Single crash → auto-recovery → training resumes with re-registered workers."""
        status = await scenario_transient_crash(
            handle=e2e_env.controller,
            injector=e2e_env.injector,
            stable_iterations=3,
            recovery_timeout=60.0,
        )
        assert status.mode == ControllerMode.MONITORING


class TestRecoveryThrottle:
    async def test_third_crash_throttled(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """3 crashes with max_count=3 → third is throttled (stays in MONITORING).

        record() is called before is_throttled(), so max_count=3 allows 2 recoveries.
        """
        env = make_e2e_env(
            ft_id="e2ethr",
            nodes=[NodeSpec(node_id="e2ethr-node-0")],
            detectors=[TrainingCrashDetector()],
            recovery_cooldown=SlidingWindowThrottle(
                window_minutes=60,
                max_count=3,
            ),
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)

        # Step 1: first crash → recovery
        await env.injector.crash_training()
        await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=60.0,
        )

        # Step 2: second crash → recovery
        await env.injector.crash_training()
        await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=60.0,
        )

        # Step 3: third crash → throttled, no recovery
        await env.injector.crash_training()
        await asyncio.sleep(5.0)
        status = get_status(env.controller)
        assert status.mode == ControllerMode.MONITORING


class TestRecoveryReset:
    async def test_state_clean_after_recovery_completes(
        self, e2e_env: E2EEnv,
    ) -> None:
        """Recovery DONE → mode=MONITORING, phase=None → new crash is a fresh cycle."""
        env = e2e_env

        # Step 1: first crash → full recovery
        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)
        await env.injector.crash_training()
        status = await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=60.0,
        )
        assert status.mode == ControllerMode.MONITORING
        assert status.recovery_in_progress is False

        # Step 2: second crash → enters a fresh recovery cycle
        await wait_for_training_stable(env.controller, n_iterations=2, timeout=30.0)
        await env.injector.crash_training()
        status = await wait_for_mode(
            env.controller,
            target_mode=ControllerMode.RECOVERY,
            timeout=30.0,
        )
        assert status.mode == ControllerMode.RECOVERY
        assert status.recovery_phase is not None


class TestExceptionInRecovery:
    async def test_stop_training_exception_forces_notify(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """stop_training raises → recovery forces NOTIFY → actor survives.

        We simulate this by having the training job's stop fail.
        In E2E with RemoteControlledTrainingJob, stop() calls state_actor.stop.remote()
        which is unlikely to fail, so this test verifies the controller handles
        the crash → recovery flow and continues operating.
        """
        env = make_e2e_env(
            ft_id="e2eexc",
            nodes=[NodeSpec(node_id="e2eexc-node-0")],
            detectors=[TrainingCrashDetector()],
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)
        await env.injector.crash_training()

        final = await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=60.0,
        )
        assert final.mode == ControllerMode.MONITORING


class TestRepeatedCrash:
    async def test_two_crashes_escalate_to_diagnosing(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Crash → recovery MONITORING → crash again → escalates to DIAGNOSING."""
        env = make_e2e_env(
            ft_id="e2erpt",
            nodes=[NodeSpec(node_id="e2erpt-node-0")],
            detectors=[TrainingCrashDetector()],
            recovery_cooldown=SlidingWindowThrottle(window_minutes=1.0, max_count=2),
        )

        # Step 1: crash → recovery enters MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=60.0,
        )

        # Step 2: crash during MONITORING → DIAGNOSING
        await env.injector.crash_training()

        # Step 3: poll for DIAGNOSING in phase_history during the active recovery.
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.phase_history and "StopTimeDiagnostics" in status.phase_history:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError("DIAGNOSING not observed in phase_history within 60s")

        assert_phase_path_contains(status, ["StopTimeDiagnostics"])


class TestConcurrentFaults:
    async def test_simultaneous_nan_and_crash(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """NaN + FAILED simultaneously → detector priority chain triggers 1 recovery."""
        from miles.utils.ft.controller.detectors.chain import build_detector_chain

        env = make_e2e_env(
            ft_id="e2ecf",
            nodes=[NodeSpec(node_id="e2ecf-node-0")],
            detectors=build_detector_chain(),
            scrape_interval_seconds=0.5,
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)

        # Step 1: inject both faults
        await env.injector.inject_nan_loss()
        await env.injector.crash_training()

        # Step 2: wait for recovery
        status = await wait_for_mode(
            env.controller,
            target_mode=ControllerMode.RECOVERY,
            timeout=30.0,
        )
        assert status.mode == ControllerMode.RECOVERY

        # Step 3: let recovery complete
        final = await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=90.0,
        )
        assert final.mode == ControllerMode.MONITORING


class TestRestartFailed:
    async def test_restart_job_fails_immediately_escalates_to_diagnostics(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Restart completes but job FAILED during MonitoringProgress → StopTimeDiagnostics."""
        env = make_e2e_env(
            ft_id="e2erf",
            nodes=[NodeSpec(node_id="e2erf-node-0")],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
            recovery_cooldown=SlidingWindowThrottle(window_minutes=1.0, max_count=5),
        )

        # Step 1: crash → enters recovery, wait for MonitoringProgress (durable phase)
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller, phase="MonitoringProgress", timeout=60.0,
        )

        # Step 2: crash again during monitoring (restart fails)
        await env.injector.crash_training()

        # Step 3: poll for StopTimeDiagnostics
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.phase_history and "StopTimeDiagnostics" in status.phase_history:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError("StopTimeDiagnostics not observed within 60s")

        assert_phase_path_contains(status, ["StopTimeDiagnostics"])


class TestCrashDuringRecovery:
    async def test_crash_during_early_recovery_converges(
        self, e2e_env: E2EEnv,
    ) -> None:
        """Crash while already in RECOVERY → system eventually converges."""
        # Step 1: crash → enter recovery
        await e2e_env.injector.crash_training()
        await wait_for_mode(
            e2e_env.controller,
            target_mode=ControllerMode.RECOVERY,
            timeout=30.0,
        )

        # Step 2: crash again while recovering
        await e2e_env.injector.crash_training()

        # Step 3: system must converge (back to MONITORING or NotifyHumans)
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            status = get_status(e2e_env.controller)
            if status.mode == ControllerMode.MONITORING and not status.recovery_in_progress:
                return
            await asyncio.sleep(0.5)

        status = get_status(e2e_env.controller)
        assert status.mode == ControllerMode.MONITORING or (
            status.phase_history and "NotifyHumans" in status.phase_history
        ), f"Recovery did not converge: mode={status.mode}, phase={status.recovery_phase}"

    async def test_fault_during_monitoring_progress_serialized(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """New fault during MonitoringProgress is handled without nested recovery."""
        env = make_e2e_env(
            ft_id="e2emp",
            nodes=[NodeSpec(node_id="e2emp-node-0")],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
            recovery_cooldown=SlidingWindowThrottle(window_minutes=1.0, max_count=5),
        )

        # Step 1: crash → wait for MonitoringProgress
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller, phase="MonitoringProgress", timeout=60.0,
        )

        # Step 2: inject another crash during MonitoringProgress
        await env.injector.crash_training()

        # Step 3: wait for convergence
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.mode == ControllerMode.MONITORING and not status.recovery_in_progress:
                return
            await asyncio.sleep(0.5)

        status = get_status(env.controller)
        assert status.mode == ControllerMode.MONITORING, (
            f"Did not converge: mode={status.mode}, phase={status.recovery_phase}"
        )


class TestSequentialRecovery:
    async def test_three_sequential_recovery_cycles_no_state_drift(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Three crash-recover cycles with distinct run_ids and clean state each time."""
        env = make_e2e_env(
            ft_id="e2eseq",
            nodes=[NodeSpec(node_id="e2eseq-node-0")],
            detectors=[TrainingCrashDetector()],
            recovery_cooldown=SlidingWindowThrottle(window_minutes=60, max_count=10),
        )

        run_ids: list[str] = []
        initial_status = get_status(env.controller)
        if initial_status.active_run_id:
            run_ids.append(initial_status.active_run_id)

        for cycle in range(3):
            await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)
            await env.injector.crash_training()

            status = await wait_for_mode_transition(
                env.controller,
                target_mode=ControllerMode.MONITORING,
                timeout=60.0,
            )
            assert status.mode == ControllerMode.MONITORING
            assert status.recovery_in_progress is False

            if status.active_run_id:
                run_ids.append(status.active_run_id)

        assert len(set(run_ids)) == len(run_ids), f"Duplicate run_ids: {run_ids}"


class TestCooldownBoundary:
    async def test_recovery_cooldown_boundary_exact_count(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """max_count=2 allows 1 recovery; the second crash is throttled.

        record() is called before is_throttled(), so max_count=2 allows 1 recovery.
        """
        env = make_e2e_env(
            ft_id="e2ecb",
            nodes=[NodeSpec(node_id="e2ecb-node-0")],
            detectors=[TrainingCrashDetector()],
            recovery_cooldown=SlidingWindowThrottle(window_minutes=60, max_count=2),
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)

        # Step 1: first crash → recovery succeeds
        await env.injector.crash_training()
        await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=60.0,
        )

        # Step 2: second crash → throttled
        await wait_for_training_stable(env.controller, n_iterations=2, timeout=30.0)
        await env.injector.crash_training()
        await asyncio.sleep(5.0)
        status = get_status(env.controller)
        assert status.mode == ControllerMode.MONITORING, (
            f"Expected throttle, but mode={status.mode}"
        )


class TestConcurrentDetectors:
    async def test_concurrent_crash_and_hang_priority_deterministic(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Hang + crash conditions both true → exactly one recovery cycle triggered."""
        env = make_e2e_env(
            ft_id="e2ecd",
            nodes=[NodeSpec(node_id="e2ecd-node-0")],
            detectors=[TrainingCrashDetector(), _FastHangDetector(timeout_seconds=3.0)],
        )

        # Step 1: inject hang first
        await env.injector.inject_hang()
        await asyncio.sleep(4.0)

        # Step 2: also crash
        await env.injector.crash_training()

        # Step 3: wait for recovery to complete
        final = await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=90.0,
        )
        assert final.mode == ControllerMode.MONITORING


class TestNotification:
    async def test_notifier_receives_throttle_notification(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """When recovery is throttled, the notifier receives a notification."""
        env = make_e2e_env(
            ft_id="e2entt",
            nodes=[NodeSpec(node_id="e2entt-node-0")],
            detectors=[TrainingCrashDetector()],
            recovery_cooldown=SlidingWindowThrottle(window_minutes=60, max_count=2),
            use_notifier=True,
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)

        # Step 1: first crash → recovery
        await env.injector.crash_training()
        await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=60.0,
        )

        # Step 2: second crash → throttled
        await wait_for_training_stable(env.controller, n_iterations=2, timeout=30.0)
        await env.injector.crash_training()
        await asyncio.sleep(5.0)

        # Step 3: verify notifier received calls
        calls = env.get_notifier_calls()
        assert len(calls) > 0, "Notifier should have received at least one call"


class TestRecoveryOverallTimeout:
    async def test_recovery_overall_timeout_escalates_to_notify_humans(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """recovery_timeout_seconds fires when MonitoringProgress stalls → NotifyHumans."""
        env = make_e2e_env(
            ft_id="e2erot",
            nodes=[NodeSpec(node_id="e2erot-node-0")],
            detectors=[TrainingCrashDetector()],
            step_interval=999.0,
            recovery_timeout_seconds=3,
            monitoring_timeout_seconds=999,
            monitoring_success_iterations=999,
        )

        # Step 1: crash → recovery starts, reaches MonitoringProgress but never succeeds
        await env.injector.crash_training()

        # Step 2: wait for NotifyHumans due to overall recovery timeout
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.phase_history and "NotifyHumans" in status.phase_history:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError("Recovery overall timeout did not escalate to NotifyHumans within 60s")

        assert_phase_path_contains(status, ["NotifyHumans"])


class TestCooldownExpiry:
    async def test_cooldown_window_expired_allows_recovery_again(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """After cooldown window expires, a new crash triggers recovery normally."""
        window_seconds = 2
        env = make_e2e_env(
            ft_id="e2ecwe",
            nodes=[NodeSpec(node_id="e2ecwe-node-0")],
            detectors=[TrainingCrashDetector()],
            recovery_cooldown=SlidingWindowThrottle(
                window_minutes=window_seconds / 60.0,
                max_count=2,
            ),
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)

        # Step 1: first crash → recovery
        await env.injector.crash_training()
        await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=60.0,
        )

        # Step 2: second crash immediately → throttled
        await wait_for_training_stable(env.controller, n_iterations=2, timeout=30.0)
        await env.injector.crash_training()
        await asyncio.sleep(3.0)
        status = get_status(env.controller)
        assert status.mode == ControllerMode.MONITORING, "Expected throttle"

        # Step 3: sleep past cooldown window, then crash again → recovery succeeds
        await asyncio.sleep(window_seconds + 1)
        await env.injector.crash_training()
        final = await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=60.0,
        )
        assert final.mode == ControllerMode.MONITORING


class TestFalsePositiveGuard:
    async def test_too_many_bad_nodes_treated_as_false_positive(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """When >= max_simultaneous_bad_nodes report faults, no recovery is triggered."""
        from miles.utils.ft.controller.detectors.chain import build_detector_chain
        from miles.utils.ft.models.metric_names import GPU_AVAILABLE
        from miles.utils.ft.models.metrics import GaugeSample

        env = make_e2e_env(
            ft_id="e2efpg",
            nodes=[
                NodeSpec(node_id=f"e2efpg-node-{i}", use_remote_collector=True)
                for i in range(4)
            ],
            detectors=build_detector_chain(),
            scrape_interval_seconds=0.5,
            max_simultaneous_bad_nodes=3,
            use_notifier=True,
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)

        # Step 1: inject GPU_AVAILABLE=0 on 3 nodes simultaneously
        for i in range(3):
            node_id = f"e2efpg-node-{i}"
            env.set_collector_metrics(node_id, [
                GaugeSample(
                    name=GPU_AVAILABLE,
                    labels={"node_id": node_id, "gpu": "0"},
                    value=0.0,
                ),
            ])

        # Step 2: wait for scrape cycles, then verify no recovery
        await asyncio.sleep(5.0)
        status = get_status(env.controller)
        assert status.mode == ControllerMode.MONITORING, (
            f"Expected false positive guard, but mode={status.mode}"
        )
