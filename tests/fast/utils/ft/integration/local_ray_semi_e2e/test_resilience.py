"""Semi-E2E: resilience — agent without controller, fire-and-forget, detector faults."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from unittest.mock import patch

import ray
from tests.fast.utils.ft.helpers.controller_fakes import CrashingDetector
from tests.fast.utils.ft.integration.local_ray_semi_e2e.conftest import _SLOW_STEP, E2EEnv, NodeSpec
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios import (
    assert_phase_path_contains,
    get_status,
    wait_for_mode_transition,
    wait_for_recovery_phase,
    wait_for_training_stable,
)

from miles.utils.ft.controller.detectors.core.training_crash import TrainingCrashDetector
from miles.utils.ft.models.recovery import ControllerMode
from miles.utils.ft.protocols.platform import ft_controller_actor_name


class TestAgentWithoutController:
    async def test_rank_agent_graceful_degrade(self, local_ray: None) -> None:
        """Creating FtTrainingRankAgent when controller doesn't exist → no crash."""
        os.environ["MILES_FT_ID"] = "nonexistent"
        os.environ["MILES_FT_TRAINING_RUN_ID"] = "fake-run"

        from miles.utils.ft.agents.core.training_rank_agent import FtTrainingRankAgent

        with patch("socket.gethostname", return_value="test-node"):
            agent = FtTrainingRankAgent(rank=0, world_size=1)

        agent.step()
        agent.shutdown()


class TestFireAndForget:
    async def test_log_step_after_controller_death(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Kill controller → worker continues log_step → doesn't crash/block."""
        env = make_e2e_env(
            ft_id="e2eff",
            nodes=[NodeSpec(node_id="e2eff-node-0")],
            detectors=[TrainingCrashDetector()],
        )

        await wait_for_training_stable(env.controller, n_iterations=2, timeout=30.0)

        # Step 1: kill controller
        controller_name = ft_controller_actor_name(env.ft_id)
        try:
            ray.get(env.controller.shutdown.remote(), timeout=5)
        except Exception:
            pass
        try:
            ray.kill(ray.get_actor(controller_name), no_restart=True)
        except (ValueError, Exception):
            pass

        # Step 2: wait a bit - worker should continue running without crashing
        await asyncio.sleep(2.0)

        # Step 3: verify workers are still alive
        for worker in env.workers:
            iteration = ray.get(worker.get_iteration.remote(), timeout=5)
            assert iteration > 0


class TestRestartStepperException:
    async def test_restart_stepper_exception_forces_notify_humans(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Repeated crash with all-pass diagnostics → NotifyHumans (no root cause).

        When the recovery stepper cannot identify a root cause (all diagnostics
        pass), it escalates to NotifyHumans to alert operators.
        """
        env = make_e2e_env(
            ft_id="e2erse",
            nodes=[NodeSpec(node_id="e2erse-node-0")],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
            monitoring_success_iterations=999,
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=30.0)

        # Step 1: crash → recovery → MonitoringProgress (stays active)
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=30.0,
        )

        # Step 2: crash during MonitoringProgress → RestartFailed →
        # StopTimeDiagnostics → single node passes → no root cause → NotifyHumans
        await env.injector.crash_training()

        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.phase_history and "NotifyHumans" in status.phase_history:
                break
            if status.mode == ControllerMode.MONITORING and not status.recovery_in_progress:
                break
            await asyncio.sleep(0.5)

        status = get_status(env.controller)
        assert_phase_path_contains(status, ["StopTimeDiagnostics", "NotifyHumans"])


class _CrashingNotifier:
    """Notifier whose send() always raises. Serializable via cloudpickle."""

    async def send(self, title: str, content: str, severity: str) -> None:
        raise RuntimeError("notifier crash for testing")

    async def aclose(self) -> None:
        pass


class TestNotifierResilience:
    async def test_notifier_send_exception_does_not_break_recovery(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Notifier raises on send() → safe_notify catches → recovery completes."""
        from miles.utils.ft.controller.recovery.helpers import SlidingWindowThrottle

        env = make_e2e_env(
            ft_id="e2enr",
            nodes=[NodeSpec(node_id="e2enr-node-0")],
            detectors=[TrainingCrashDetector()],
            recovery_cooldown=SlidingWindowThrottle(window_minutes=60, max_count=2),
            notifier_override=_CrashingNotifier(),
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)

        # Step 1: first crash → recovery (notifier not called on normal recovery)
        await env.injector.crash_training()
        await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=60.0,
        )

        # Step 2: second crash → throttled → notifier.send() called → raises → controller survives
        await wait_for_training_stable(env.controller, n_iterations=2, timeout=30.0)
        await env.injector.crash_training()
        await asyncio.sleep(5.0)

        # Step 3: controller is still alive and functional
        status = get_status(env.controller)
        assert status.mode == ControllerMode.MONITORING


class TestDetectorResilience:
    async def test_detector_exception_does_not_crash_controller(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """A crashing detector is skipped; subsequent detectors still fire normally."""
        env = make_e2e_env(
            ft_id="e2edr",
            nodes=[NodeSpec(node_id="e2edr-node-0")],
            detectors=[CrashingDetector(), TrainingCrashDetector()],
        )

        # Step 1: let training run for a while with CrashingDetector throwing every tick
        await wait_for_training_stable(env.controller, n_iterations=5, timeout=30.0)

        # Step 2: crash training → TrainingCrashDetector should still work
        await env.injector.crash_training()
        final = await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=60.0,
        )
        assert final.mode == ControllerMode.MONITORING
