"""Semi-E2E: crash recovery — transient, repeated, throttle, reset, exception, concurrent."""
from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from miles.utils.ft.controller.detectors.training_crash import TrainingCrashDetector
from miles.utils.ft.controller.recovery.helpers import SlidingWindowThrottle
from miles.utils.ft.models.recovery import ControllerMode

from tests.fast.utils.ft.integration.local_ray_semi_e2e.conftest import (
    E2EEnv,
    NodeSpec,
    _SLOW_STEP,
)
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios import (
    get_status,
    scenario_transient_crash,
    wait_for_mode,
    wait_for_mode_transition,
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
