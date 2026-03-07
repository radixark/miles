"""Semi-E2E: status — consistency, run_id switch, metric scrape, no false positive, monotonicity."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

import ray

from miles.utils.ft.models.recovery import ControllerMode

from tests.fast.utils.ft.integration.local_ray_semi_e2e.conftest import (
    E2EEnv,
    NodeSpec,
    _FastHangDetector,
)
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios import (
    get_status,
    scenario_no_false_positive,
    wait_for_mode_transition,
    wait_for_training_stable,
)


class TestStatusConsistency:
    async def test_status_snapshots_internally_consistent(
        self, e2e_env: E2EEnv,
    ) -> None:
        """High-frequency polling during recovery → every snapshot is internally consistent."""
        env = e2e_env

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)
        await env.injector.crash_training()

        snapshots: list[Any] = []
        deadline = time.monotonic() + 60.0

        while time.monotonic() < deadline:
            status = get_status(env.controller)
            snapshots.append(status)

            if status.mode == ControllerMode.MONITORING and not status.recovery_in_progress:
                if len(snapshots) > 5:
                    break
            await asyncio.sleep(0.05)

        assert len(snapshots) > 5, "Not enough snapshots collected"

        for s in snapshots:
            if s.mode == ControllerMode.RECOVERY:
                assert s.recovery_in_progress is True
                assert s.recovery_phase is not None
            elif s.mode == ControllerMode.MONITORING:
                if s.recovery_in_progress:
                    assert s.recovery_phase is not None


class TestRunIdSwitch:
    async def test_new_run_id_reregistration_and_metric_isolation(
        self, e2e_env: E2EEnv,
    ) -> None:
        """After recovery, worker re-registers with new run_id and metrics use new run."""
        env = e2e_env

        pre_status = get_status(env.controller)
        pre_run_id = pre_status.active_run_id

        # Step 1: crash → recovery → new run
        await env.injector.crash_training()
        final = await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=60.0,
        )

        # Step 2: verify new run_id
        post_run_id = final.active_run_id
        assert post_run_id is not None
        assert post_run_id != pre_run_id

        # Step 3: verify iteration progresses under new run
        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)


class TestMetricScrapeE2E:
    async def test_prometheus_exporter_to_metric_store_pipeline(
        self, e2e_full_detector_env: E2EEnv,
    ) -> None:
        """Worker pushes iteration → controller scrapes → status.latest_iteration tracks it."""
        env = e2e_full_detector_env

        await wait_for_training_stable(env.controller, n_iterations=5, timeout=30.0)
        status = get_status(env.controller)
        assert status.latest_iteration is not None
        assert status.latest_iteration >= 5


class TestNoFalsePositive:
    async def test_healthy_training_stays_in_monitoring(
        self,
        e2e_env: E2EEnv,
    ) -> None:
        """Healthy training with real workers never triggers spurious recovery."""
        status = await scenario_no_false_positive(
            handle=e2e_env.controller,
            observation_iterations=10,
            poll_interval=0.2,
        )

        assert status.mode == ControllerMode.MONITORING
        assert status.recovery_in_progress is False

    async def test_metric_scrape_jitter_no_false_hang(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Low-frequency scrape (5s) with fast ticks (0.1s) does not false-trigger hang."""
        env = make_e2e_env(
            ft_id="e2esfj",
            nodes=[NodeSpec(node_id="e2esfj-node-0")],
            detectors=[_FastHangDetector(timeout_seconds=10.0)],
            scrape_interval_seconds=5.0,
            tick_interval=0.1,
        )

        status = await scenario_no_false_positive(
            handle=env.controller,
            observation_iterations=10,
            poll_interval=0.5,
        )

        assert status.mode == ControllerMode.MONITORING
        assert status.recovery_in_progress is False


class TestStaleMetrics:
    async def test_log_step_with_stale_run_id_discarded(
        self, e2e_env: E2EEnv,
    ) -> None:
        """log_step with old run_id after recovery does not pollute iteration counter."""
        env = e2e_env
        old_run_id = get_status(env.controller).active_run_id

        # Step 1: crash → recovery → new run
        await env.injector.crash_training()
        status = await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=60.0,
        )
        new_run_id = status.active_run_id
        assert new_run_id != old_run_id

        # Step 2: inject a stale log_step with the old run_id
        ray.get(env.controller.log_step.remote(
            run_id=old_run_id, step=999999,
            metrics={"iteration": 999999.0},
        ), timeout=5)

        # Step 3: verify iteration is not polluted
        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)
        final = get_status(env.controller)
        assert final.latest_iteration is not None
        assert final.latest_iteration < 999999


class TestRecoveryPhaseMonotonicity:
    async def test_recovery_phase_order_monotonic_during_polling(
        self, e2e_env: E2EEnv,
    ) -> None:
        """High-frequency polling during recovery: mode transitions are orderly."""
        env = e2e_env

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)
        await env.injector.crash_training()

        snapshots: list[Any] = []
        deadline = time.monotonic() + 60.0

        while time.monotonic() < deadline:
            status = get_status(env.controller)
            snapshots.append(status)

            if status.mode == ControllerMode.MONITORING and not status.recovery_in_progress:
                if len(snapshots) > 5:
                    break
            await asyncio.sleep(0.05)

        assert len(snapshots) > 5, "Not enough snapshots collected"

        saw_recovery = False
        saw_monitoring_after_recovery = False
        for s in snapshots:
            if s.mode == ControllerMode.RECOVERY:
                saw_recovery = True
                assert not saw_monitoring_after_recovery, (
                    "Mode went back to RECOVERY after returning to MONITORING"
                )
            elif s.mode == ControllerMode.MONITORING and saw_recovery:
                saw_monitoring_after_recovery = True

        assert saw_recovery, "Never observed RECOVERY mode"
        assert saw_monitoring_after_recovery, "Never returned to MONITORING after recovery"
