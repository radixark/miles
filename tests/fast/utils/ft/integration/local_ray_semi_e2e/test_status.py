"""Semi-E2E: status — consistency, run_id switch, metric scrape, no false positive, monotonicity."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

import ray
from tests.fast.utils.ft.integration.conftest import FAST_TIMEOUT, RECOVERY_TIMEOUT
from tests.fast.utils.ft.integration.local_ray_semi_e2e.conftest import _FAST_SCRAPE, E2EEnv, NodeSpec
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios import (
    get_status,
    scenario_no_false_positive,
    wait_for_mode_transition,
    wait_for_training_stable,
)
from tests.fast.utils.ft.utils.controller_fakes import FastHangDetector

from miles.utils.ft.agents.types import GaugeSample
from miles.utils.ft.controller.detectors.chain import build_detector_chain
from miles.utils.ft.controller.metrics.metric_names import GPU_AVAILABLE
from miles.utils.ft.controller.types import ControllerMode


class TestStatusConsistency:
    async def test_status_snapshots_internally_consistent(
        self,
        e2e_env: E2EEnv,
    ) -> None:
        """High-frequency polling during recovery → every snapshot is internally consistent."""
        env = e2e_env

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=FAST_TIMEOUT)
        await env.injector.crash_training()

        snapshots: list[Any] = []
        deadline = time.monotonic() + RECOVERY_TIMEOUT

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
                assert s.recovery is not None
            elif s.mode == ControllerMode.MONITORING:
                assert s.recovery is None


class TestRunIdSwitch:
    async def test_new_run_id_reregistration_and_metric_isolation(
        self,
        e2e_env: E2EEnv,
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
            timeout=RECOVERY_TIMEOUT,
        )

        # Step 2: verify new run_id
        post_run_id = final.active_run_id
        assert post_run_id is not None
        assert post_run_id != pre_run_id

        # Step 3: verify iteration progresses under new run
        await wait_for_training_stable(env.controller, n_iterations=3, timeout=FAST_TIMEOUT)


class TestMetricScrapeE2E:
    async def test_prometheus_exporter_to_metric_store_pipeline(
        self,
        e2e_full_detector_env: E2EEnv,
    ) -> None:
        """Worker pushes iteration → controller scrapes → status.latest_iteration tracks it."""
        env = e2e_full_detector_env

        await wait_for_training_stable(env.controller, n_iterations=5, timeout=FAST_TIMEOUT)
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
            detectors=[FastHangDetector(timeout_seconds=10.0)],
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
        self,
        e2e_env: E2EEnv,
    ) -> None:
        """log_step with old run_id after recovery does not pollute iteration counter."""
        env = e2e_env
        old_run_id = get_status(env.controller).active_run_id

        # Step 1: crash → recovery → new run
        await env.injector.crash_training()
        status = await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=RECOVERY_TIMEOUT,
        )
        new_run_id = status.active_run_id
        assert new_run_id != old_run_id

        # Step 2: inject a stale log_step with the old run_id
        ray.get(
            env.controller.log_step.remote(
                run_id=old_run_id,
                step=999999,
                metrics={"iteration": 999999.0},
            ),
            timeout=5,
        )

        # Step 3: verify iteration is not polluted
        await wait_for_training_stable(env.controller, n_iterations=3, timeout=FAST_TIMEOUT)
        final = get_status(env.controller)
        assert final.latest_iteration is not None
        assert final.latest_iteration < 999999


class TestRecoveryPhaseMonotonicity:
    async def test_recovery_phase_order_monotonic_during_polling(
        self,
        e2e_env: E2EEnv,
    ) -> None:
        """High-frequency polling during recovery: mode transitions are orderly."""
        env = e2e_env

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=FAST_TIMEOUT)
        await env.injector.crash_training()

        snapshots: list[Any] = []
        deadline = time.monotonic() + RECOVERY_TIMEOUT

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
                assert not saw_monitoring_after_recovery, "Mode went back to RECOVERY after returning to MONITORING"
            elif s.mode == ControllerMode.MONITORING and saw_recovery:
                saw_monitoring_after_recovery = True

        assert saw_recovery, "Never observed RECOVERY mode"
        assert saw_monitoring_after_recovery, "Never returned to MONITORING after recovery"


class TestBadNodesDuringEviction:
    async def test_status_reports_bad_nodes_during_eviction(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """During eviction, status.bad_nodes includes the faulted node; cleared after recovery."""
        env = make_e2e_env(
            ft_id="e2ebn",
            nodes=[
                NodeSpec(node_id="e2ebn-node-0", use_remote_collector=True),
                NodeSpec(node_id="e2ebn-node-1", use_remote_collector=True),
            ],
            detectors=build_detector_chain(),
            scrape_interval_seconds=_FAST_SCRAPE,
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=FAST_TIMEOUT)

        # Step 1: inject GPU fault on node-0
        env.set_collector_metrics(
            "e2ebn-node-0",
            [
                GaugeSample(
                    name=GPU_AVAILABLE,
                    labels={"node_id": "e2ebn-node-0", "gpu": "0"},
                    value=0.0,
                ),
            ],
        )

        # Step 2: poll status during recovery for bad_nodes
        saw_bad_nodes = False
        deadline = time.monotonic() + RECOVERY_TIMEOUT
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.recovery is not None and "e2ebn-node-0" in status.recovery.bad_nodes:
                saw_bad_nodes = True
            if status.mode == ControllerMode.MONITORING and not status.recovery_in_progress:
                if saw_bad_nodes:
                    break
            await asyncio.sleep(0.3)

        assert saw_bad_nodes, "Never observed bad_nodes containing faulted node during recovery"

        # Step 3: after recovery, recovery should be None (back to monitoring)
        final = get_status(env.controller)
        assert final.recovery is None


class TestStaleCombined:
    async def test_stale_register_and_stale_log_step_together_are_ignored(
        self,
        e2e_env: E2EEnv,
    ) -> None:
        """Concurrent stale register_training_rank + stale log_step do not pollute new run."""
        env = e2e_env
        old_run_id = get_status(env.controller).active_run_id

        # Step 1: crash → recovery → new run
        await env.injector.crash_training()
        status = await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=RECOVERY_TIMEOUT,
        )
        new_run_id = status.active_run_id
        assert new_run_id != old_run_id

        # Step 2: send stale register + stale log_step with old run_id
        ray.get(
            env.controller.register_training_rank.remote(
                run_id=old_run_id,
                rank=99,
                world_size=100,
                node_id="stale-node",
                exporter_address="http://stale:9090",
                pid=99999,
            ),
            timeout=5,
        )
        ray.get(
            env.controller.log_step.remote(
                run_id=old_run_id,
                step=999999,
                metrics={"iteration": 999999.0},
            ),
            timeout=5,
        )

        # Step 3: verify new run is not polluted
        await wait_for_training_stable(env.controller, n_iterations=3, timeout=FAST_TIMEOUT)
        final = get_status(env.controller)
        assert final.active_run_id == new_run_id
        assert final.latest_iteration is not None
        assert final.latest_iteration < 999999
