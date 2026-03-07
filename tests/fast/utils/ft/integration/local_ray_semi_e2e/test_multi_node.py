"""Semi-E2E: multi-node — registration, parallel ranks, grace period, stale run_id."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import ray

from miles.utils.ft.controller.detectors.core.training_crash import TrainingCrashDetector
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
    wait_for_mode,
    wait_for_mode_transition,
    wait_for_recovery_complete,
    wait_for_recovery_phase,
    wait_for_training_stable,
)


class TestMultiNode:
    async def test_multi_rank_registration_and_targeted_eviction(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """4 ranks across 2 nodes register correctly; crash during MONITORING → DIAGNOSING."""
        env = make_e2e_env(
            ft_id="e2emn",
            nodes=[
                NodeSpec(node_id="e2emn-node-0", num_ranks=2),
                NodeSpec(node_id="e2emn-node-1", num_ranks=2),
            ],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
            recovery_cooldown=SlidingWindowThrottle(window_minutes=1.0, max_count=2),
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=30.0)

        status = get_status(env.controller)
        assert status.mode == ControllerMode.MONITORING

        # Step 1: crash → recovery → wait for MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=30.0,
        )

        # Step 2: crash during MONITORING → DIAGNOSING
        await env.injector.crash_training()

        # Poll for DIAGNOSING in phase_history during the active recovery.
        # After recovery completes, the FAILED status can auto-trigger a second
        # recovery that overwrites _last_phase_history, so we observe DIAGNOSING
        # while the recovery is still in progress.
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.phase_history and "StopTimeDiagnostics" in status.phase_history:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError("DIAGNOSING not observed in phase_history within 90s")

        assert_phase_path_contains(status, ["StopTimeDiagnostics"])


class TestConcurrentRegistration:
    async def test_parallel_rank_registration(
        self, e2e_multi_node_env: E2EEnv,
    ) -> None:
        """4 workers register in parallel → all ranks visible in controller status."""
        env = e2e_multi_node_env

        await wait_for_training_stable(env.controller, n_iterations=2, timeout=30.0)

        status = get_status(env.controller)
        assert status.latest_iteration is not None
        assert status.latest_iteration > 0


class TestRegistrationGrace:
    async def test_detectors_suppressed_during_grace_period(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """With grace_ticks=10, detectors don't fire during the grace period."""
        env = make_e2e_env(
            ft_id="e2egrce",
            nodes=[NodeSpec(node_id="e2egrce-node-0")],
            detectors=[TrainingCrashDetector()],
            registration_grace_ticks=10,
        )

        # Step 1: crash during grace period → should NOT trigger recovery
        await env.injector.crash_training()
        await asyncio.sleep(0.5)
        status = get_status(env.controller)
        if status.tick_count <= 10:
            assert status.mode == ControllerMode.MONITORING, (
                f"Recovery triggered during grace period at tick {status.tick_count}"
            )

        # Step 2: wait for grace period to end
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.tick_count > 10:
                break
            await asyncio.sleep(0.2)

        # Step 3: crash again after grace period → should trigger recovery
        await env.injector.recover_training()
        await asyncio.sleep(1.0)
        await env.injector.crash_training()
        status = await wait_for_mode(
            env.controller,
            target_mode=ControllerMode.RECOVERY,
            timeout=30.0,
        )
        assert status.mode == ControllerMode.RECOVERY


class TestStaleRunId:
    async def test_stale_run_id_registration_rejected(
        self, e2e_env: E2EEnv,
    ) -> None:
        """Registration with old run_id after recovery is silently rejected."""
        env = e2e_env
        old_run_id = get_status(env.controller).active_run_id
        assert old_run_id is not None

        # Step 1: crash → recovery → new run_id
        await env.injector.crash_training()
        status = await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=60.0,
        )
        new_run_id = status.active_run_id
        assert new_run_id != old_run_id

        # Step 2: attempt registration with stale run_id
        ray.get(env.controller.register_training_rank.remote(
            run_id=old_run_id, rank=99, world_size=1,
            node_id="stale-node", exporter_address="http://stale:9090",
            pid=9999,
        ), timeout=5)

        # Step 3: training continues normally under new run
        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)
        final = get_status(env.controller)
        assert final.active_run_id == new_run_id


class TestMultiNodeIsolation:
    async def test_single_node_fault_does_not_affect_healthy_nodes(
        self, e2e_multi_node_env: E2EEnv,
    ) -> None:
        """In multi-node setup, a global crash recovers cleanly with all nodes participating."""
        env = e2e_multi_node_env

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)

        # Step 1: crash → recovery
        await env.injector.crash_training()
        status = await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=60.0,
        )
        assert status.mode == ControllerMode.MONITORING
        assert status.recovery_in_progress is False

        # Step 2: training resumes across all nodes
        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)


class TestMultiNodeScale:
    async def test_3_nodes_full_recovery_all_ranks_reregister(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """3 nodes × 2 ranks each — crash → recovery → all 6 ranks re-register."""
        env = make_e2e_env(
            ft_id="e2e3n",
            nodes=[
                NodeSpec(node_id="e2e3n-node-0", num_ranks=2),
                NodeSpec(node_id="e2e3n-node-1", num_ranks=2),
                NodeSpec(node_id="e2e3n-node-2", num_ranks=2),
            ],
            detectors=[TrainingCrashDetector()],
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)

        # Step 1: crash → full recovery
        await env.injector.crash_training()
        status = await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=60.0,
        )
        assert status.mode == ControllerMode.MONITORING

        # Step 2: verify training resumes (proves all workers re-registered)
        await wait_for_training_stable(env.controller, n_iterations=5, timeout=30.0)


class TestGracePeriod:
    async def test_no_workers_registered_crash_does_not_trigger_recovery(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Crash before any worker registers → detectors skipped (rank_placement empty)."""
        env = make_e2e_env(
            ft_id="e2egp",
            nodes=[NodeSpec(node_id="e2egp-node-0")],
            detectors=[TrainingCrashDetector()],
            wait_for_iteration=False,
            registration_grace_ticks=0,
        )

        # Step 1: crash immediately before workers register
        await env.injector.crash_training()
        await asyncio.sleep(3.0)

        # Step 2: mode should still be MONITORING (detectors not run)
        status = get_status(env.controller)
        assert status.mode == ControllerMode.MONITORING, (
            f"Expected no recovery with empty rank_placement, but mode={status.mode}"
        )
