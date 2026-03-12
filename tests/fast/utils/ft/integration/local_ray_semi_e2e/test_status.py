"""Semi-E2E: status — consistency, run_id, metric scrape, false positive, monotonicity, bad nodes."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
import ray

from miles.utils.ft.agents.types import GaugeSample
from miles.utils.ft.controller.detectors.chain import build_detector_chain
from miles.utils.ft.controller.detectors.core.training_crash import TrainingCrashDetector
from miles.utils.ft.controller.types import ControllerMode
from miles.utils.ft.utils.metric_names import GPU_AVAILABLE
from tests.fast.utils.ft.integration.conftest import FAST_TIMEOUT, RECOVERY_TIMEOUT, RayNodeInfo
from tests.fast.utils.ft.integration.local_ray_semi_e2e.conftest import assert_no_recovery_triggered
from tests.fast.utils.ft.testbed.config import TestbedConfig, TestbedNodeConfig
from tests.fast.utils.ft.testbed.train import MilesTestbed
from tests.fast.utils.ft.utils.controller_fakes import FastHangDetector

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.anyio,
]


@pytest.fixture
async def make_testbed(
    local_ray_nodes: list[RayNodeInfo],
) -> AsyncIterator[Callable[..., MilesTestbed]]:
    """Factory fixture for creating MilesTestbed instances with custom configs."""
    created: list[MilesTestbed] = []

    async def _factory(**kwargs: Any) -> MilesTestbed:
        config = TestbedConfig(**kwargs)
        tb = await MilesTestbed.launch(config=config, ray_nodes=local_ray_nodes)
        created.append(tb)
        return tb

    yield _factory

    for tb in created:
        await tb.shutdown()


# ------------------------------------------------------------------
# 1. test_status_snapshots_consistent
# ------------------------------------------------------------------


async def test_status_snapshots_consistent(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """High-frequency polling during recovery: every snapshot is internally consistent."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector()],
    )

    # Step 1: verify training is stable, then crash
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)
    await testbed.crash_training()

    # Step 2: high-frequency poll and collect snapshots through recovery cycle
    snapshots: list[Any] = []
    deadline = time.monotonic() + RECOVERY_TIMEOUT

    while time.monotonic() < deadline:
        status = await testbed.get_status()
        snapshots.append(status)

        if status.mode == ControllerMode.MONITORING and not status.recovery_in_progress:
            if len(snapshots) > 5:
                break
        await asyncio.sleep(0.05)

    assert len(snapshots) > 5, "Not enough snapshots collected"

    # Step 3: verify consistency — RECOVERY implies recovery!=None, MONITORING implies recovery==None
    for s in snapshots:
        if s.mode == ControllerMode.RECOVERY:
            assert s.recovery_in_progress is True
            assert s.recovery is not None
        elif s.mode == ControllerMode.MONITORING:
            assert s.recovery is None


# ------------------------------------------------------------------
# 2. test_new_run_id_after_recovery
# ------------------------------------------------------------------


async def test_new_run_id_after_recovery(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Crash -> recovery -> new run_id differs from old run_id."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector()],
    )

    pre_status = await testbed.get_status()
    pre_run_id = pre_status.active_run_id

    # Step 1: crash -> recovery -> new run
    await testbed.crash_training()
    final = await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=RECOVERY_TIMEOUT,
    )

    # Step 2: verify new run_id
    post_run_id = final.active_run_id
    assert post_run_id is not None
    assert post_run_id != pre_run_id

    # Step 3: verify iteration progresses under new run
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)


# ------------------------------------------------------------------
# 3. test_metric_scrape_pipeline
# ------------------------------------------------------------------


async def test_metric_scrape_pipeline(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Worker pushes iteration -> controller scrapes -> status.latest_iteration tracks it."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=build_detector_chain(),
        scrape_interval_seconds=0.5,
    )

    # Step 1: wait for 5 iterations
    await testbed.wait_for_training_stable(n_iterations=5, timeout=FAST_TIMEOUT)

    # Step 2: verify latest_iteration >= 5
    status = await testbed.get_status()
    assert status.latest_iteration is not None
    assert status.latest_iteration >= 5


# test_healthy_training_no_false_positive removed: covered by test_scenarios::test_no_false_positive


# ------------------------------------------------------------------
# 5. test_metric_scrape_jitter_no_false_hang
# ------------------------------------------------------------------


async def test_metric_scrape_jitter_no_false_hang(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Low-frequency scrape (5s) with fast ticks (0.1s) does not false-trigger hang."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[FastHangDetector(timeout_seconds=10.0)],
        scrape_interval_seconds=5.0,
        tick_interval=0.1,
    )

    # Step 1: observe for a meaningful period — no false hang should be triggered
    status = await assert_no_recovery_triggered(
        testbed,
        observation_ticks=20,
        timeout=FAST_TIMEOUT,
    )
    assert status.mode == ControllerMode.MONITORING
    assert status.recovery_in_progress is False


# ------------------------------------------------------------------
# 6. test_stale_log_step_discarded
# ------------------------------------------------------------------


async def test_stale_log_step_discarded(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """log_step with old run_id after recovery does not pollute iteration counter."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector()],
    )

    old_status = await testbed.get_status()
    old_run_id = old_status.active_run_id

    # Step 1: crash -> recovery -> new run
    await testbed.crash_training()
    status = await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=RECOVERY_TIMEOUT,
    )
    new_run_id = status.active_run_id
    assert new_run_id != old_run_id

    # Step 2: inject a stale log_step with the old run_id
    ray.get(
        testbed.controller.log_step.remote(
            run_id=old_run_id,
            step=999999,
            metrics={"iteration": 999999.0},
        ),
        timeout=5,
    )

    # Step 3: verify iteration is not polluted
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)
    final = await testbed.get_status()
    assert final.latest_iteration is not None
    assert final.latest_iteration < 999999


# ------------------------------------------------------------------
# 7. test_recovery_phase_monotonicity
# ------------------------------------------------------------------


async def test_recovery_phase_monotonicity(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """High-frequency polling: mode transitions MONITORING -> RECOVERY -> MONITORING (never back)."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector()],
    )

    # Step 1: verify stable, then crash
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)
    await testbed.crash_training()

    # Step 2: high-frequency poll and collect snapshots
    snapshots: list[Any] = []
    deadline = time.monotonic() + RECOVERY_TIMEOUT

    while time.monotonic() < deadline:
        status = await testbed.get_status()
        snapshots.append(status)

        if status.mode == ControllerMode.MONITORING and not status.recovery_in_progress:
            if len(snapshots) > 5:
                break
        await asyncio.sleep(0.05)

    assert len(snapshots) > 5, "Not enough snapshots collected"

    # Step 3: verify monotonicity — once MONITORING is reached after RECOVERY, no going back
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


# ------------------------------------------------------------------
# 8. test_bad_nodes_reported_during_eviction
# ------------------------------------------------------------------


async def test_bad_nodes_reported_during_eviction(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """During eviction, recovery.bad_nodes includes the faulted node; cleared after recovery."""
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="n-0", num_ranks=2),
            TestbedNodeConfig(node_id="n-1", num_ranks=2),
        ],
        detectors=build_detector_chain(),
        scrape_interval_seconds=0.5,
    )

    # Step 1: verify stable
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 2: inject GPU fault on n-0
    await testbed.inject_collector_metrics(
        node_id="n-0",
        metrics=[
            GaugeSample(
                name=GPU_AVAILABLE,
                labels={"node_id": "n-0", "gpu": "0"},
                value=0.0,
            ),
        ],
    )

    # Step 3: poll status during recovery for bad_nodes
    saw_bad_nodes = False
    deadline = time.monotonic() + RECOVERY_TIMEOUT
    while time.monotonic() < deadline:
        status = await testbed.get_status()
        if status.recovery is not None and "n-0" in status.recovery.bad_nodes:
            saw_bad_nodes = True
        if status.mode == ControllerMode.MONITORING and not status.recovery_in_progress:
            if saw_bad_nodes:
                break
        await asyncio.sleep(0.3)

    assert saw_bad_nodes, "Never observed bad_nodes containing faulted node during recovery"

    # Step 4: after recovery, recovery should be None (back to monitoring)
    final = await testbed.get_status()
    assert final.recovery is None


# ------------------------------------------------------------------
# 9. test_stale_register_and_log_step_ignored
# ------------------------------------------------------------------


async def test_stale_register_and_log_step_ignored(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Concurrent stale register_training_rank + stale log_step do not pollute new run."""
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector()],
    )

    old_status = await testbed.get_status()
    old_run_id = old_status.active_run_id

    # Step 1: crash -> recovery -> new run
    await testbed.crash_training()
    status = await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=RECOVERY_TIMEOUT,
    )
    new_run_id = status.active_run_id
    assert new_run_id != old_run_id

    # Step 2: send stale register + stale log_step with old run_id
    ray.get(
        testbed.controller.register_training_rank.remote(
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
        testbed.controller.log_step.remote(
            run_id=old_run_id,
            step=999999,
            metrics={"iteration": 999999.0},
        ),
        timeout=5,
    )

    # Step 3: verify new run is not polluted
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)
    final = await testbed.get_status()
    assert final.active_run_id == new_run_id
    assert final.latest_iteration is not None
    assert final.latest_iteration < 999999
