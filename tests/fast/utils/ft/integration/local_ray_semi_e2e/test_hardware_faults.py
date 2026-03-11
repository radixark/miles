"""Semi-E2E: hardware faults — GPU lost, NaN loss, XID, disk space, MFU, fault during recovery."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from tests.fast.utils.ft.integration.conftest import FAST_TIMEOUT, LONG_RECOVERY_TIMEOUT, RECOVERY_TIMEOUT
from tests.fast.utils.ft.integration.local_ray_semi_e2e.conftest import _FAST_SCRAPE, E2EEnv, NodeSpec
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios import (
    get_status,
    scenario_no_false_positive,
    wait_for_mode,
    wait_for_mode_transition,
    wait_for_recovery_complete,
    wait_for_training_stable,
)

from miles.utils.ft.agents.types import GaugeSample
from miles.utils.ft.controller.detectors.chain import build_detector_chain
from miles.utils.ft.controller.detectors.core.gpu_fault import GpuFaultDetector
from miles.utils.ft.controller.detectors.core.mfu_decline import MfuDeclineDetector, MfuDeclineDetectorConfig
from miles.utils.ft.controller.detectors.core.training_crash import TrainingCrashDetector
from miles.utils.ft.controller.metrics.metric_names import (
    GPU_AVAILABLE,
    NODE_FILESYSTEM_AVAIL_BYTES,
    XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL,
)
from miles.utils.ft.controller.types import ControllerMode


class TestHardwareAlert:
    async def test_gpu_lost_triggers_direct_eviction(
        self,
        e2e_full_detector_env: E2EEnv,
    ) -> None:
        """GPU_AVAILABLE=0 → GpuFaultDetector → ENTER_RECOVERY."""
        env = e2e_full_detector_env

        old_run_id = get_status(env.controller).active_run_id

        # Step 1: inject GPU unavailable metric
        env.set_collector_metrics(
            "e2efd-node-0",
            [
                GaugeSample(
                    name=GPU_AVAILABLE,
                    labels={"node_id": "e2efd-node-0", "gpu": "0"},
                    value=0.0,
                ),
            ],
        )

        # Step 2: ENTER_RECOVERY triggers recovery state machine. Poll until
        # active_run_id changes AND controller returns to MONITORING.
        deadline = time.monotonic() + LONG_RECOVERY_TIMEOUT
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.active_run_id != old_run_id and status.mode == ControllerMode.MONITORING:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError(
                f"Recovery did not complete within {LONG_RECOVERY_TIMEOUT}s: "
                f"run_id changed={status.active_run_id != old_run_id}, mode={status.mode}"
            )

        assert status.mode == ControllerMode.MONITORING


class TestNanLoss:
    async def test_nan_loss_triggers_recovery(
        self,
        e2e_full_detector_env: E2EEnv,
    ) -> None:
        """loss=NaN via custom log metrics → NanLossDetector → ENTER_RECOVERY."""
        env = e2e_full_detector_env

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=FAST_TIMEOUT)

        # Step 1: inject NaN loss
        await env.injector.inject_nan_loss()

        # Step 2: wait for recovery
        status = await wait_for_mode(
            env.controller,
            target_mode=ControllerMode.RECOVERY,
            timeout=FAST_TIMEOUT,
        )
        assert status.mode == ControllerMode.RECOVERY


class TestFaultDuringRecovery:
    async def test_hardware_fault_during_reattempting_escalates(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Crash → REATTEMPTING → inject GPU_AVAILABLE=0 → critical detector finds → EVICT_AND_RESTART."""
        env = make_e2e_env(
            ft_id="e2efdr",
            nodes=[
                NodeSpec(
                    node_id="e2efdr-node-0",
                    use_remote_collector=True,
                )
            ],
            detectors=build_detector_chain(),
            scrape_interval_seconds=_FAST_SCRAPE,
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=FAST_TIMEOUT)

        # Step 1: first crash → enters recovery
        await env.injector.crash_training()
        await wait_for_mode(
            env.controller,
            target_mode=ControllerMode.RECOVERY,
            timeout=FAST_TIMEOUT,
        )

        # Step 2: while in recovery, inject hardware fault
        env.set_collector_metrics(
            "e2efdr-node-0",
            [
                GaugeSample(
                    name=GPU_AVAILABLE,
                    labels={"node_id": "e2efdr-node-0", "gpu": "0"},
                    value=0.0,
                ),
            ],
        )

        # Step 3: wait for recovery to complete with eviction
        await wait_for_recovery_complete(env.controller, timeout=LONG_RECOVERY_TIMEOUT)


class TestDynamicBadNodes:
    async def test_too_many_dynamic_bad_nodes_during_recovery_aborts(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """During recovery, if critical detectors find >= max_simultaneous_bad_nodes, recovery aborts."""
        env = make_e2e_env(
            ft_id="e2edbn",
            nodes=[NodeSpec(node_id=f"e2edbn-node-{i}", use_remote_collector=True) for i in range(4)],
            detectors=build_detector_chain(),
            scrape_interval_seconds=_FAST_SCRAPE,
            max_simultaneous_bad_nodes=3,
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=FAST_TIMEOUT)

        # Step 1: crash to enter recovery
        await env.injector.crash_training()
        await wait_for_mode(
            env.controller,
            target_mode=ControllerMode.RECOVERY,
            timeout=FAST_TIMEOUT,
        )

        # Step 2: during recovery, inject GPU_AVAILABLE=0 on 3 nodes
        for i in range(3):
            node_id = f"e2edbn-node-{i}"
            env.set_collector_metrics(
                node_id,
                [
                    GaugeSample(
                        name=GPU_AVAILABLE,
                        labels={"node_id": node_id, "gpu": "0"},
                        value=0.0,
                    ),
                ],
            )

        # Step 3: recovery should abort, returning to MONITORING
        deadline = time.monotonic() + RECOVERY_TIMEOUT
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.mode == ControllerMode.MONITORING and not status.recovery_in_progress:
                return
            await asyncio.sleep(0.5)

        status = get_status(env.controller)
        recovery_phase = status.recovery.phase if status.recovery else None
        assert (
            status.mode == ControllerMode.MONITORING
        ), f"Recovery did not abort: mode={status.mode}, phase={recovery_phase}"


class TestBadNodeMerging:
    async def test_critical_fault_during_recovery_merges_bad_nodes(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """New critical fault during recovery merges bad nodes: node-0 + node-1 evicted."""
        env = make_e2e_env(
            ft_id="e2ebnm",
            nodes=[
                NodeSpec(node_id="e2ebnm-node-0", use_remote_collector=True),
                NodeSpec(node_id="e2ebnm-node-1", use_remote_collector=True),
                NodeSpec(node_id="e2ebnm-node-2", use_remote_collector=True),
            ],
            detectors=build_detector_chain(),
            scrape_interval_seconds=_FAST_SCRAPE,
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=FAST_TIMEOUT)

        # Step 1: inject GPU fault on node-0 → recovery with bad_node_ids=[node-0]
        env.set_collector_metrics(
            "e2ebnm-node-0",
            [
                GaugeSample(
                    name=GPU_AVAILABLE,
                    labels={"node_id": "e2ebnm-node-0", "gpu": "0"},
                    value=0.0,
                ),
            ],
        )

        await wait_for_mode(
            env.controller,
            target_mode=ControllerMode.RECOVERY,
            timeout=FAST_TIMEOUT,
        )

        # Step 2: during recovery, also inject GPU fault on node-1
        env.set_collector_metrics(
            "e2ebnm-node-1",
            [
                GaugeSample(
                    name=GPU_AVAILABLE,
                    labels={"node_id": "e2ebnm-node-1", "gpu": "0"},
                    value=0.0,
                ),
            ],
        )

        # Step 3: wait for recovery to complete with eviction of both nodes
        await wait_for_recovery_complete(env.controller, timeout=LONG_RECOVERY_TIMEOUT)


class TestCrossFaultTypeThrottle:
    async def test_different_fault_types_share_throttle_window(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Crash recovery + NaN recovery share the same cooldown window."""
        from miles.utils.ft.utils.sliding_window import SlidingWindowThrottle

        env = make_e2e_env(
            ft_id="e2ecft",
            nodes=[NodeSpec(node_id="e2ecft-node-0", use_remote_collector=True)],
            detectors=build_detector_chain(),
            scrape_interval_seconds=_FAST_SCRAPE,
            recovery_cooldown=SlidingWindowThrottle(window_minutes=60, max_count=2),
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=RECOVERY_TIMEOUT)

        # Step 1: crash → first recovery
        await env.injector.crash_training()
        await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=LONG_RECOVERY_TIMEOUT,
        )

        # Step 2: NaN → should be throttled (shared window, max_count=2)
        await wait_for_training_stable(env.controller, n_iterations=2, timeout=RECOVERY_TIMEOUT)
        await env.injector.inject_nan_loss()
        await scenario_no_false_positive(
            env.controller,
            observation_ticks=20,
            timeout=FAST_TIMEOUT,
        )


class TestNaNRecovery:
    async def test_nan_loss_cleared_after_recovery(
        self,
        e2e_full_detector_env: E2EEnv,
    ) -> None:
        """NaN triggers recovery, after which training resumes with clean metrics."""
        env = e2e_full_detector_env

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=FAST_TIMEOUT)

        # Step 1: inject NaN → recovery
        await env.injector.inject_nan_loss()
        await wait_for_mode(
            env.controller,
            target_mode=ControllerMode.RECOVERY,
            timeout=FAST_TIMEOUT,
        )

        # Step 2: wait for recovery to complete
        final = await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=LONG_RECOVERY_TIMEOUT,
        )
        assert final.mode == ControllerMode.MONITORING

        # Step 3: training resumes normally (NaN state cleared by submit())
        await wait_for_training_stable(env.controller, n_iterations=5, timeout=FAST_TIMEOUT)


class TestDiskSpaceLow:
    async def test_disk_space_low_triggers_notify_not_recovery(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """NODE_FILESYSTEM_AVAIL_BYTES < 1GB → DiskSpaceLowDetector → NOTIFY_HUMAN, no recovery."""
        env = make_e2e_env(
            ft_id="e2edsk",
            nodes=[NodeSpec(node_id="e2edsk-node-0", use_remote_collector=True)],
            detectors=build_detector_chain(),
            scrape_interval_seconds=_FAST_SCRAPE,
            use_notifier=True,
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=FAST_TIMEOUT)

        # Step 1: inject low disk space (500 MB < 1 GB threshold)
        env.set_collector_metrics(
            "e2edsk-node-0",
            [
                GaugeSample(
                    name=NODE_FILESYSTEM_AVAIL_BYTES,
                    labels={"node_id": "e2edsk-node-0", "mountpoint": "/data"},
                    value=500_000_000.0,
                ),
            ],
        )

        # Step 2: wait for detector cycles, verify no recovery
        await scenario_no_false_positive(
            env.controller,
            observation_ticks=20,
            timeout=FAST_TIMEOUT,
        )

        # Step 3: notifier should have received a disk-related notification
        calls = env.get_notifier_calls()
        assert len(calls) > 0, "Notifier should have received disk space notification"


class TestXidFault:
    async def test_xid_non_auto_recoverable_triggers_eviction(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL > 0 → GpuFaultDetector → eviction."""
        env = make_e2e_env(
            ft_id="e2exid",
            nodes=[NodeSpec(node_id="e2exid-node-0", use_remote_collector=True)],
            detectors=build_detector_chain(),
            scrape_interval_seconds=_FAST_SCRAPE,
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=FAST_TIMEOUT)
        old_run_id = get_status(env.controller).active_run_id

        # Step 1: inject non-auto-recoverable XID event
        env.set_collector_metrics(
            "e2exid-node-0",
            [
                GaugeSample(
                    name=XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL,
                    labels={"node_id": "e2exid-node-0"},
                    value=1.0,
                ),
            ],
        )

        # Step 2: wait for eviction + recovery (run_id changes) AND return to MONITORING
        deadline = time.monotonic() + LONG_RECOVERY_TIMEOUT
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.active_run_id != old_run_id and status.mode == ControllerMode.MONITORING:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError(
                f"XID fault did not complete recovery within {LONG_RECOVERY_TIMEOUT}s: "
                f"run_id changed={status.active_run_id != old_run_id}, mode={status.mode}"
            )

        assert status.mode == ControllerMode.MONITORING


class TestMfuAbsoluteMinimum:
    async def test_mfu_absolute_minimum_breach_notifies_human(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """MFU below absolute minimum → MfuDeclineDetector → NOTIFY_HUMAN, no recovery."""
        env = make_e2e_env(
            ft_id="e2emfu",
            nodes=[NodeSpec(node_id="e2emfu-node-0")],
            detectors=[
                TrainingCrashDetector(),
                MfuDeclineDetector(
                    config=MfuDeclineDetectorConfig(
                        mfu_absolute_minimum=0.5,
                        consecutive_steps=2,
                        baseline_steps=2,
                    )
                ),
            ],
            use_notifier=True,
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=FAST_TIMEOUT)

        # Step 1: inject very low MFU
        await env.injector.inject_custom_metrics({"mfu": 0.01})

        # Step 2: wait for detector cycles, verify no recovery
        await scenario_no_false_positive(
            env.controller,
            observation_ticks=20,
            timeout=FAST_TIMEOUT,
        )

        # Step 3: notifier should have received MFU alert
        calls = env.get_notifier_calls()
        assert len(calls) > 0, "Notifier should have received MFU alert"


class TestNonCriticalSuppressed:
    async def test_non_critical_detectors_suppressed_during_recovery(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """During recovery, DiskSpaceLow (non-critical) does not interrupt the recovery flow."""
        env = make_e2e_env(
            ft_id="e2encs",
            nodes=[NodeSpec(node_id="e2encs-node-0", use_remote_collector=True)],
            detectors=build_detector_chain(),
            scrape_interval_seconds=_FAST_SCRAPE,
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=FAST_TIMEOUT)

        # Step 1: crash → enters recovery
        await env.injector.crash_training()
        await wait_for_mode(
            env.controller,
            target_mode=ControllerMode.RECOVERY,
            timeout=FAST_TIMEOUT,
        )

        # Step 2: while in recovery, inject disk space fault
        env.set_collector_metrics(
            "e2encs-node-0",
            [
                GaugeSample(
                    name=NODE_FILESYSTEM_AVAIL_BYTES,
                    labels={"node_id": "e2encs-node-0", "mountpoint": "/data"},
                    value=100_000_000.0,
                ),
            ],
        )

        # Step 3: recovery should complete normally (disk alert ignored)
        final = await wait_for_recovery_complete(env.controller, timeout=LONG_RECOVERY_TIMEOUT)
        assert final.mode == ControllerMode.MONITORING


class TestRealtimeChecksDiscovery:
    async def test_crash_plus_hardware_alert_found_in_realtime_checks(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Crash + GPU fault metric → GpuFaultDetector discovers bad node during
        recovery's collect_evictable_bad_nodes → RealtimeChecks has pre_identified_bad_nodes → Evicting."""
        env = make_e2e_env(
            ft_id="e2ertc",
            nodes=[NodeSpec(node_id="e2ertc-node-0", use_remote_collector=True)],
            detectors=[TrainingCrashDetector(), GpuFaultDetector()],
            scrape_interval_seconds=_FAST_SCRAPE,
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=FAST_TIMEOUT)

        # Step 1: inject GPU fault metric (but only TrainingCrashDetector is in chain,
        # so the detector won't pre-identify the bad node)
        env.set_collector_metrics(
            "e2ertc-node-0",
            [
                GaugeSample(
                    name=GPU_AVAILABLE,
                    labels={"node_id": "e2ertc-node-0", "gpu": "0"},
                    value=0.0,
                ),
            ],
        )
        await asyncio.sleep(_FAST_SCRAPE * 3)

        # Step 2: crash → TrainingCrashDetector fires with empty bad_node_ids
        # → RealtimeChecks → check_alerts() finds GPU fault → Evicting
        old_run_id = get_status(env.controller).active_run_id
        await env.injector.crash_training()

        deadline = time.monotonic() + LONG_RECOVERY_TIMEOUT
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.active_run_id != old_run_id and status.mode == ControllerMode.MONITORING:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError(
                f"Recovery did not complete within {LONG_RECOVERY_TIMEOUT}s: "
                f"run_id changed={status.active_run_id != old_run_id}, mode={status.mode}"
            )

        assert status.mode == ControllerMode.MONITORING


class TestMaxBadNodesOneBoundary:
    async def test_max_simultaneous_bad_nodes_one_single_fault_is_false_positive(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """max_simultaneous_bad_nodes=1 → even a single GPU fault is treated as false positive."""
        env = make_e2e_env(
            ft_id="e2emb1",
            nodes=[NodeSpec(node_id="e2emb1-node-0", use_remote_collector=True)],
            detectors=build_detector_chain(),
            scrape_interval_seconds=_FAST_SCRAPE,
            max_simultaneous_bad_nodes=1,
            use_notifier=True,
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=FAST_TIMEOUT)

        # Step 1: inject GPU fault on the single node
        env.set_collector_metrics(
            "e2emb1-node-0",
            [
                GaugeSample(
                    name=GPU_AVAILABLE,
                    labels={"node_id": "e2emb1-node-0", "gpu": "0"},
                    value=0.0,
                ),
            ],
        )

        # Step 2: wait for detection cycles, verify no recovery
        await scenario_no_false_positive(
            env.controller,
            observation_ticks=20,
            timeout=FAST_TIMEOUT,
        )


class TestFaultClearedNoRetrigger:
    async def test_hardware_fault_cleared_after_eviction_no_retriggering(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """GPU fault → eviction → clear fault metric → no re-trigger (no infinite loop)."""
        env = make_e2e_env(
            ft_id="e2efcl",
            nodes=[NodeSpec(node_id="e2efcl-node-0", use_remote_collector=True)],
            detectors=build_detector_chain(),
            scrape_interval_seconds=_FAST_SCRAPE,
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=FAST_TIMEOUT)

        # Step 1: inject GPU fault → triggers eviction
        env.set_collector_metrics(
            "e2efcl-node-0",
            [
                GaugeSample(
                    name=GPU_AVAILABLE,
                    labels={"node_id": "e2efcl-node-0", "gpu": "0"},
                    value=0.0,
                ),
            ],
        )

        await wait_for_mode(
            env.controller,
            target_mode=ControllerMode.RECOVERY,
            timeout=FAST_TIMEOUT,
        )

        # Step 2: clear fault metric (simulating node replacement)
        env.set_collector_metrics(
            "e2efcl-node-0",
            [
                GaugeSample(
                    name=GPU_AVAILABLE,
                    labels={"node_id": "e2efcl-node-0", "gpu": "0"},
                    value=1.0,
                ),
            ],
        )

        # Step 3: recovery completes
        final = await wait_for_recovery_complete(env.controller, timeout=LONG_RECOVERY_TIMEOUT)
        assert final.mode == ControllerMode.MONITORING

        # Step 4: verify no new recovery triggered after several ticks
        await scenario_no_false_positive(
            env.controller,
            observation_ticks=20,
            timeout=FAST_TIMEOUT,
        )
