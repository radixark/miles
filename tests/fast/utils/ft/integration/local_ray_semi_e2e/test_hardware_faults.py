"""Semi-E2E: hardware faults — GPU lost, NaN loss, XID, disk space, MFU, fault during recovery."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from miles.utils.ft.controller.detectors.chain import build_detector_chain
from miles.utils.ft.controller.detectors.mfu_decline import MfuDeclineDetector, MfuDeclineDetectorConfig
from miles.utils.ft.controller.detectors.training_crash import TrainingCrashDetector
from miles.utils.ft.models.metric_names import GPU_AVAILABLE, NODE_FILESYSTEM_AVAIL_BYTES, XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL
from miles.utils.ft.models.metrics import GaugeSample
from miles.utils.ft.models.recovery import ControllerMode

from tests.fast.utils.ft.integration.local_ray_semi_e2e.conftest import (
    E2EEnv,
    NodeSpec,
    _FAST_SCRAPE,
)
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios import (
    assert_phase_path_contains,
    get_status,
    wait_for_mode,
    wait_for_mode_transition,
    wait_for_recovery_complete,
    wait_for_training_stable,
)


class TestHardwareAlert:
    async def test_gpu_lost_triggers_direct_eviction(
        self, e2e_full_detector_env: E2EEnv,
    ) -> None:
        """GPU_AVAILABLE=0 → HighConfidenceHardwareDetector → ENTER_RECOVERY."""
        env = e2e_full_detector_env

        old_run_id = get_status(env.controller).active_run_id

        # Step 1: inject GPU unavailable metric
        env.set_collector_metrics("e2efd-node-0", [
            GaugeSample(
                name=GPU_AVAILABLE,
                labels={"node_id": "e2efd-node-0", "gpu": "0"},
                value=0.0,
            ),
        ])

        # Step 2: ENTER_RECOVERY triggers recovery state machine. Poll until
        # active_run_id changes, indicating eviction and restart happened.
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.active_run_id != old_run_id:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError("active_run_id did not change within 60s")

        assert status.mode == ControllerMode.MONITORING


class TestNanLoss:
    async def test_nan_loss_triggers_recovery(
        self, e2e_full_detector_env: E2EEnv,
    ) -> None:
        """loss=NaN via custom log metrics → NanLossDetector → ENTER_RECOVERY."""
        env = e2e_full_detector_env

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)

        # Step 1: inject NaN loss
        await env.injector.inject_nan_loss()

        # Step 2: wait for recovery
        status = await wait_for_mode(
            env.controller,
            target_mode=ControllerMode.RECOVERY,
            timeout=30.0,
        )
        assert status.mode == ControllerMode.RECOVERY


class TestFaultDuringRecovery:
    async def test_hardware_fault_during_reattempting_escalates(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Crash → REATTEMPTING → inject GPU_AVAILABLE=0 → critical detector finds → EVICT_AND_RESTART."""
        env = make_e2e_env(
            ft_id="e2efdr",
            nodes=[NodeSpec(
                node_id="e2efdr-node-0",
                use_remote_collector=True,
            )],
            detectors=build_detector_chain(),
            scrape_interval_seconds=_FAST_SCRAPE,
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)

        # Step 1: first crash → enters recovery
        await env.injector.crash_training()
        await wait_for_mode(
            env.controller,
            target_mode=ControllerMode.RECOVERY,
            timeout=30.0,
        )

        # Step 2: while in recovery, inject hardware fault
        env.set_collector_metrics("e2efdr-node-0", [
            GaugeSample(
                name=GPU_AVAILABLE,
                labels={"node_id": "e2efdr-node-0", "gpu": "0"},
                value=0.0,
            ),
        ])

        # Step 3: wait for recovery to complete with eviction
        final = await wait_for_recovery_complete(env.controller, timeout=90.0)
        assert_phase_path_contains(final, ["Evicting"])


class TestDynamicBadNodes:
    async def test_too_many_dynamic_bad_nodes_during_recovery_aborts(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """During recovery, if critical detectors find >= max_simultaneous_bad_nodes, recovery aborts."""
        env = make_e2e_env(
            ft_id="e2edbn",
            nodes=[
                NodeSpec(node_id=f"e2edbn-node-{i}", use_remote_collector=True)
                for i in range(4)
            ],
            detectors=build_detector_chain(),
            scrape_interval_seconds=_FAST_SCRAPE,
            max_simultaneous_bad_nodes=3,
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)

        # Step 1: crash to enter recovery
        await env.injector.crash_training()
        await wait_for_mode(
            env.controller,
            target_mode=ControllerMode.RECOVERY,
            timeout=30.0,
        )

        # Step 2: during recovery, inject GPU_AVAILABLE=0 on 3 nodes
        for i in range(3):
            node_id = f"e2edbn-node-{i}"
            env.set_collector_metrics(node_id, [
                GaugeSample(
                    name=GPU_AVAILABLE,
                    labels={"node_id": node_id, "gpu": "0"},
                    value=0.0,
                ),
            ])

        # Step 3: recovery should abort, returning to MONITORING
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.mode == ControllerMode.MONITORING and not status.recovery_in_progress:
                return
            await asyncio.sleep(0.5)

        status = get_status(env.controller)
        assert status.mode == ControllerMode.MONITORING, (
            f"Recovery did not abort: mode={status.mode}, phase={status.recovery_phase}"
        )


class TestBadNodeMerging:
    async def test_critical_fault_during_recovery_merges_bad_nodes(
        self, make_e2e_env: Callable[..., E2EEnv],
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

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)

        # Step 1: inject GPU fault on node-0 → recovery with bad_node_ids=[node-0]
        env.set_collector_metrics("e2ebnm-node-0", [
            GaugeSample(
                name=GPU_AVAILABLE,
                labels={"node_id": "e2ebnm-node-0", "gpu": "0"},
                value=0.0,
            ),
        ])

        await wait_for_mode(
            env.controller,
            target_mode=ControllerMode.RECOVERY,
            timeout=30.0,
        )

        # Step 2: during recovery, also inject GPU fault on node-1
        env.set_collector_metrics("e2ebnm-node-1", [
            GaugeSample(
                name=GPU_AVAILABLE,
                labels={"node_id": "e2ebnm-node-1", "gpu": "0"},
                value=0.0,
            ),
        ])

        # Step 3: wait for recovery to complete with eviction of both nodes
        final = await wait_for_recovery_complete(env.controller, timeout=120.0)
        assert_phase_path_contains(final, ["Evicting"])


class TestCrossFaultTypeThrottle:
    async def test_different_fault_types_share_throttle_window(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Crash recovery + NaN recovery share the same cooldown window."""
        from miles.utils.ft.controller.recovery.helpers import SlidingWindowThrottle

        env = make_e2e_env(
            ft_id="e2ecft",
            nodes=[NodeSpec(node_id="e2ecft-node-0", use_remote_collector=True)],
            detectors=build_detector_chain(),
            scrape_interval_seconds=_FAST_SCRAPE,
            recovery_cooldown=SlidingWindowThrottle(window_minutes=60, max_count=2),
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)

        # Step 1: crash → first recovery
        await env.injector.crash_training()
        await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=60.0,
        )

        # Step 2: NaN → should be throttled (shared window, max_count=2)
        await wait_for_training_stable(env.controller, n_iterations=2, timeout=30.0)
        await env.injector.inject_nan_loss()
        await asyncio.sleep(5.0)
        status = get_status(env.controller)
        assert status.mode == ControllerMode.MONITORING, (
            f"Expected NaN recovery to be throttled, but mode={status.mode}"
        )


class TestNaNRecovery:
    async def test_nan_loss_cleared_after_recovery(
        self, e2e_full_detector_env: E2EEnv,
    ) -> None:
        """NaN triggers recovery, after which training resumes with clean metrics."""
        env = e2e_full_detector_env

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)

        # Step 1: inject NaN → recovery
        await env.injector.inject_nan_loss()
        await wait_for_mode(
            env.controller,
            target_mode=ControllerMode.RECOVERY,
            timeout=30.0,
        )

        # Step 2: wait for recovery to complete
        final = await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=90.0,
        )
        assert final.mode == ControllerMode.MONITORING

        # Step 3: training resumes normally (NaN state cleared by submit())
        await wait_for_training_stable(env.controller, n_iterations=5, timeout=30.0)


class TestDiskSpaceLow:
    async def test_disk_space_low_triggers_notify_not_recovery(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """NODE_FILESYSTEM_AVAIL_BYTES < 1GB → DiskSpaceLowDetector → NOTIFY_HUMAN, no recovery."""
        env = make_e2e_env(
            ft_id="e2edsk",
            nodes=[NodeSpec(node_id="e2edsk-node-0", use_remote_collector=True)],
            detectors=build_detector_chain(),
            scrape_interval_seconds=_FAST_SCRAPE,
            use_notifier=True,
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)

        # Step 1: inject low disk space (500 MB < 1 GB threshold)
        env.set_collector_metrics("e2edsk-node-0", [
            GaugeSample(
                name=NODE_FILESYSTEM_AVAIL_BYTES,
                labels={"node_id": "e2edsk-node-0", "mountpoint": "/data"},
                value=500_000_000.0,
            ),
        ])

        # Step 2: wait for detector cycles, verify no recovery
        await asyncio.sleep(5.0)
        status = get_status(env.controller)
        assert status.mode == ControllerMode.MONITORING, (
            f"DiskSpaceLow should not trigger recovery, but mode={status.mode}"
        )

        # Step 3: notifier should have received a disk-related notification
        calls = env.get_notifier_calls()
        assert len(calls) > 0, "Notifier should have received disk space notification"


class TestXidFault:
    async def test_xid_non_auto_recoverable_triggers_eviction(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL > 0 → HighConfidenceHardwareDetector → eviction."""
        env = make_e2e_env(
            ft_id="e2exid",
            nodes=[NodeSpec(node_id="e2exid-node-0", use_remote_collector=True)],
            detectors=build_detector_chain(),
            scrape_interval_seconds=_FAST_SCRAPE,
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)
        old_run_id = get_status(env.controller).active_run_id

        # Step 1: inject non-auto-recoverable XID event
        env.set_collector_metrics("e2exid-node-0", [
            GaugeSample(
                name=XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL,
                labels={"node_id": "e2exid-node-0"},
                value=1.0,
            ),
        ])

        # Step 2: wait for eviction + recovery (run_id changes)
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.active_run_id != old_run_id:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError("XID fault did not trigger eviction within 60s")

        assert status.mode == ControllerMode.MONITORING


class TestMfuAbsoluteMinimum:
    async def test_mfu_absolute_minimum_breach_notifies_human(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """MFU below absolute minimum → MfuDeclineDetector → NOTIFY_HUMAN, no recovery."""
        env = make_e2e_env(
            ft_id="e2emfu",
            nodes=[NodeSpec(node_id="e2emfu-node-0")],
            detectors=[
                TrainingCrashDetector(),
                MfuDeclineDetector(config=MfuDeclineDetectorConfig(
                    mfu_absolute_minimum=0.5,
                    consecutive_steps=2,
                    baseline_steps=2,
                )),
            ],
            use_notifier=True,
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)

        # Step 1: inject very low MFU
        await env.injector.inject_custom_metrics({"mfu": 0.01})

        # Step 2: wait for detector to evaluate MFU (needs consecutive_steps=2 readings)
        await asyncio.sleep(5.0)

        # Step 3: verify no recovery triggered, but notifier received alert
        status = get_status(env.controller)
        assert status.mode == ControllerMode.MONITORING, (
            f"MFU NOTIFY_HUMAN should not trigger recovery, but mode={status.mode}"
        )

        calls = env.get_notifier_calls()
        assert len(calls) > 0, "Notifier should have received MFU alert"
