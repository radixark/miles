"""Semi-E2E: network faults — ephemeral NIC down, sustained NIC down, majority NIC down."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import timedelta

from miles.utils.ft.controller.detectors.chain import build_detector_chain
from miles.utils.ft.controller.detectors.network import NetworkAlertDetector, NetworkAlertDetectorConfig
from miles.utils.ft.models.metric_names import NODE_NETWORK_UP
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
    scenario_no_false_positive,
    wait_for_mode,
    wait_for_recovery_complete,
    wait_for_training_stable,
)


class TestEphemeralNic:
    async def test_ephemeral_nic_fault_goes_to_reattempting(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """NIC up→down transition → NetworkAlertDetector → ENTER_RECOVERY.

        ENTER_RECOVERY evicts directly without entering recovery mode,
        so we detect the action by observing run_id change.
        """
        env = make_e2e_env(
            ft_id="e2enic",
            nodes=[NodeSpec(
                node_id="e2enic-node-0",
                use_remote_collector=True,
            )],
            detectors=[
                NetworkAlertDetector(config=NetworkAlertDetectorConfig(
                    alert_window_minutes=10 / 60,
                    alert_threshold=1,
                )),
            ],
            scrape_interval_seconds=_FAST_SCRAPE,
        )

        await wait_for_training_stable(env.controller, n_iterations=2, timeout=30.0)

        # Step 1: inject NIC up so MiniPrometheus has a baseline
        env.set_collector_metrics("e2enic-node-0", [
            GaugeSample(
                name=NODE_NETWORK_UP,
                labels={"node_id": "e2enic-node-0", "device": "eth0"},
                value=1.0,
            ),
        ])
        await asyncio.sleep(_FAST_SCRAPE * 3)

        old_run_id = get_status(env.controller).active_run_id

        # Step 2: inject NIC down to create up→down transition
        env.set_collector_metrics("e2enic-node-0", [
            GaugeSample(
                name=NODE_NETWORK_UP,
                labels={"node_id": "e2enic-node-0", "device": "eth0"},
                value=0.0,
            ),
        ])

        # Step 3: ENTER_RECOVERY evicts and restarts without entering
        # recovery mode; poll until active_run_id changes.
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.active_run_id != old_run_id:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError("active_run_id did not change within 60s")

        assert status.mode == ControllerMode.MONITORING


class TestNetworkAlert:
    async def test_sustained_nic_down_triggers_eviction(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Sustained NIC up→down → NetworkAlertDetector → ENTER_RECOVERY.

        ENTER_RECOVERY evicts directly without entering recovery mode.
        """
        env = make_e2e_env(
            ft_id="e2enet",
            nodes=[NodeSpec(
                node_id="e2enet-node-0",
                use_remote_collector=True,
            )],
            detectors=[
                NetworkAlertDetector(config=NetworkAlertDetectorConfig(
                    alert_window_minutes=10 / 60,
                    alert_threshold=1,
                )),
            ],
            scrape_interval_seconds=_FAST_SCRAPE,
        )

        await wait_for_training_stable(env.controller, n_iterations=2, timeout=30.0)

        # Step 1: inject NIC up baseline
        env.set_collector_metrics("e2enet-node-0", [
            GaugeSample(
                name=NODE_NETWORK_UP,
                labels={"node_id": "e2enet-node-0", "device": "eth0"},
                value=1.0,
            ),
            GaugeSample(
                name=NODE_NETWORK_UP,
                labels={"node_id": "e2enet-node-0", "device": "eth1"},
                value=1.0,
            ),
        ])
        await asyncio.sleep(_FAST_SCRAPE * 3)

        old_run_id = get_status(env.controller).active_run_id

        # Step 2: inject sustained NIC down to create up→down transitions
        env.set_collector_metrics("e2enet-node-0", [
            GaugeSample(
                name=NODE_NETWORK_UP,
                labels={"node_id": "e2enet-node-0", "device": "eth0"},
                value=0.0,
            ),
            GaugeSample(
                name=NODE_NETWORK_UP,
                labels={"node_id": "e2enet-node-0", "device": "eth1"},
                value=0.0,
            ),
        ])

        # Step 3: poll until active_run_id changes (ENTER_RECOVERY
        # evicts and restarts without entering recovery mode)
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.active_run_id != old_run_id:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError("active_run_id did not change within 60s")

        assert status.mode == ControllerMode.MONITORING


class TestTransientFault:
    async def test_fault_clears_before_sustained_threshold_no_recovery(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Brief NIC down followed by quick recovery → no recovery triggered."""
        env = make_e2e_env(
            ft_id="e2etf",
            nodes=[NodeSpec(
                node_id="e2etf-node-0",
                use_remote_collector=True,
            )],
            detectors=[
                NetworkAlertDetector(config=NetworkAlertDetectorConfig(
                    alert_window_minutes=10 / 60,
                    alert_threshold=2,
                )),
            ],
            scrape_interval_seconds=_FAST_SCRAPE,
        )

        await wait_for_training_stable(env.controller, n_iterations=2, timeout=30.0)

        # Step 1: inject NIC up baseline
        env.set_collector_metrics("e2etf-node-0", [
            GaugeSample(
                name=NODE_NETWORK_UP,
                labels={"node_id": "e2etf-node-0", "device": "eth0"},
                value=1.0,
            ),
        ])
        await asyncio.sleep(_FAST_SCRAPE * 3)

        # Step 2: inject NIC down briefly
        env.set_collector_metrics("e2etf-node-0", [
            GaugeSample(
                name=NODE_NETWORK_UP,
                labels={"node_id": "e2etf-node-0", "device": "eth0"},
                value=0.0,
            ),
        ])
        await asyncio.sleep(_FAST_SCRAPE * 2)

        # Step 3: restore NIC up
        env.set_collector_metrics("e2etf-node-0", [
            GaugeSample(
                name=NODE_NETWORK_UP,
                labels={"node_id": "e2etf-node-0", "device": "eth0"},
                value=1.0,
            ),
        ])

        # Step 4: verify no recovery triggered
        await asyncio.sleep(3.0)
        status = get_status(env.controller)
        assert status.mode == ControllerMode.MONITORING


class TestEphemeralNicNoEviction:
    async def test_ephemeral_nic_recovery_does_not_evict_nodes(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Ephemeral NIC fault → recovery via EvictingAndRestarting (no Evicting phase)."""
        env = make_e2e_env(
            ft_id="e2enne",
            nodes=[NodeSpec(
                node_id="e2enne-node-0",
                use_remote_collector=True,
            )],
            detectors=[
                NetworkAlertDetector(config=NetworkAlertDetectorConfig(
                    alert_window_minutes=10 / 60,
                    alert_threshold=1,
                )),
            ],
            scrape_interval_seconds=_FAST_SCRAPE,
        )

        await wait_for_training_stable(env.controller, n_iterations=2, timeout=30.0)

        # Step 1: inject NIC up baseline
        env.set_collector_metrics("e2enne-node-0", [
            GaugeSample(
                name=NODE_NETWORK_UP,
                labels={"node_id": "e2enne-node-0", "device": "eth0"},
                value=1.0,
            ),
        ])
        await asyncio.sleep(_FAST_SCRAPE * 3)

        # Step 2: inject NIC down
        env.set_collector_metrics("e2enne-node-0", [
            GaugeSample(
                name=NODE_NETWORK_UP,
                labels={"node_id": "e2enne-node-0", "device": "eth0"},
                value=0.0,
            ),
        ])

        # Step 3: wait for recovery to complete
        final = await wait_for_recovery_complete(env.controller, timeout=90.0)
        assert final.mode == ControllerMode.MONITORING

        # Step 4: verify no eviction in phase_history
        if final.phase_history:
            assert "Evicting" not in final.phase_history, (
                f"Ephemeral NIC should not evict, but got: {final.phase_history}"
            )


class TestMajorityNicDown:
    async def test_majority_nic_down_triggers_non_ephemeral_eviction(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Majority NICs down (snapshot) → _check_majority_nic_down → non-ephemeral → Evicting."""
        env = make_e2e_env(
            ft_id="e2emaj",
            nodes=[NodeSpec(node_id="e2emaj-node-0", use_remote_collector=True)],
            detectors=build_detector_chain(),
            scrape_interval_seconds=_FAST_SCRAPE,
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)
        old_run_id = get_status(env.controller).active_run_id

        # Step 1: inject 3 NICs — 2 down, 1 up (majority down)
        env.set_collector_metrics("e2emaj-node-0", [
            GaugeSample(
                name=NODE_NETWORK_UP,
                labels={"node_id": "e2emaj-node-0", "device": "eth0"},
                value=0.0,
            ),
            GaugeSample(
                name=NODE_NETWORK_UP,
                labels={"node_id": "e2emaj-node-0", "device": "eth1"},
                value=0.0,
            ),
            GaugeSample(
                name=NODE_NETWORK_UP,
                labels={"node_id": "e2emaj-node-0", "device": "eth2"},
                value=1.0,
            ),
        ])

        # Step 2: wait for recovery to complete (eviction expected)
        final = await wait_for_recovery_complete(env.controller, timeout=90.0)
        assert final.mode == ControllerMode.MONITORING
        assert_phase_path_contains(final, ["Evicting"])
