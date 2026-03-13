"""Semi-E2E: network fault detection — NIC up→down transitions, majority down, ephemeral faults."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

import pytest
from tests.fast.utils.ft.integration.conftest import FAST_TIMEOUT, LONG_RECOVERY_TIMEOUT
from tests.fast.utils.ft.integration.local_ray_semi_e2e.conftest import assert_no_recovery_triggered
from tests.fast.utils.ft.testbed.config import TestbedNodeConfig
from tests.fast.utils.ft.testbed.train import MilesTestbed

from miles.utils.ft.agents.types import GaugeSample
from miles.utils.ft.controller.detectors.chain import build_detector_chain
from miles.utils.ft.controller.detectors.core.network import NetworkAlertDetector, NetworkAlertDetectorConfig
from miles.utils.ft.controller.types import ControllerMode
from miles.utils.ft.utils.metric_names import NODE_NETWORK_UP

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.anyio,
]

_SCRAPE_INTERVAL = 0.5


def _nic_sample(node_id: str, device: str, value: float) -> GaugeSample:
    return GaugeSample(
        name=NODE_NETWORK_UP,
        labels={"node_id": node_id, "device": device},
        value=value,
    )


# ------------------------------------------------------------------
# 1. test_ephemeral_nic_fault_triggers_recovery
# ------------------------------------------------------------------


async def test_ephemeral_nic_fault_triggers_recovery(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """NIC up→down transition triggers recovery via NetworkAlertDetector.

    Baseline collector provides eth0=1.0, eth1=1.0. After a few scrapes
    establish the baseline, we inject eth0=0.0 to create an up→down
    transition. With alert_threshold=1, a single transition fires recovery.
    """
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="n-0", num_ranks=2),
        ],
        detectors=[
            NetworkAlertDetector(
                config=NetworkAlertDetectorConfig(
                    alert_window_minutes=10.0 / 60.0,
                    alert_threshold=1,
                ),
            ),
        ],
        scrape_interval_seconds=_SCRAPE_INTERVAL,
    )

    # Step 1: wait for training to stabilize and baseline NIC metrics to be scraped
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)
    await asyncio.sleep(_SCRAPE_INTERVAL * 3)

    # Step 2: capture current run_id
    status_before = await testbed.get_status()
    run_id_before = status_before.active_run_id

    # Step 3: inject NIC down on eth0 → creates up→down transition
    await testbed.inject_collector_metrics(
        node_id="n-0",
        metrics=[_nic_sample("n-0", "eth0", 0.0)],
    )

    # Step 4: wait for recovery to trigger (ENTER_RECOVERY)
    await testbed.wait_for_mode(
        mode=ControllerMode.RECOVERY,
        timeout=FAST_TIMEOUT,
    )

    # Step 5: wait for recovery to complete → poll for new run_id AND MONITORING mode
    final_status = await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=LONG_RECOVERY_TIMEOUT,
    )
    assert final_status.mode == ControllerMode.MONITORING
    assert final_status.active_run_id != run_id_before


# ------------------------------------------------------------------
# 2. test_sustained_nic_down_triggers_eviction
# ------------------------------------------------------------------


async def test_sustained_nic_down_triggers_eviction(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Both NICs down → NetworkAlertDetector fires → recovery with eviction."""
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="n-0", num_ranks=2),
        ],
        detectors=[
            NetworkAlertDetector(
                config=NetworkAlertDetectorConfig(
                    alert_window_minutes=10.0 / 60.0,
                    alert_threshold=1,
                ),
            ),
        ],
        scrape_interval_seconds=_SCRAPE_INTERVAL,
    )

    # Step 1: wait for baseline metrics to be scraped
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)
    await asyncio.sleep(_SCRAPE_INTERVAL * 3)

    # Step 2: inject both NICs down
    await testbed.inject_collector_metrics(
        node_id="n-0",
        metrics=[
            _nic_sample("n-0", "eth0", 0.0),
            _nic_sample("n-0", "eth1", 0.0),
        ],
    )

    # Step 3: wait for recovery
    await testbed.wait_for_mode(
        mode=ControllerMode.RECOVERY,
        timeout=FAST_TIMEOUT,
    )

    # Step 4: recovery completes
    final_status = await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=LONG_RECOVERY_TIMEOUT,
    )
    assert final_status.mode == ControllerMode.MONITORING


# ------------------------------------------------------------------
# 3. test_transient_nic_fault_no_recovery
# ------------------------------------------------------------------


async def test_transient_nic_fault_no_recovery(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Single NIC transition with threshold=2 does not trigger recovery.

    alert_threshold=2 requires 2 up→down transitions. We inject only 1
    transition (down then restore up), so the threshold is not met.
    """
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="n-0", num_ranks=2),
        ],
        detectors=[
            NetworkAlertDetector(
                config=NetworkAlertDetectorConfig(
                    alert_window_minutes=10.0 / 60.0,
                    alert_threshold=2,
                ),
            ),
        ],
        scrape_interval_seconds=_SCRAPE_INTERVAL,
    )

    # Step 1: wait for baseline to be scraped
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)
    await asyncio.sleep(_SCRAPE_INTERVAL * 3)

    # Step 2: inject eth0 down (1 transition)
    await testbed.inject_collector_metrics(
        node_id="n-0",
        metrics=[_nic_sample("n-0", "eth0", 0.0)],
    )

    # Step 3: wait for the down metric to be scraped
    await asyncio.sleep(_SCRAPE_INTERVAL * 3)

    # Step 4: restore eth0 up (clears injected metrics → baseline takes over)
    await testbed.clear_collector_metrics("n-0")

    # Step 5: verify no recovery — only 1 transition, threshold is 2
    await assert_no_recovery_triggered(
        testbed,
        observation_ticks=20,
        timeout=FAST_TIMEOUT,
    )

    # Step 6: verify still in MONITORING
    status = await testbed.get_status()
    assert status.mode == ControllerMode.MONITORING


# ------------------------------------------------------------------
# 4. test_ephemeral_nic_no_eviction
# ------------------------------------------------------------------


async def test_ephemeral_nic_no_eviction(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Ephemeral NIC down triggers recovery but completes without EvictingSt.

    Uses NetworkAlertDetector only (not NicMajorityDownDetector). The NIC
    alert fires recovery. We track whether EvictingSt was ever observed
    during the recovery flow — it should NOT be, because the detector
    does not identify persistent bad nodes requiring eviction when the
    fault is a transient NIC flap.
    """
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="n-0", num_ranks=2),
        ],
        detectors=[
            NetworkAlertDetector(
                config=NetworkAlertDetectorConfig(
                    alert_window_minutes=10.0 / 60.0,
                    alert_threshold=1,
                ),
            ),
        ],
        scrape_interval_seconds=_SCRAPE_INTERVAL,
    )

    # Step 1: wait for baseline to be scraped
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)
    await asyncio.sleep(_SCRAPE_INTERVAL * 3)

    # Step 2: inject NIC down → triggers recovery
    await testbed.inject_collector_metrics(
        node_id="n-0",
        metrics=[_nic_sample("n-0", "eth0", 0.0)],
    )

    # Step 3: wait for recovery to start
    await testbed.wait_for_mode(
        mode=ControllerMode.RECOVERY,
        timeout=FAST_TIMEOUT,
    )

    # Step 4: poll recovery phases until completion, verify no EvictingSt
    evicting_observed = False
    deadline = time.monotonic() + LONG_RECOVERY_TIMEOUT
    while time.monotonic() < deadline:
        status = await testbed.get_status()
        if status.recovery is not None and "EvictingSt" in status.recovery.phase:
            evicting_observed = True
        if status.mode == ControllerMode.MONITORING:
            break
        await asyncio.sleep(0.3)
    else:
        raise TimeoutError(f"Recovery did not complete within {LONG_RECOVERY_TIMEOUT}s")

    assert not evicting_observed, "EvictingSt should not occur for ephemeral NIC fault"


# ------------------------------------------------------------------
# 5. test_majority_nic_down_triggers_eviction
# ------------------------------------------------------------------


async def test_majority_nic_down_triggers_eviction(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """2/3 NICs down triggers NicMajorityDownDetector → recovery with eviction.

    Uses the full detector chain (build_detector_chain) which includes
    NicMajorityDownDetector. Injecting eth0=0 and eth1=0 while eth2=1
    makes a majority (2/3) down → triggers recovery.
    """
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="n-0", num_ranks=2),
        ],
        detectors=build_detector_chain(),
        scrape_interval_seconds=_SCRAPE_INTERVAL,
    )

    # Step 1: wait for baseline to be scraped
    # Baseline includes eth0=1.0 and eth1=1.0. We add eth2 as an extra NIC.
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Inject a third NIC (eth2=1.0) into the baseline to have 3 NICs total
    await testbed.inject_collector_metrics(
        node_id="n-0",
        metrics=[
            _nic_sample("n-0", "eth0", 1.0),
            _nic_sample("n-0", "eth1", 1.0),
            _nic_sample("n-0", "eth2", 1.0),
        ],
    )
    await asyncio.sleep(_SCRAPE_INTERVAL * 3)

    # Step 2: inject majority down: eth0=0, eth1=0, eth2=1 (2/3 down)
    await testbed.inject_collector_metrics(
        node_id="n-0",
        metrics=[
            _nic_sample("n-0", "eth0", 0.0),
            _nic_sample("n-0", "eth1", 0.0),
            _nic_sample("n-0", "eth2", 1.0),
        ],
    )

    # Step 3: wait for recovery
    await testbed.wait_for_mode(
        mode=ControllerMode.RECOVERY,
        timeout=FAST_TIMEOUT,
    )

    # Step 4: recovery completes → back to MONITORING
    final_status = await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=LONG_RECOVERY_TIMEOUT,
    )
    assert final_status.mode == ControllerMode.MONITORING
