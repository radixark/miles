"""Semi-E2E: thermal throttling — temp spike with/without MFU decline."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

import pytest
from tests.fast.utils.ft.integration.conftest import FAST_TIMEOUT, LONG_RECOVERY_TIMEOUT
from tests.fast.utils.ft.integration.local_ray_semi_e2e.conftest import assert_no_recovery_triggered
from tests.fast.utils.ft.testbed import MilesTestbed, TestbedNodeConfig

from miles.utils.ft.agents.types import GaugeSample
from miles.utils.ft.controller.detectors.core.thermal_throttling import (
    ThermalThrottlingDetector,
    ThermalThrottlingDetectorConfig,
)
from miles.utils.ft.controller.detectors.core.training_crash import TrainingCrashDetector
from miles.utils.ft.controller.types import ControllerMode
from miles.utils.ft.utils.metric_names import DCGM_FI_DEV_GPU_TEMP

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.anyio,
]

_THERMAL_CONFIG = ThermalThrottlingDetectorConfig(
    temperature_delta_threshold=20.0,
    mfu_decline_threshold_ratio=0.9,
    mfu_baseline=0.5,
    mfu_consecutive_steps=2,
    mfu_baseline_steps=2,
)


# ------------------------------------------------------------------
# 1. test_temperature_spike_with_mfu_decline_triggers_recovery
# ------------------------------------------------------------------


async def test_temperature_spike_with_mfu_decline_triggers_recovery(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """High GPU temp on one node + MFU decline triggers ThermalThrottlingDetector recovery."""
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="n-0", num_ranks=2),
            TestbedNodeConfig(node_id="n-1", num_ranks=2),
        ],
        detectors=[
            TrainingCrashDetector(),
            ThermalThrottlingDetector(config=_THERMAL_CONFIG),
        ],
        scrape_interval_seconds=0.5,
    )

    # Step 1: verify training is stable
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)
    old_run_id = (await testbed.get_status()).active_run_id

    # Step 2: inject high temperature on n-1 (baseline on n-0 is 45C)
    await testbed.inject_collector_metrics(
        node_id="n-1",
        metrics=[
            GaugeSample(
                name=DCGM_FI_DEV_GPU_TEMP,
                labels={"node_id": "n-1"},
                value=90.0,
            ),
        ],
    )

    # Step 3: inject low MFU to confirm thermal throttling is degrading performance
    await testbed.inject_custom_metrics({"mfu": 0.1})

    # Step 4: wait for recovery to complete (run_id changes + back to MONITORING)
    deadline = time.monotonic() + LONG_RECOVERY_TIMEOUT
    while time.monotonic() < deadline:
        status = await testbed.get_status()
        if status.active_run_id != old_run_id and status.mode == ControllerMode.MONITORING:
            break
        await asyncio.sleep(0.5)
    else:
        raise TimeoutError(
            f"Thermal throttling recovery did not complete within {LONG_RECOVERY_TIMEOUT}s: "
            f"run_id changed={status.active_run_id != old_run_id}, mode={status.mode}"
        )

    assert status.mode == ControllerMode.MONITORING


# ------------------------------------------------------------------
# 2. test_temperature_spike_without_mfu_decline_no_recovery
# ------------------------------------------------------------------


async def test_temperature_spike_without_mfu_decline_no_recovery(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """High GPU temp alone (without MFU decline) does NOT trigger recovery."""
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="n-0", num_ranks=2),
            TestbedNodeConfig(node_id="n-1", num_ranks=2),
        ],
        detectors=[
            TrainingCrashDetector(),
            ThermalThrottlingDetector(config=_THERMAL_CONFIG),
        ],
        scrape_interval_seconds=0.5,
    )

    # Step 1: verify training is stable
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 2: inject high temperature on n-1 only (no MFU injection)
    await testbed.inject_collector_metrics(
        node_id="n-1",
        metrics=[
            GaugeSample(
                name=DCGM_FI_DEV_GPU_TEMP,
                labels={"node_id": "n-1"},
                value=90.0,
            ),
        ],
    )

    # Step 3: allow scrape + detector cycles to propagate the temperature data
    await asyncio.sleep(1.5)

    # Step 4: verify no recovery triggered (temp outlier alone is insufficient)
    await assert_no_recovery_triggered(
        testbed,
        observation_ticks=20,
        timeout=FAST_TIMEOUT,
    )
