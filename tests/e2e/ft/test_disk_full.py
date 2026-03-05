"""E2E Scenario 5: Disk full → hardware detection → eviction.

Validates the highest-confidence hardware detection path (~32.5% per ByteRobust):
  1. Fill disk on target node (write large file)
  2. HighConfidenceHardwareDetector triggers MARK_BAD_AND_RESTART
  3. Target node is marked as bad
  4. Training restarts on remaining healthy nodes
  5. Cleanup: remove filled file
"""

from __future__ import annotations

import logging
import time

import pytest
import ray
from miles.utils.ft.models import ControllerMode
from tests.e2e.ft.conftest import FaultInjectorFactory, FtSystem, wait_for_recovery_complete, wait_for_training_stable

logger = logging.getLogger(__name__)

_FILL_SIZE_BYTES = 200 * 1024 * 1024 * 1024  # 200 GB — enough to trigger low disk threshold
_FILL_PATH = "/data/ft_e2e_disk_fill_test"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.timeout(600),
]


async def test_disk_full_eviction(
    ft_system: FtSystem,
    fault_injector: FaultInjectorFactory,
    target_node: str,
) -> None:
    controller = ft_system.controller

    await wait_for_training_stable(
        controller=controller,
        mini_wandb=ft_system.mini_wandb,
        n_iterations=5,
        timeout=300.0,
    )

    injector = fault_injector.deploy_to(node_id=target_node)

    try:
        t_inject = time.monotonic()
        ray.get(
            injector.fill_disk.remote(
                path=_FILL_PATH,
                size_bytes=_FILL_SIZE_BYTES,
            )
        )
        logger.info("disk_filled path=%s size=%d node=%s", _FILL_PATH, _FILL_SIZE_BYTES, target_node)

        # Wait for HighConfidenceHardwareDetector to trigger MARK_BAD_AND_RESTART
        status = await wait_for_recovery_complete(
            controller=controller,
            timeout=300.0,
        )

        t_evict = time.monotonic() - t_inject
        logger.info("disk_full_eviction_complete t_evict=%.1fs", t_evict)

        assert status.mode == ControllerMode.MONITORING

        # bad_nodes in get_status() reflects _diagnosing_nodes which is
        # cleared after recovery completes. Check node_manager directly.
        node_bad = await ft_system.node_manager.get_bad_nodes()
        assert target_node in node_bad or any(
            target_node in str(n) for n in node_bad
        ), f"Expected {target_node} in bad nodes, got: {node_bad}"

        # Training should recover on remaining healthy nodes
        await wait_for_training_stable(
            controller=controller,
            mini_wandb=ft_system.mini_wandb,
            n_iterations=10,
            timeout=300.0,
        )

    finally:
        ray.get(injector.cleanup_disk.remote(path=_FILL_PATH))
        logger.info("disk_fill_cleaned path=%s", _FILL_PATH)
