"""E2E: Disk full → hardware detection → eviction.

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
from miles.utils.ft.models import ControllerMode, RecoveryPhase
from miles.utils.ft.platform.k8s_node_manager import K8sNodeManager
from tests.e2e.ft.conftest import (
    FaultInjectorFactory,
    assert_phase_path_contains,
    get_status,
    wait_for_recovery_complete,
    wait_for_training_stable,
)

logger = logging.getLogger(__name__)

_FILL_SIZE_BYTES = 200 * 1024 * 1024 * 1024  # 200 GB
_FILL_PATH = "/data/ft_e2e_disk_fill_test"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.timeout(600),
]


async def test_disk_full_eviction(
    ft_controller_handle: ray.actor.ActorHandle,
    fault_injector: FaultInjectorFactory,
    target_node: str,
    _cleanup_node_manager: K8sNodeManager,
) -> None:
    await wait_for_training_stable(
        handle=ft_controller_handle,
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

        status = await wait_for_recovery_complete(
            handle=ft_controller_handle,
            timeout=300.0,
        )

        t_evict = time.monotonic() - t_inject
        logger.info("disk_full_eviction_complete t_evict=%.1fs", t_evict)

        assert status.mode == ControllerMode.MONITORING

        node_bad = await _cleanup_node_manager.get_bad_nodes()
        assert target_node in node_bad or any(
            target_node in str(n) for n in node_bad
        ), f"Expected {target_node} in bad nodes, got: {node_bad}"

        final_status = get_status(ft_controller_handle)
        if final_status.phase_history is not None:
            assert_phase_path_contains(final_status, [
                RecoveryPhase.EVICT_AND_RESTART,
                RecoveryPhase.DONE,
            ])

        await wait_for_training_stable(
            handle=ft_controller_handle,
            n_iterations=10,
            timeout=300.0,
        )

    finally:
        ray.get(injector.cleanup_disk.remote(path=_FILL_PATH))
        logger.info("disk_fill_cleaned path=%s", _FILL_PATH)
