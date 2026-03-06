from __future__ import annotations

import logging
import os
import socket
from typing import Any, Literal

from miles.utils.ft.agents.utils.controller_handle import get_controller_handle
from miles.utils.ft.agents.utils.training_rank_heartbeat import TrainingRankHeartbeat
from miles.utils.ft.utils.graceful_degrade import graceful_degrade
from miles.utils.ft.utils.retry import retry_sync

logger = logging.getLogger(__name__)


class FtTrainingRankAgent:
    """Embedded fault-tolerance agent for each training rank.

    Each rank creates one instance. Delegates heartbeat gauges (iteration,
    phase) to a TrainingRankHeartbeat, and handles rank registration with
    the FtController.

    Training metrics (loss, grad_norm, etc.) are forwarded separately through
    FtTrackingAgent, which hooks into tracking_utils.log().
    """

    def __init__(self, rank: int, world_size: int) -> None:
        self._ft_id: str = os.environ.get("MILES_FT_ID", "")
        self._controller_handle: Any | None = None
        self._rank = rank
        self._world_size = world_size
        self._run_id: str = os.environ.get("MILES_FT_TRAINING_RUN_ID", "")
        self._node_id: str = socket.gethostname()

        self._heartbeat = TrainingRankHeartbeat(rank=rank, node_id=self._node_id)

        self._register_training_rank()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    @graceful_degrade()
    def maybe_create(
        cls,
        rank: int,
        world_size: int,
        enabled: bool = True,
    ) -> FtTrainingRankAgent | None:
        if not enabled:
            return None
        return cls(rank=rank, world_size=world_size)

    # ------------------------------------------------------------------
    # Public API — delegated to TrainingRankHeartbeat
    # ------------------------------------------------------------------

    def get_exporter_address(self) -> str:
        return self._heartbeat.get_exporter_address()

    def set_phase(
        self,
        phase: Literal["idle", "training", "checkpoint_saving"],
    ) -> None:
        self._heartbeat.set_phase(phase)

    def step(self, iteration: int) -> None:
        self._heartbeat.step(iteration)

    def shutdown(self) -> None:
        self._heartbeat.shutdown()

    # ------------------------------------------------------------------
    # Internal: controller communication
    # ------------------------------------------------------------------

    _REGISTER_MAX_ATTEMPTS = 3
    _REGISTER_RETRY_DELAY = 2.0

    def _register_training_rank(self) -> None:
        if not self._run_id:
            logger.info("No MILES_FT_TRAINING_RUN_ID set, skipping rank registration")
            return

        if self._controller_handle is None:
            self._controller_handle = get_controller_handle(self._ft_id)
        controller = self._controller_handle
        if controller is None:
            logger.warning("Cannot register rank: controller not available")
            return

        import ray

        def _do_register() -> None:
            ray.get(
                controller.register_training_rank.remote(
                    run_id=self._run_id,
                    rank=self._rank,
                    world_size=self._world_size,
                    node_id=self._node_id,
                    exporter_address=self.get_exporter_address(),
                    pid=os.getpid(),
                ),
                timeout=10,
            )

        result = retry_sync(
            func=_do_register,
            description=f"register_training_rank({self._rank})",
            max_retries=self._REGISTER_MAX_ATTEMPTS,
            backoff_base=self._REGISTER_RETRY_DELAY,
            max_backoff=self._REGISTER_RETRY_DELAY,
        )
        if result.ok:
            logger.info("Rank %d registered successfully", self._rank)
