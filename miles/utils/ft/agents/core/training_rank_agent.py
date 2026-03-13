from __future__ import annotations

import logging
import os
import socket
from typing import TYPE_CHECKING, Literal

from miles.utils.ft.agents.metrics.training_rank_exporter import TrainingRankExporter
from miles.utils.ft.utils.env import get_run_id
from miles.utils.ft.utils.graceful_degrade import graceful_degrade
from miles.utils.ft.utils.retry import retry_sync

if TYPE_CHECKING:
    from miles.utils.ft.adapters.types import ControllerClientProtocol

logger = logging.getLogger(__name__)


class FtTrainingRankAgent:
    """Embedded fault-tolerance agent for each training rank.

    Each rank creates one instance. Delegates metric exposition (iteration,
    phase gauges) to a TrainingRankExporter, and handles rank
    registration with the FtController.

    Training metrics (loss, grad_norm, etc.) are forwarded separately through
    FtTrackingAgent, which hooks into tracking_utils.log().
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        controller_client: ControllerClientProtocol | None = None,
        node_id: str | None = None,
    ) -> None:
        self._rank = rank
        self._world_size = world_size
        self._controller_client = controller_client
        self._run_id: str = get_run_id()
        self._node_id: str = node_id or socket.gethostname()

        self._metric_exporter = TrainingRankExporter(
            rank=rank,
            node_id=self._node_id,
            run_id=self._run_id,
        )

        logger.info(
            "training_rank_agent: initializing: rank=%d, world_size=%d, node_id=%s, run_id=%s",
            rank,
            world_size,
            self._node_id,
            self._run_id,
        )
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
        controller_client: ControllerClientProtocol | None = None,
        node_id: str | None = None,
    ) -> FtTrainingRankAgent | None:
        if not enabled:
            logger.debug("training_rank_agent: creation skipped, not enabled: rank=%d", rank)
            return None
        return cls(rank=rank, world_size=world_size, controller_client=controller_client, node_id=node_id)

    # ------------------------------------------------------------------
    # Public API — delegated to TrainingRankExporter
    # ------------------------------------------------------------------

    def get_exporter_address(self) -> str:
        return self._metric_exporter.get_exporter_address()

    def set_phase(
        self,
        phase: Literal["idle", "training", "checkpoint_saving"],
    ) -> None:
        logger.debug("training_rank_agent: phase transition: rank=%d, phase=%s", self._rank, phase)
        self._metric_exporter.set_phase(phase)

    def step(self) -> None:
        self._metric_exporter.step()

    def shutdown(self) -> None:
        logger.info("training_rank_agent: shutting down: rank=%d", self._rank)
        self._metric_exporter.shutdown()

    # ------------------------------------------------------------------
    # Internal: controller communication
    # ------------------------------------------------------------------

    _REGISTER_MAX_ATTEMPTS = 3
    _REGISTER_RETRY_DELAY = 2.0

    def _register_training_rank(self) -> None:
        if self._controller_client is None:
            logger.warning("training_rank_agent: cannot register rank, no controller client: rank=%d", self._rank)
            return

        self._metric_exporter.wait_until_ready()

        def _do_register() -> None:
            self._controller_client.register_training_rank(  # type: ignore[union-attr]
                run_id=self._run_id,
                rank=self._rank,
                world_size=self._world_size,
                node_id=self._node_id,
                exporter_address=self.get_exporter_address(),
                pid=os.getpid(),
            )

        result = retry_sync(
            func=_do_register,
            description=f"register_training_rank({self._rank})",
            max_retries=self._REGISTER_MAX_ATTEMPTS,
            backoff_base=self._REGISTER_RETRY_DELAY,
            max_backoff=self._REGISTER_RETRY_DELAY,
        )
        if result.ok:
            logger.info(
                "training_rank_agent: rank registered successfully: rank=%d, run_id=%s", self._rank, self._run_id
            )
        else:
            logger.warning(
                "training_rank_agent: rank registration failed: rank=%d, max_retries=%d, error=%s",
                self._rank,
                self._REGISTER_MAX_ATTEMPTS,
                result.exception,
                exc_info=result.exception,
            )
