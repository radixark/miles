"""Ray-based implementation of ControllerClientProtocol.

Encapsulates ray.get_actor() lookup and .remote() calls so that agent
code never touches Ray directly.
"""

from __future__ import annotations

import logging
from typing import Any

import ray

from miles.utils.ft.adapters.types import REGISTER_TIMEOUT_SECONDS, ControllerClientProtocol, ft_controller_actor_name
from miles.utils.ft.utils.graceful_degrade import graceful_degrade

logger = logging.getLogger(__name__)


class RayControllerClient(ControllerClientProtocol):
    """Communicates with the FtController Ray actor on behalf of agents.

    Implements :class:`ControllerClientProtocol`.

    * ``register_training_rank`` — synchronous (``ray.get``)
    * ``log_step`` — fire-and-forget (``.remote()`` only)
    """

    def __init__(self, ft_id: str) -> None:
        self._ft_id = ft_id
        self._handle: Any | None = None

    @graceful_degrade(msg="Failed to get ft_controller actor handle")
    def _get_handle(self) -> Any | None:
        if self._handle is None:
            self._handle = ray.get_actor(ft_controller_actor_name(self._ft_id))
        return self._handle

    def register_training_rank(
        self,
        run_id: str,
        rank: int,
        world_size: int,
        node_id: str,
        exporter_address: str,
        pid: int,
        timeout_seconds: float = REGISTER_TIMEOUT_SECONDS,
    ) -> None:
        controller = self._get_handle()
        if controller is None:
            logger.error("ray: register_training_rank failed, controller not available")
            raise RuntimeError("controller not available")

        logger.info(
            "ray: register_training_rank run_id=%s, rank=%d, world_size=%d, node_id=%s, pid=%d",
            run_id, rank, world_size, node_id, pid,
        )
        ray.get(
            controller.register_training_rank.remote(
                run_id=run_id,
                rank=rank,
                world_size=world_size,
                node_id=node_id,
                exporter_address=exporter_address,
                pid=pid,
            ),
            timeout=timeout_seconds,
        )

    def log_step(
        self,
        run_id: str,
        step: int,
        metrics: dict[str, float],
    ) -> None:
        controller = self._get_handle()
        if controller is None:
            logger.debug("ray: log_step skipped, controller handle not available")
            return

        controller.log_step.remote(
            run_id=run_id,
            step=step,
            metrics=metrics,
        )
