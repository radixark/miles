from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from miles.utils.ft.adapters.types import STOP_TRAINING_TIMEOUT_SECONDS, JobStatus, SubsystemActuatorProtocol

logger = logging.getLogger(__name__)


class RayRolloutActuator(SubsystemActuatorProtocol):
    """Per-cell SubsystemActuatorProtocol via Ray remote calls to RolloutManager.

    Each rollout subsystem has its own RayRolloutActuator instance.
    All instances share the same RolloutManager Ray actor handle, accessed
    lazily through ``get_handle`` so the handle can be set after construction.
    The handle exposes stop_cell/start_cell/get_cell_status as remote methods.
    """

    def __init__(
        self,
        *,
        get_handle: Callable[[], object],
        cell_id: str,
    ) -> None:
        self._get_handle = get_handle
        self._cell_id = cell_id

    async def start(self) -> str:
        logger.info("rollout_actuator_start cell_id=%s", self._cell_id)
        result = await self._get_handle().start_cell.remote(self._cell_id)  # type: ignore[attr-defined]
        return str(result)

    async def stop(self, timeout_seconds: int = STOP_TRAINING_TIMEOUT_SECONDS) -> None:
        logger.info("rollout_actuator_stop cell_id=%s timeout=%d", self._cell_id, timeout_seconds)
        try:
            await asyncio.wait_for(
                self._get_handle().stop_cell.remote(self._cell_id),  # type: ignore[attr-defined]
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "ray: rollout_actuator_stop timed out cell_id=%s, timeout=%d",
                self._cell_id,
                timeout_seconds,
            )
            raise TimeoutError(
                f"rollout actuator stop timed out after {timeout_seconds}s for cell_id={self._cell_id}"
            ) from None

    async def get_status(self) -> JobStatus:
        logger.debug("ray: rollout_actuator_get_status cell_id=%s", self._cell_id)
        status = await self._get_handle().get_cell_status.remote(self._cell_id)  # type: ignore[attr-defined]
        logger.debug("ray: rollout_actuator_get_status cell_id=%s, status=%s", self._cell_id, status)
        return status
