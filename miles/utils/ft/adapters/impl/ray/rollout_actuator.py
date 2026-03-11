from __future__ import annotations

import logging

from miles.utils.ft.adapters.types import JobStatus, SubsystemActuatorProtocol

logger = logging.getLogger(__name__)


class RayRolloutActuator(SubsystemActuatorProtocol):
    """Per-cell SubsystemActuatorProtocol via Ray remote calls to RolloutManager.

    Each rollout subsystem has its own RayRolloutActuator instance.
    All instances share the same rm_handle (RolloutManager Ray actor handle).
    rm_handle exposes stop_cell/start_cell/get_cell_status as remote methods.
    """

    def __init__(
        self,
        *,
        rm_handle: object,
        cell_id: str,
    ) -> None:
        self._rm_handle = rm_handle
        self._cell_id = cell_id

    async def stop(self) -> None:
        logger.info("rollout_actuator_stop cell_id=%s", self._cell_id)
        await self._rm_handle.stop_cell.remote(self._cell_id)  # type: ignore[attr-defined]

    async def start(self) -> str:
        logger.info("rollout_actuator_start cell_id=%s", self._cell_id)
        result = await self._rm_handle.start_cell.remote(self._cell_id)  # type: ignore[attr-defined]

        if isinstance(result, dict):
            return str(result["count"])
        return str(result)

    async def get_status(self) -> JobStatus:
        return await self._rm_handle.get_cell_status.remote(self._cell_id)  # type: ignore[attr-defined]
