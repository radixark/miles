import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import ray

from miles.ray.rollout.server_engine import ServerEngine

if TYPE_CHECKING:
    from miles.ray.rollout.rollout_server import RolloutServer

logger = logging.getLogger(__name__)

SHUTDOWN_TIMEOUT = 30


@dataclass
class ServerCell:
    engines: list[ServerEngine]

    @property
    def primary_engine(self) -> ServerEngine:
        return self.engines[0]

    def stop(self):
        for local_index, engine in enumerate(self.engines):
            if engine.is_allocated:
                logger.info(f"Shutting down and killing engine at cell-local index {local_index}")
                try:
                    ray.get(engine.actor_handle.shutdown.remote(), timeout=SHUTDOWN_TIMEOUT)
                except Exception as e:
                    logger.warning(
                        f"Graceful shutdown of engine at cell-local index {local_index} failed, killing anyway (e: {e})"
                    )
                try:
                    ray.kill(engine.actor_handle)
                    logger.info(f"Successfully killed engine at cell-local index {local_index}")
                except Exception as e:
                    logger.warning(f"Fail to kill engine at cell-local index {local_index} (e: {e})")
            else:
                logger.info(f"Engine at cell-local index {local_index} is already None")
            self.engines[local_index].mark_stopped()

    async def offload(self, tags: list[str] | None):
        return await self.primary_engine.api_client.release_memory_occupation(tags=tags)

    async def onload(self, tags: list[str] | None):
        return await self.primary_engine.api_client.resume_memory_occupation(tags=tags)

    async def check_weights(self, action: str, allow_quant_error: bool, selector: str, skip_list: list[str] | None):
        return await self.primary_engine.api_client.check_weights(
            action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list
        )


def flatten_cells(cells: list[ServerCell]) -> list[ServerEngine]:
    return [engine for cell in cells for engine in cell.engines]


class CellIndexer(NamedTuple):
    srv_key: str
    group_index: int
    engine_indices: list[int]


def get_cell_indexer_of_id_map(servers: dict[str, "RolloutServer"]) -> list[CellIndexer]:
    """Flatten ``servers`` into a list whose position is the cell id.

    ``engine_indices`` covers the cell's entries in the group's flat engine
    list. Order is sorted by ``srv_key``, so cell ids are stable across calls
    when the topology is unchanged.
    """
    result: list[CellIndexer] = []
    for srv_key in sorted(servers):
        srv = servers[srv_key]
        for group_index, group in enumerate(srv.server_groups):
            engine_offset = 0
            for cell in group.cells:
                result.append(
                    CellIndexer(
                        srv_key=srv_key,
                        group_index=group_index,
                        engine_indices=list(range(engine_offset, engine_offset + len(cell.engines))),
                    )
                )
                engine_offset += len(cell.engines)
    return result
