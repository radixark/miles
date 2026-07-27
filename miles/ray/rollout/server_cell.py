from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from miles.ray.rollout.server_engine import ServerEngine

if TYPE_CHECKING:
    from miles.ray.rollout.rollout_server import RolloutServer


@dataclass
class ServerCell:
    engines: list[ServerEngine]

    @property
    def primary_engine(self) -> ServerEngine:
        return self.engines[0]

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
