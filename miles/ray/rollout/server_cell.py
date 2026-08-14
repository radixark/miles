from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from miles.ray.rollout.server_engine import ServerEngine

if TYPE_CHECKING:
    from miles.ray.rollout.rollout_server import RolloutServer


@dataclass
class ServerCell:
    engines: list[ServerEngine]


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
