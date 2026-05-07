from typing import NamedTuple

from miles.ray.rollout.rollout_server import RolloutServer


class CellIndexer(NamedTuple):
    srv_key: str
    group_index: int
    engine_indices: list[int]


def get_cell_indexer_of_id_map(servers: dict[str, RolloutServer]) -> list[CellIndexer]:
    """Flatten ``servers`` into a list whose position is the cell id.

    A cell is one node-0 engine; ``engine_indices`` covers its ``nodes_per_engine``
    underlying entries in ``group.all_engines``. Order is sorted by ``srv_key``, so
    cell ids are stable across calls when the topology is unchanged.
    """
    result: list[CellIndexer] = []
    for srv_key in sorted(servers):
        srv = servers[srv_key]
        for group_index, group in enumerate(srv.server_groups):
            assert len(group.all_engines) == len(group.engines) * group.nodes_per_engine
            for local_index in range(len(group.engines)):
                result.append(
                    CellIndexer(
                        srv_key=srv_key,
                        group_index=group_index,
                        engine_indices=list(
                            range(local_index * group.nodes_per_engine, (local_index + 1) * group.nodes_per_engine)
                        ),
                    )
                )
    return result
