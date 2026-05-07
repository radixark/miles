from typing import NamedTuple

from miles.ray.rollout.rollout_server import RolloutServer


class CellIndexer(NamedTuple):
    srv_key: str
    group_index: int
    engine_indices: list[int]


def get_cell_indexer_of_id_map(servers: dict[str, RolloutServer]) -> list[CellIndexer]:
    """Flatten ``servers`` into a list of ``CellIndexer`` whose position is the cell id.

    Each cell is one node-0 engine. The returned list assigns a stable cell id (its
    list index) to every cell across all servers, and lets callers resolve a cell id
    back to ``(srv_key, group_index, engine_indices)`` so they can address the
    underlying engines (engine_indices spans ``nodes_per_engine`` consecutive entries
    of ``group.all_engines`` for multi-node serving). Iteration order is sorted by
    ``srv_key``, then ``group_index``, then local engine index, so cell ids are
    deterministic across calls as long as the server topology is unchanged.
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
