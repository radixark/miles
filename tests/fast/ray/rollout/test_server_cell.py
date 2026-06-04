from __future__ import annotations

from unittest.mock import MagicMock

from tests.fast.ray.rollout.conftest import fake_actor_handle, make_args

from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import get_cell_indexer_of_id_map
from miles.ray.rollout.server_engine import ServerEngine
from miles.ray.rollout.server_group import ServerGroup


def _build_servers(
    *, num_servers: int = 1, groups_per_server: int = 1, engines_per_group: int = 2, num_gpus_per_engine: int = 1
) -> dict[str, RolloutServer]:
    args = make_args(num_gpus_per_node=8)
    servers: dict[str, RolloutServer] = {}
    for s_idx in range(num_servers):
        groups = []
        for _g in range(groups_per_server):
            engines = [ServerEngine() for _ in range(engines_per_group)]
            for e in engines:
                e.mark_allocated_uninitialized(fake_actor_handle())
                e.mark_alive()
            groups.append(
                ServerGroup(
                    args=args,
                    pg=None,
                    all_engines=engines,
                    num_gpus_per_engine=num_gpus_per_engine,
                    has_new_engines=False,
                    update_weights=True,
                )
            )
        servers[f"model_{s_idx}"] = RolloutServer(
            server_groups=groups,
            model_name=f"model_{s_idx}",
            update_weights=True,
        )
    return servers


class TestGetCellIndexerOfIdMap:
    def test_single_server_single_group_one_cell_per_engine(self):
        """Happy path: one server with one group of N engines → N cells, each
        engine_indices=[i], all under model_0/group_0."""
        servers = _build_servers(num_servers=1, groups_per_server=1, engines_per_group=3)
        cells = get_cell_indexer_of_id_map(servers)
        assert len(cells) == 3
        for i, cell in enumerate(cells):
            assert cell.srv_key == "model_0"
            assert cell.group_index == 0
            assert cell.engine_indices == [i]

    def test_multi_group_cells_increment_continuously_across_groups(self):
        servers = _build_servers(num_servers=1, groups_per_server=2, engines_per_group=2)
        cells = get_cell_indexer_of_id_map(servers)
        assert len(cells) == 4
        # cells 0,1 → group 0; cells 2,3 → group 1
        assert [c.group_index for c in cells] == [0, 0, 1, 1]
        assert all(c.srv_key == "model_0" for c in cells)

    def test_multi_server_ordered_by_key_alphabetically(self):
        """When multiple servers exist, cells are emitted in srv_key order."""
        servers = _build_servers(num_servers=2, groups_per_server=1, engines_per_group=1)
        cells = get_cell_indexer_of_id_map(servers)
        srv_keys_in_order = [c.srv_key for c in cells]
        assert srv_keys_in_order == sorted(srv_keys_in_order)
        assert srv_keys_in_order == ["model_0", "model_1"]

    def test_multinode_engine_cells_span_contiguous_engine_slots(self):
        """num_gpus_per_engine=16 and num_gpus_per_node=8 → nodes_per_engine=2;
        each cell maps to 2 contiguous engine slots."""
        servers = _build_servers(num_servers=1, groups_per_server=1, engines_per_group=2, num_gpus_per_engine=16)
        cells = get_cell_indexer_of_id_map(servers)
        assert len(cells) == 1
        assert cells[0].engine_indices == [0, 1]

    def test_placeholder_group_with_zero_engines_emits_zero_cells(self):
        """``placeholder`` worker_type groups have empty all_engines/engines.
        The internal assertion ``len(all_engines) == len(engines) *
        nodes_per_engine`` still holds (0 == 0 * N)."""
        srv = MagicMock()
        group = MagicMock()
        group.all_engines = []
        group.engines = []
        group.nodes_per_engine = 2
        srv.server_groups = [group]
        out = get_cell_indexer_of_id_map({"only": srv})
        assert out == []

    def test_empty_server_dict_returns_empty_list(self):
        assert get_cell_indexer_of_id_map({}) == []
