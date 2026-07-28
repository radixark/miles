from __future__ import annotations

import pytest
from tests.fast.ray.rollout.conftest import (
    fake_actor_handle,
    make_args,
    make_dataclass_cells,
    make_sglang_config_yaml,
)

from miles.ray.rollout import rollout_server
from miles.ray.rollout.cell_state import AddrInfo
from miles.ray.rollout.rollout_server import (
    RolloutServer,
    _compute_megatron_num_gpus,
    _compute_rollout_offset,
    _resolve_sglang_config,
    start_rollout_servers,
)


class TestRolloutServerPureFunctions:
    def test_resolve_sglang_config_yaml_gpu_mismatch_asserts(self, tmp_path):
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "sglang:\n"
            "  - name: actor\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 4\n"
            "        num_gpus_per_engine: 1\n"
        )
        args = make_args(sglang_config=str(cfg_path), rollout_num_gpus=8)
        with pytest.raises(AssertionError, match="total GPUs"):
            _resolve_sglang_config(args)

    def test_compute_rollout_offset_colocate_returns_zero(self):
        args = make_args(
            colocate=True,
            debug_train_only=False,
            debug_rollout_only=False,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            use_critic=False,
        )
        assert _compute_rollout_offset(args) == 0

    def test_compute_rollout_offset_critic_train_only(self):
        args = make_args(
            colocate=False,
            debug_train_only=False,
            debug_rollout_only=False,
            critic_train_only=True,
            critic_num_nodes=1,
            critic_num_gpus_per_node=4,
        )
        assert _compute_rollout_offset(args) == 4

    def test_compute_rollout_offset_shared_actor_critic(self):
        args = make_args(
            colocate=False,
            debug_train_only=False,
            debug_rollout_only=False,
            critic_train_only=False,
            use_critic=True,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            critic_num_nodes=1,
            critic_num_gpus_per_node=4,
        )
        assert _compute_rollout_offset(args) == 8

    def test_compute_megatron_num_gpus_for_actor_only(self):
        args = make_args(
            actor_num_nodes=2,
            actor_num_gpus_per_node=8,
            use_critic=False,
            debug_rollout_only=False,
            critic_train_only=False,
        )
        assert _compute_megatron_num_gpus(args) == 16

    def test_compute_megatron_num_gpus_with_shared_critic(self):
        args = make_args(
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            use_critic=True,
            critic_num_nodes=1,
            critic_num_gpus_per_node=4,
            debug_rollout_only=False,
            critic_train_only=False,
        )
        assert _compute_megatron_num_gpus(args) == 8

    def test_compute_megatron_num_gpus_zero_when_debug_rollout_only(self):
        args = make_args(debug_rollout_only=True)
        assert _compute_megatron_num_gpus(args) == 0


class TestRolloutServerCrossCellProperties:
    def test_api_clients_expose_one_client_per_cell(self):
        """Each cell is addressed through its primary (node-0) endpoint."""
        cells = make_dataclass_cells(num_cells=2, gpu_offset=0) + make_dataclass_cells(num_cells=2, gpu_offset=2)
        for index, cell in enumerate(cells):
            cell._mark_allocated_uninitialized([fake_actor_handle()])
            cell._mark_addressing([AddrInfo(server_url=f"http://10.0.0.{index + 1}:30000")])
        srv = RolloutServer(server_cells={f"cell-{i}": cell for i, cell in enumerate(cells)})
        assert [client.server_url for client in srv.api_clients] == [
            f"http://10.0.0.{index + 1}:30000" for index in range(4)
        ]

    def test_engine_gpu_counts_parallel_to_engines(self):
        cells = make_dataclass_cells(num_cells=2, num_gpus_per_engine=1) + make_dataclass_cells(
            num_cells=2, num_gpus_per_engine=2
        )
        srv = RolloutServer(server_cells={f"cell-{i}": cell for i, cell in enumerate(cells)})
        assert srv.engine_gpu_counts == [1, 1, 2, 2]

    def test_engine_gpu_offsets_consistent_across_cells(self):
        cells = make_dataclass_cells(num_cells=2, num_gpus_per_engine=1, gpu_offset=0) + make_dataclass_cells(
            num_cells=2, num_gpus_per_engine=2, gpu_offset=4
        )
        srv = RolloutServer(server_cells={f"cell-{i}": cell for i, cell in enumerate(cells)})
        assert srv.engine_gpu_offsets == [0, 1, 4, 6]


class TestStartRolloutServersCellChunking:
    @pytest.fixture
    def stub_engine_startup(self, monkeypatch):
        async def _no_cells(self, *args, **kwargs):
            return None

        monkeypatch.setattr(rollout_server, "start_router", lambda *args, **kwargs: ("127.0.0.1", 30000))
        monkeypatch.setattr(RolloutServer, "start_all_cells", _no_cells)

    def _cells_for(self, tmp_path, *, num_gpus: int, num_gpus_per_engine: int):
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            make_sglang_config_yaml(
                server_groups=[
                    {"worker_type": "regular", "num_gpus": num_gpus, "num_gpus_per_engine": num_gpus_per_engine}
                ]
            )
        )
        args = make_args(sglang_config=str(cfg_path), rollout_num_gpus=num_gpus, num_gpus_per_node=8)
        return start_rollout_servers(args, pg=None)["default"].server_cells

    def test_a_single_node_engine_becomes_its_own_cell(self, stub_engine_startup, tmp_path):
        """With one gpu per engine on 8-gpu nodes, every engine is a one-engine cell."""
        cells = self._cells_for(tmp_path, num_gpus=8, num_gpus_per_engine=1)
        assert [cell.num_nodes for cell in cells] == [1] * 8

    def test_a_multi_node_engine_chunks_its_node_ranks_into_one_cell(self, stub_engine_startup, tmp_path):
        """With 16 gpus per engine on 8-gpu nodes, each cell holds both node-ranks."""
        cells = self._cells_for(tmp_path, num_gpus=32, num_gpus_per_engine=16)
        assert [cell.num_nodes for cell in cells] == [2, 2]

    def test_a_trailing_partial_multi_node_engine_is_rejected(self, stub_engine_startup, tmp_path):
        """24 gpus do not divide into whole 2-node engines, so startup must fail fast."""
        with pytest.raises(AssertionError, match="whole number of"):
            self._cells_for(tmp_path, num_gpus=24, num_gpus_per_engine=16)

    def test_cells_carry_contiguous_rank_and_gpu_offsets(self, stub_engine_startup, tmp_path):
        """Each multi-node cell starts where the previous one ended, so node-0 detection stays valid."""
        cells = self._cells_for(tmp_path, num_gpus=32, num_gpus_per_engine=16)
        assert [cell.rank_offset for cell in cells] == [0, 2]
        assert [cell.gpu_offset for cell in cells] == [0, 16]

    def test_every_multi_node_cell_starts_on_an_aligned_rank(self, stub_engine_startup, tmp_path):
        """sglang derives node_rank from the global rank, so a cell must not start mid-engine."""
        cells = self._cells_for(tmp_path, num_gpus=32, num_gpus_per_engine=16)
        for cell in cells:
            assert cell.rank_offset % cell.num_nodes == 0

    def test_a_group_starting_at_a_misaligned_rank_is_rejected(self, stub_engine_startup, tmp_path):
        """One single-node engine ahead of a 2-node group leaves an odd engine_offset and must fail fast."""
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            make_sglang_config_yaml(
                server_groups=[
                    {"worker_type": "prefill", "num_gpus": 1, "num_gpus_per_engine": 1},
                    {"worker_type": "decode", "num_gpus": 32, "num_gpus_per_engine": 16},
                ]
            )
        )
        args = make_args(sglang_config=str(cfg_path), rollout_num_gpus=33, num_gpus_per_node=8)
        with pytest.raises(AssertionError, match="not aligned to"):
            start_rollout_servers(args, pg=None)
