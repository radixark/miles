from __future__ import annotations

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_dataclass_group, make_sglang_config_yaml

from miles.ray.rollout import rollout_server
from miles.ray.rollout.rollout_server import (
    RolloutServer,
    _compute_megatron_num_gpus,
    _compute_rollout_offset,
    _resolve_sglang_config,
    start_rollout_servers,
)
from miles.ray.rollout.server_group import ServerGroup


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


class TestRolloutServerCrossGroupProperties:
    def test_engines_collects_node0_engines_from_each_group(self):
        a = make_dataclass_group(num_engines=2, gpu_offset=0)
        b = make_dataclass_group(num_engines=2, gpu_offset=2)
        srv = RolloutServer(server_groups=[a, b])
        assert len(srv.engines) == 4

    def test_engine_gpu_counts_parallel_to_engines(self):
        a = make_dataclass_group(num_engines=2, num_gpus_per_engine=1)
        b = make_dataclass_group(num_engines=2, num_gpus_per_engine=2)
        srv = RolloutServer(server_groups=[a, b])
        assert srv.engine_gpu_counts == [1, 1, 2, 2]

    def test_engine_gpu_offsets_consistent_across_groups(self):
        a = make_dataclass_group(num_engines=2, num_gpus_per_engine=1, gpu_offset=0)
        b = make_dataclass_group(num_engines=2, num_gpus_per_engine=2, gpu_offset=4)
        srv = RolloutServer(server_groups=[a, b])
        assert srv.engine_gpu_offsets == [0, 1, 4, 6]


class TestRolloutServerNodesPerEngineHeterogeneity:
    def test_homogeneous_groups_return_single_value(self):
        a = make_dataclass_group(num_gpus_per_engine=1)
        b = make_dataclass_group(num_gpus_per_engine=1)
        srv = RolloutServer(server_groups=[a, b])
        assert srv.nodes_per_engine == 1

    def test_heterogeneous_groups_raise_value_error(self):
        # 1 gpu/engine vs 16 gpu/engine on 8-gpu nodes → 1 vs 2 nodes/engine
        a = make_dataclass_group(num_gpus_per_engine=1)
        b = make_dataclass_group(num_gpus_per_engine=16)
        srv = RolloutServer(server_groups=[a, b])
        with pytest.raises(ValueError, match="Heterogeneous nodes_per_engine"):
            _ = srv.nodes_per_engine


class TestStartRolloutServersCellChunking:
    @pytest.fixture
    def stub_engine_startup(self, monkeypatch):
        async def _no_engines(self, *args, **kwargs):
            return []

        monkeypatch.setattr(rollout_server, "start_router", lambda *args, **kwargs: ("127.0.0.1", 30000))
        monkeypatch.setattr(ServerGroup, "start_engines", _no_engines)

    def _servers_for(self, tmp_path, *, num_gpus: int, num_gpus_per_engine: int):
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            make_sglang_config_yaml(
                server_groups=[
                    {"worker_type": "regular", "num_gpus": num_gpus, "num_gpus_per_engine": num_gpus_per_engine}
                ]
            )
        )
        args = make_args(sglang_config=str(cfg_path), rollout_num_gpus=num_gpus, num_gpus_per_node=8)
        return start_rollout_servers(args, pg=None)

    def test_a_single_node_engine_becomes_its_own_cell(self, stub_engine_startup, tmp_path):
        """With one gpu per engine on 8-gpu nodes, every engine is a one-engine cell."""
        servers = self._servers_for(tmp_path, num_gpus=8, num_gpus_per_engine=1)
        (group,) = servers["default"].server_groups
        assert group.nodes_per_engine == 1
        assert [len(cell.engines) for cell in group.cells] == [1] * 8

    def test_a_multi_node_engine_chunks_its_node_ranks_into_one_cell(self, stub_engine_startup, tmp_path):
        """With 16 gpus per engine on 8-gpu nodes, each cell holds both node-ranks."""
        servers = self._servers_for(tmp_path, num_gpus=32, num_gpus_per_engine=16)
        (group,) = servers["default"].server_groups
        assert group.nodes_per_engine == 2
        assert [len(cell.engines) for cell in group.cells] == [2, 2]

    def test_a_trailing_partial_multi_node_engine_is_rejected(self, stub_engine_startup, tmp_path):
        """24 gpus do not divide into whole 2-node engines, so startup must fail fast."""
        with pytest.raises(AssertionError, match="whole number of"):
            self._servers_for(tmp_path, num_gpus=24, num_gpus_per_engine=16)

    def test_cells_carry_contiguous_rank_and_gpu_offsets(self, stub_engine_startup, tmp_path):
        """Each multi-node cell starts where the previous one ended, so node-0 detection stays valid."""
        servers = self._servers_for(tmp_path, num_gpus=32, num_gpus_per_engine=16)
        (group,) = servers["default"].server_groups
        assert [cell.rank_offset for cell in group.cells] == [0, 2]
        assert [cell.gpu_offset for cell in group.cells] == [0, 16]
