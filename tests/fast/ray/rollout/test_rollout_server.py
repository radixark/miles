from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_dataclass_cells, make_sglang_config_yaml

from miles.backends.sglang_utils.sglang_config import (
    _compute_megatron_num_gpus,
    _compute_rollout_offset,
    resolve_sglang_config,
)
from miles.ray.rollout import rollout_server
from miles.ray.rollout.cell_state import AddrInfo
from miles.ray.rollout.rollout_server import RolloutServer, start_rollout_servers
from miles.utils.workers.worker_spec import HostAndPort


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
            resolve_sglang_config(args)

    def test_eval_fleet_inherits_rollout_engine_settings(self):
        """The eval model carries only what makes it an eval fleet; the rest is inherited."""
        args = make_args(eval_num_gpus=2, eval_num_gpus_per_engine=2)
        config = _resolve_sglang_config(args)

        [eval_model] = [m for m in config.models if m.name == "eval"]
        assert eval_model.update_weights is False
        [group] = eval_model.server_groups
        assert (group.num_gpus, group.num_gpus_per_engine) == (2, 2)
        # Eval samples never feed training, so the replay side-channels are forced off.
        assert group.overrides["enable_return_routed_experts"] is False
        assert group.overrides["enable_return_indexer_topk"] is False

        # The fleet boots on --hf-checkpoint; every eval overwrites those weights anyway.
        eval_model.resolve(args)
        assert group.overrides["model_path"] == args.hf_checkpoint

    def test_tp_coupled_sizes_are_not_inherited_across_a_different_eval_tp(self):
        """Inheriting these across a different eval tp gives an engine that will not boot."""
        args = make_args(rollout_num_gpus_per_engine=8, eval_num_gpus=1, eval_num_gpus_per_engine=1)
        [group] = [m for m in _resolve_sglang_config(args).models if m.name == "eval"][0].server_groups

        assert {k: group.overrides[k] for k in ("dp_size", "pp_size", "ep_size", "attn_cp_size")} == dict.fromkeys(
            ("dp_size", "pp_size", "ep_size", "attn_cp_size"), 1
        )

    def test_tp_coupled_sizes_are_inherited_when_the_tp_matches(self):
        args = make_args(rollout_num_gpus_per_engine=2, eval_num_gpus=2, eval_num_gpus_per_engine=2)
        [group] = [m for m in _resolve_sglang_config(args).models if m.name == "eval"][0].server_groups

        assert "ep_size" not in group.overrides  # left to the shared --sglang-* fill-in

    def test_eval_sglang_overrides_win_over_the_tp_coupled_defaults(self):
        args = make_args(
            rollout_num_gpus_per_engine=8,
            eval_num_gpus=2,
            eval_num_gpus_per_engine=2,
            eval_sglang_ep_size=2,
        )
        [group] = [m for m in _resolve_sglang_config(args).models if m.name == "eval"][0].server_groups

        assert group.overrides["ep_size"] == 2

    def test_eval_sglang_overrides_reach_the_eval_group_only(self):
        args = make_args(eval_num_gpus=2, eval_sglang_mem_fraction_static=0.95)
        config = _resolve_sglang_config(args)

        by_name = {m.name: m for m in config.models}
        assert by_name["eval"].server_groups[0].overrides["mem_fraction_static"] == 0.95
        assert by_name["default"].server_groups[0].overrides == {}

    def test_yaml_eval_model_is_filled_from_cli_without_clobbering(self, tmp_path):
        """Anything the YAML leaves unset falls through to the eval CLI args."""
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "sglang:\n"
            "  - name: default\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "        num_gpus_per_engine: 1\n"
            "  - name: eval\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 2\n"
        )
        args = make_args(
            sglang_config=str(cfg_path),
            rollout_num_gpus=8,
            eval_num_gpus=2,
            eval_num_gpus_per_engine=2,
            eval_sglang_mem_fraction_static=0.95,
        )
        config = _resolve_sglang_config(args)

        [eval_model] = [m for m in config.models if m.name == "eval"]
        # Auto-inference would give True here and put the fleet in the broadcast group.
        assert eval_model.update_weights is False
        [group] = eval_model.server_groups
        assert group.num_gpus_per_engine == 2
        assert group.overrides["mem_fraction_static"] == 0.95

    def test_yaml_group_overrides_win_over_eval_cli(self, tmp_path):
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "sglang:\n"
            "  - name: default\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "  - name: eval\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 2\n"
            "        num_gpus_per_engine: 1\n"
            "        overrides:\n"
            "          mem_fraction_static: 0.5\n"
        )
        args = make_args(
            sglang_config=str(cfg_path), rollout_num_gpus=8, eval_num_gpus=2, eval_sglang_mem_fraction_static=0.95
        )
        config = _resolve_sglang_config(args)

        [eval_model] = [m for m in config.models if m.name == "eval"]
        assert eval_model.server_groups[0].overrides["mem_fraction_static"] == 0.5

    async def test_probe_and_mark_dead(self, monkeypatch):
        """recover() only restarts engines already marked stopped, so something has to mark them."""
        import miles.ray.rollout.rollout_server as rollout_server_mod

        monkeypatch.setattr(rollout_server_mod.ray, "kill", lambda handle: None)

        class _Engine:
            def __init__(self, alive):
                self.is_allocated, self._alive = True, alive

            @property
            def actor_handle(self):
                async def probe():
                    if not self._alive:
                        raise RuntimeError("actor died")

                return SimpleNamespace(get_weight_version=SimpleNamespace(remote=probe))

            def mark_stopped(self):
                self.is_allocated = False

        alive, dead = _Engine(True), _Engine(False)
        srv = RolloutServer(server_groups=[SimpleNamespace(all_engines=[alive, dead])])

        await srv.probe_and_mark_dead()

        assert alive.is_allocated and not dead.is_allocated

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
            cell._mark_allocated_uninitialized()
            cell._mark_addressing(AddrInfo(server_url=f"http://10.0.0.{index + 1}:30000"))
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

        async def _fake_router_ready(*args, **kwargs):
            return HostAndPort(host="127.0.0.1", port=30000)

        monkeypatch.setattr(rollout_server, "wait_router_ready", _fake_router_ready)
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
        return list(asyncio.run(start_rollout_servers(args))["default"].server_cells.values())

    def test_a_single_node_engine_becomes_its_own_cell(self, stub_engine_startup, tmp_path):
        """With one gpu per engine on 8-gpu nodes, every engine is a one-engine cell."""
        cells = self._cells_for(tmp_path, num_gpus=8, num_gpus_per_engine=1)
        assert [cell.cell_index for cell in cells] == list(range(8))

    def test_a_multi_node_engine_chunks_its_node_ranks_into_one_cell(self, stub_engine_startup, tmp_path):
        """With 16 gpus per engine on 8-gpu nodes, the 32 gpus collapse into two cells."""
        cells = self._cells_for(tmp_path, num_gpus=32, num_gpus_per_engine=16)
        assert [cell.cell_index for cell in cells] == [0, 1]

    def test_a_trailing_partial_multi_node_engine_is_rejected(self, stub_engine_startup, tmp_path):
        """24 gpus do not divide into whole 2-node engines, so startup must fail fast."""
        with pytest.raises(AssertionError, match="whole number of"):
            self._cells_for(tmp_path, num_gpus=24, num_gpus_per_engine=16)

    def test_cells_carry_contiguous_gpu_offsets(self, stub_engine_startup, tmp_path):
        """Each multi-node cell's gpu span starts where the previous one ended."""
        cells = self._cells_for(tmp_path, num_gpus=32, num_gpus_per_engine=16)
        assert [cell.gpu_offset for cell in cells] == [0, 16]
