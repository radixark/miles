from __future__ import annotations

from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_dataclass_group

from miles.ray.rollout.rollout_server import (
    RolloutServer,
    _compute_megatron_num_gpus,
    _compute_rollout_offset,
    _resolve_sglang_config,
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
