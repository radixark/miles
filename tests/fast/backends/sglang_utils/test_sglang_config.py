from __future__ import annotations

from argparse import Namespace

import pytest

from miles.backends.sglang_utils.sglang_config import (
    ServerGroupConfig,
    _compute_megatron_num_gpus,
    _compute_rollout_offset,
    resolve_sglang_config,
)


def _make_args(**overrides) -> Namespace:
    defaults = dict(
        sglang_config=None,
        prefill_num_servers=None,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=1,
        num_gpus_per_node=8,
        eval_num_gpus=0,
        eval_num_gpus_per_engine=1,
        hf_checkpoint="/ckpt/actor",
        offload_rollout=False,
        debug_train_only=False,
        debug_rollout_only=False,
        colocate=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        critic_num_nodes=0,
        critic_num_gpus_per_node=0,
        use_critic=False,
        critic_train_only=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _resolve_yaml(tmp_path, yaml_text: str, **args_overrides):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml_text)
    return resolve_sglang_config(_make_args(sglang_config=str(cfg_path), **args_overrides))


class TestNumGpusPerEnginePrecedence:
    def test_group_level_value_wins_over_model_and_args(self, tmp_path):
        """A group's own num_gpus_per_engine beats the model-level and args-level defaults."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: actor\n"
            "    num_gpus_per_engine: 4\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "        num_gpus_per_engine: 2\n",
            rollout_num_gpus=8,
            rollout_num_gpus_per_engine=1,
        )
        assert cfg.models[0].server_groups[0].num_gpus_per_engine == 2

    def test_model_then_global_num_gpus_per_engine_defaults_are_resolved(self, tmp_path):
        """A group without its own value takes the model-level value, or the args-level one when the model is silent."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: actor\n"
            "    num_gpus_per_engine: 4\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "  - name: ref\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 4\n",
            rollout_num_gpus=12,
            rollout_num_gpus_per_engine=2,
        )
        assert cfg.models[0].server_groups[0].num_gpus_per_engine == 4
        assert cfg.models[1].server_groups[0].num_gpus_per_engine == 2

    def test_a_group_asking_for_zero_gpus_per_engine_is_rejected_not_defaulted(self, tmp_path):
        """Falling back on any falsy value hid the typo and silently started engines with the wrong tp topology."""
        with pytest.raises(ValueError, match="greater than 0"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: actor\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 8\n"
                "        num_gpus_per_engine: 0\n",
                rollout_num_gpus=8,
                rollout_num_gpus_per_engine=1,
            )

    def test_a_model_asking_for_zero_gpus_per_engine_is_rejected_not_defaulted(self, tmp_path):
        """The model-level default took the same falsy fallback, so its groups silently inherited the args value."""
        with pytest.raises(ValueError, match="greater than 0"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: actor\n"
                "    num_gpus_per_engine: 0\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 8\n",
                rollout_num_gpus=8,
                rollout_num_gpus_per_engine=1,
            )


class TestModelPathConsistency:
    def test_groups_for_one_model_cannot_resolve_to_different_model_paths(self, tmp_path):
        """One model serving two different checkpoints would silently mis-route, so it fails at resolve time."""
        with pytest.raises(AssertionError, match="different model_path values"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: actor\n"
                "    model_path: /model/a\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 4\n"
                "      - worker_type: regular\n"
                "        num_gpus: 4\n"
                "        overrides:\n"
                "          model_path: /model/b\n",
                rollout_num_gpus=8,
            )


class TestResolvedServerGroupValidation:
    def test_a_resolved_group_with_zero_gpus_is_rejected(self):
        """A group reserving no GPUs cannot host an engine and must fail construction."""
        with pytest.raises(ValueError, match="greater than 0"):
            ServerGroupConfig(
                worker_type="regular",
                num_gpus=0,
                num_gpus_per_engine=2,
                gpu_offset=0,
                engine_offset=0,
                needs_offload=False,
            )

    def test_a_resolved_group_with_non_positive_gpus_per_engine_is_rejected(self):
        """A non-positive engine width would make the engine count division meaningless."""
        with pytest.raises(ValueError, match="greater than 0"):
            ServerGroupConfig(
                worker_type="regular",
                num_gpus=8,
                num_gpus_per_engine=-1,
                gpu_offset=0,
                engine_offset=0,
                needs_offload=False,
            )


class TestNumServerCells:
    def test_non_placeholder_engine_cells_are_counted(self, tmp_path):
        """The server cell count includes every engine except placeholder reservations."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: actor\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "        num_gpus_per_engine: 4\n"
            "      - worker_type: prefill\n"
            "        num_gpus: 4\n"
            "        num_gpus_per_engine: 2\n"
            "      - worker_type: placeholder\n"
            "        num_gpus: 4\n"
            "        num_gpus_per_engine: 1\n",
            rollout_num_gpus=16,
        )

        assert cfg.models[0].num_server_cells == 4


class TestOverridesResolution:
    def test_an_explicit_model_path_override_is_not_replaced(self, tmp_path):
        """A group's own model_path override is parsed verbatim."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: actor\n"
            "    model_path: /model/level/path\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "        overrides:\n"
            "          model_path: /group/level/path\n",
            rollout_num_gpus=8,
        )
        assert cfg.models[0].server_groups[0].overrides["model_path"] == "/group/level/path"

    def test_unrelated_override_keys_pass_through(self, tmp_path):
        """Arbitrary ServerArgs overrides are parsed verbatim."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: actor\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "        overrides:\n"
            "          mem_fraction_static: 0.5\n",
            rollout_num_gpus=8,
        )
        overrides = cfg.models[0].server_groups[0].overrides
        assert overrides["mem_fraction_static"] == 0.5


class TestYamlShapeValidation:
    def test_engine_groups_is_accepted_as_an_alias_for_server_groups(self, tmp_path):
        """The documented engine_groups spelling keeps parsing."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n  - name: actor\n    engine_groups:\n      - worker_type: regular\n        num_gpus: 8\n",
            rollout_num_gpus=8,
        )
        assert cfg.models[0].server_groups[0].num_gpus == 8

    def test_a_yaml_without_the_sglang_key_is_rejected(self, tmp_path):
        """A config missing the top-level sglang key fails loudly."""
        with pytest.raises((AssertionError, ValueError), match="sglang|models"):
            _resolve_yaml(tmp_path, "other_key:\n  - name: actor\n", rollout_num_gpus=8)

    def test_an_unknown_model_key_is_rejected(self, tmp_path):
        """Typos at the model level fail parsing instead of being silently dropped."""
        with pytest.raises(ValueError, match="typo_key"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: actor\n"
                "    typo_key: 1\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 8\n",
                rollout_num_gpus=8,
            )

    def test_an_unknown_server_group_key_is_rejected(self, tmp_path):
        """Typos at the server-group level fail parsing instead of dropping an intended SGLang override."""
        with pytest.raises(ValueError, match="typo_key"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: actor\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 8\n"
                "        typo_key: 1\n",
                rollout_num_gpus=8,
            )

    def test_giving_both_group_spellings_is_rejected(self, tmp_path):
        """server_groups plus engine_groups on one model is ambiguous and fails parsing."""
        with pytest.raises(ValueError, match="engine_groups"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: actor\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 8\n"
                "    engine_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 8\n",
                rollout_num_gpus=8,
            )


class TestPrefillNumServersPath:
    def test_prefill_num_servers_counts_engines_not_gpus(self):
        """prefill_num_servers is a server count, so its GPU span scales with the engine width."""
        cfg = resolve_sglang_config(
            _make_args(rollout_num_gpus=16, prefill_num_servers=3, rollout_num_gpus_per_engine=2)
        )
        groups = cfg.models[0].server_groups
        assert [(group.worker_type, group.num_gpus) for group in groups] == [("prefill", 6), ("decode", 10)]

    def test_prefill_consuming_all_gpus_is_rejected(self):
        """prefill_num_servers leaving no decode gpus fails loudly."""
        args = _make_args(rollout_num_gpus=4, prefill_num_servers=4, rollout_num_gpus_per_engine=1)
        with pytest.raises(AssertionError, match="No decode GPUs"):
            resolve_sglang_config(args)


class TestGpuOffset:
    def test_gpu_offsets_accumulate_across_groups_and_models_including_placeholders(self, tmp_path):
        """Each group's gpu_offset equals the num_gpus sum of all preceding groups, counting placeholders."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: actor\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 4\n"
            "      - worker_type: placeholder\n"
            "        num_gpus: 4\n"
            "  - name: ref\n"
            "    update_weights: false\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n",
            rollout_num_gpus=16,
        )
        assert [group.gpu_offset for group in cfg.models[0].server_groups] == [0, 4]
        assert cfg.models[1].server_groups[0].gpu_offset == 8


class TestEngineOffset:
    def test_engine_offsets_count_the_workers_of_every_preceding_group(self, tmp_path):
        """Groups of different engine widths contribute different worker counts, so a gpu offset alone cannot number them."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: actor\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "        num_gpus_per_engine: 2\n"
            "      - worker_type: placeholder\n"
            "        num_gpus: 4\n"
            "        num_gpus_per_engine: 4\n"
            "  - name: ref\n"
            "    update_weights: false\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 16\n"
            "        num_gpus_per_engine: 8\n",
            rollout_num_gpus=28,
            num_gpus_per_node=4,
        )
        assert [group.engine_offset for group in cfg.models[0].server_groups] == [0, 4]
        assert cfg.models[1].server_groups[0].engine_offset == 5

    def test_a_group_whose_engine_spans_nodes_contributes_one_worker_per_node(self, tmp_path):
        """A cross-node engine is launched by one actor per node, so it consumes that many numbers, not one."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: actor\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 16\n"
            "        num_gpus_per_engine: 8\n"
            "      - worker_type: regular\n"
            "        num_gpus: 4\n"
            "        num_gpus_per_engine: 2\n",
            rollout_num_gpus=20,
            num_gpus_per_node=4,
        )
        assert [group.engine_offset for group in cfg.models[0].server_groups] == [0, 4]

    def test_prefill_and_decode_groups_are_numbered_in_that_order(self):
        """The legacy --prefill-num-servers layout has no YAML to carry offsets, so the cursor must number it too."""
        cfg = resolve_sglang_config(
            _make_args(rollout_num_gpus=16, prefill_num_servers=3, rollout_num_gpus_per_engine=2)
        )

        assert [group.engine_offset for group in cfg.models[0].server_groups] == [0, 3]

    def test_the_generated_eval_model_is_numbered_after_every_rollout_engine(self):
        """The eval fleet is appended without YAML, and reusing the rollout numbers would clone their RNG streams."""
        cfg = resolve_sglang_config(
            _make_args(
                rollout_num_gpus=8,
                rollout_num_gpus_per_engine=2,
                eval_num_gpus=4,
                eval_num_gpus_per_engine=2,
            )
        )

        assert cfg.models[0].server_groups[0].engine_offset == 0
        assert cfg.models[1].server_groups[0].engine_offset == 4


class TestNeedsOffload:
    def test_no_offload_flag_means_no_group_needs_offload(self, tmp_path):
        """With offload_rollout off, no group offloads and no memory-saver override is injected."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n  - name: actor\n    server_groups:\n      - worker_type: regular\n        num_gpus: 8\n",
            rollout_num_gpus=8,
        )
        group = cfg.models[0].server_groups[0]
        assert group.needs_offload is False
        assert "enable_memory_saver" not in group.overrides

    def test_groups_overlapping_megatron_offload_and_the_rest_disable_memory_saver(self, tmp_path):
        """Only groups starting inside the megatron gpu range offload; later ones get enable_memory_saver=False."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: actor\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n",
            rollout_num_gpus=16,
            offload_rollout=True,
            colocate=True,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
        )
        first, second = cfg.models[0].server_groups
        assert first.needs_offload is True
        assert "enable_memory_saver" not in first.overrides
        assert second.needs_offload is False
        assert second.overrides["enable_memory_saver"] is False

    def test_the_gpu_cursor_accumulates_across_models(self, tmp_path):
        """A later model's group starts where the previous model's gpus ended."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: actor\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "  - name: ref\n"
            "    update_weights: false\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n",
            rollout_num_gpus=16,
            offload_rollout=True,
            colocate=True,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
        )
        assert cfg.models[0].server_groups[0].needs_offload is True
        assert cfg.models[1].server_groups[0].needs_offload is False

    def test_a_disaggregated_rollout_group_past_the_megatron_gpus_does_not_offload(self, tmp_path):
        """Without colocate the rollout pg offset pushes the group past the megatron gpus, so it disables memory saver."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n  - name: actor\n    server_groups:\n      - worker_type: regular\n        num_gpus: 8\n",
            rollout_num_gpus=8,
            offload_rollout=True,
            colocate=False,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
        )
        group = cfg.models[0].server_groups[0]
        assert group.needs_offload is False
        assert group.overrides["enable_memory_saver"] is False

    def test_an_explicit_memory_saver_override_wins(self, tmp_path):
        """A user-provided enable_memory_saver survives the conditional injection."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: actor\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "        overrides:\n"
            "          enable_memory_saver: true\n",
            rollout_num_gpus=8,
            offload_rollout=True,
            colocate=True,
            actor_num_nodes=0,
            actor_num_gpus_per_node=0,
        )
        group = cfg.models[0].server_groups[0]
        assert group.needs_offload is False
        assert group.overrides["enable_memory_saver"] is True


class TestYamlEvalModel:
    def test_a_yaml_without_an_eval_model_is_rejected_when_eval_gpus_are_requested(self, tmp_path):
        """--eval-num-gpus with no eval model in the YAML would silently run evals on the rollout fleet."""
        with pytest.raises(AssertionError, match="exactly one model named 'eval'"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: default\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 10\n",
                rollout_num_gpus=8,
                eval_num_gpus=2,
            )

    def test_a_yaml_eval_model_sized_differently_from_eval_num_gpus_is_rejected(self, tmp_path):
        """A matching grand total is not enough; the eval model itself must own exactly --eval-num-gpus."""
        with pytest.raises(AssertionError, match="exactly one model named 'eval'"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: default\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 8\n"
                "  - name: eval\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 2\n",
                rollout_num_gpus=6,
                eval_num_gpus=4,
            )

    def test_yaml_eval_update_weights_is_not_overridden(self, tmp_path):
        """An explicit update_weights in the YAML survives the eval model's False default."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: default\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "  - name: eval\n"
            "    update_weights: true\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 2\n",
            rollout_num_gpus=8,
            eval_num_gpus=2,
        )
        [eval_model] = [model for model in cfg.models if model.name == "eval"]
        assert eval_model.update_weights is True

    def test_yaml_eval_group_num_gpus_per_engine_wins_over_eval_cli(self, tmp_path):
        """A group that states its own engine width keeps it instead of taking --eval-num-gpus-per-engine."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: default\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "  - name: eval\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 4\n"
            "        num_gpus_per_engine: 2\n",
            rollout_num_gpus=8,
            eval_num_gpus=4,
            eval_num_gpus_per_engine=4,
        )
        [eval_model] = [model for model in cfg.models if model.name == "eval"]
        assert eval_model.server_groups[0].num_gpus_per_engine == 2


class TestRolloutOffset:
    def test_debug_train_only_has_zero_rollout_placement_offset(self):
        """In train-only debug runs nothing is placed before the rollout bundles."""
        args = _make_args(debug_train_only=True, colocate=False, actor_num_nodes=2, actor_num_gpus_per_node=8)
        assert _compute_rollout_offset(args) == 0

    def test_debug_rollout_only_has_zero_rollout_placement_offset(self):
        """In rollout-only debug runs no megatron bundles are reserved ahead of the rollout ones."""
        args = _make_args(debug_rollout_only=True, colocate=False, actor_num_nodes=2, actor_num_gpus_per_node=8)
        assert _compute_rollout_offset(args) == 0


class TestMegatronNumGpus:
    def test_compute_megatron_num_gpus_for_critic_train_only(self):
        """With only the critic training, the megatron span is the critic's own gpus, not the actor's."""
        args = _make_args(
            critic_train_only=True,
            debug_rollout_only=False,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            critic_num_nodes=1,
            critic_num_gpus_per_node=4,
        )
        assert _compute_megatron_num_gpus(args) == 4


class TestHostPortOverrideRejection:
    def test_a_port_override_is_rejected_at_resolve_time(self, tmp_path):
        """Overriding the allocator-owned port must fail fast instead of desyncing engine and controller."""
        with pytest.raises(AssertionError, match="must not override host/port"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: actor\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 8\n"
                "        overrides:\n"
                "          port: 12345\n",
                rollout_num_gpus=8,
            )

    def test_a_host_override_is_rejected_at_resolve_time(self, tmp_path):
        """Overriding the allocator-owned host must fail fast as well."""
        with pytest.raises(AssertionError, match="must not override host/port"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: actor\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 8\n"
                "        overrides:\n"
                "          host: 10.0.0.1\n",
                rollout_num_gpus=8,
            )

    def test_a_gated_launch_port_override_is_rejected_at_resolve_time(self, tmp_path):
        """The launch gate port belongs to the allocator; an override would make the engine wait on the wrong socket."""
        with pytest.raises(AssertionError, match="must not override host/port"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: actor\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 8\n"
                "        overrides:\n"
                "          gated_launch_port: 13007\n",
                rollout_num_gpus=8,
            )

    def test_a_gated_launch_port_override_in_a_later_group_is_rejected(self, tmp_path):
        """The check runs per group, so a gate port hidden behind valid groups and valid keys still fails fast."""
        with pytest.raises(AssertionError, match="must not override host/port"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: actor\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 4\n"
                "  - name: ref\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 2\n"
                "      - worker_type: regular\n"
                "        num_gpus: 2\n"
                "        overrides:\n"
                "          mem_fraction_static: 0.5\n"
                "          gated_launch_port: 13007\n",
                rollout_num_gpus=8,
            )

    def test_a_gated_launch_port_reaching_the_eval_fleet_from_the_cli_is_rejected(self, tmp_path):
        """An --eval-sglang-* gate port becomes an eval override, and that channel must be refused too."""
        with pytest.raises(AssertionError, match="must not override host/port"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: actor\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 8\n"
                "  - name: eval\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 2\n",
                rollout_num_gpus=8,
                eval_num_gpus=2,
                eval_sglang_gated_launch_port=13007,
            )
