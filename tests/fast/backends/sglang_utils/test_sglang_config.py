from __future__ import annotations

from argparse import Namespace

import pytest

from miles.backends.sglang_utils.sglang_config import resolve_sglang_config


def _make_args(**overrides) -> Namespace:
    defaults = dict(
        sglang_config=None,
        prefill_num_servers=None,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=1,
        eval_num_gpus=0,
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
