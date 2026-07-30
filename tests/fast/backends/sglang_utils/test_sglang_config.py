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


class TestPrefillNumServersPath:
    def test_prefill_consuming_all_gpus_is_rejected(self):
        """prefill_num_servers leaving no decode gpus fails loudly."""
        args = _make_args(rollout_num_gpus=4, prefill_num_servers=4, rollout_num_gpus_per_engine=1)
        with pytest.raises(AssertionError, match="No decode GPUs"):
            resolve_sglang_config(args)
