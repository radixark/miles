from argparse import Namespace

import pytest
import yaml
from tests.fast.fixtures.args_fixtures import parser_defaults
from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config

from miles.backends.megatron_utils.megatron_config import resolve_megatron_config


def _write_yaml(data: dict, tmp_path) -> str:
    path = tmp_path / "megatron.yaml"
    path.write_text(yaml.dump(data))
    return str(path)


def _make_args(megatron_config: str | None = None, **overrides) -> Namespace:
    defaults = dict(megatron_config=megatron_config)
    defaults.update(overrides)
    return Namespace(**{**parser_defaults(), **defaults})


class TestResolveMegatronConfig:
    def test_a_run_without_the_flag_synthesizes_a_plain_actor_trainer(self):
        """Legacy single policy runs must keep working, with no model id anywhere downstream."""
        config = resolve_megatron_config(_make_args())

        assert [(t.trainer_id, t.role, t.model_id, t.overrides) for t in config.trainers] == [
            ("actor", "actor", None, {})
        ]
        assert config.model_ids == []
        assert not config.is_multi_policy

    def test_the_legacy_megatron_key_is_still_accepted(self, tmp_path):
        """Configs written against the first name of the field must keep resolving."""
        path = _write_yaml({"megatron": [{"model_id": "a"}, {"model_id": "b"}]}, tmp_path)

        assert resolve_megatron_config(_make_args(path)).model_ids == ["a", "b"]

    def test_the_yaml_model_ids_become_the_trainer_model_ids(self, tmp_path):
        """The `model_id` field is the source of truth for trainer_model_id and spec names."""
        path = _write_yaml({"trainers": [{"model_id": "a", "overrides": {"lr": 1e-5}}, {"model_id": "b"}]}, tmp_path)

        config = resolve_megatron_config(_make_args(path))

        assert config.model_ids == ["a", "b"]
        assert config.leader_model_id == "a"
        assert config.is_multi_policy

    def test_the_first_model_is_the_leader_policy(self, tmp_path):
        """The leader owns the global checkpoint index, so its identity must be positional and stable."""
        path = _write_yaml({"trainers": [{"model_id": "second"}, {"model_id": "first"}]}, tmp_path)

        assert resolve_megatron_config(_make_args(path)).leader_model_id == "second"

    def test_an_inline_base64_payload_is_accepted(self, tmp_path):
        """Launchers that cannot ship a file still need to pass the config."""
        config = resolve_megatron_config(_make_args(encode_megatron_config("solo")))

        assert config.model_ids == ["solo"]

    def test_a_trainer_id_defaults_to_the_model_id_and_the_role(self, tmp_path):
        """The trainer id addresses a pool, so its default must stay the name every deployment already uses."""
        path = _write_yaml({"trainers": [{"model_id": "a"}, {"model_id": "b"}]}, tmp_path)

        config = resolve_megatron_config(_make_args(path))

        assert [trainer.trainer_id for trainer in config.trainers] == ["a-actor", "b-actor"]
        assert [trainer.role for trainer in config.trainers] == ["actor", "actor"]

    def test_an_explicit_trainer_id_wins_over_the_derived_one(self, tmp_path):
        """A deployment that already named its pools must be able to keep those names."""
        path = _write_yaml({"trainers": [{"model_id": "a", "trainer_id": "legacy-actor"}]}, tmp_path)

        assert resolve_megatron_config(_make_args(path)).trainers[0].trainer_id == "legacy-actor"

    def test_several_entries_of_one_model_id_are_not_a_multi_policy_run(self, tmp_path):
        """An actor and a critic of one policy share its id, and one policy is not several policies."""
        path = _write_yaml({"trainers": [{"model_id": "a"}]}, tmp_path)

        config = resolve_megatron_config(_make_args(path, use_critic=True))

        assert [trainer.trainer_id for trainer in config.trainers] == ["a-actor", "a-critic"]
        assert config.model_ids == ["a"]
        assert config.leader_model_id == "a"
        assert not config.is_multi_policy

    def test_an_unknown_yaml_key_is_refused(self, tmp_path):
        """A strict model turns a typo into a startup error instead of a silently ignored setting."""
        path = _write_yaml({"trainers": [{"model_id": "a", "override": {"lr": 1e-5}}]}, tmp_path)

        with pytest.raises(Exception, match="override"):
            resolve_megatron_config(_make_args(path))

    def test_getting_an_unknown_model_id_fails_loudly(self, tmp_path):
        """Callers routing by model id must not silently fall back to another policy."""
        path = _write_yaml({"trainers": [{"model_id": "a"}]}, tmp_path)

        with pytest.raises(KeyError, match="Unknown trainer model id"):
            resolve_megatron_config(_make_args(path)).get("b")

    def test_a_config_declaring_no_trainer_is_refused(self, tmp_path):
        """An empty list would resolve to a run with nothing to train, and fail much later and less clearly."""
        path = _write_yaml({"trainers": []}, tmp_path)

        with pytest.raises(AssertionError, match="must declare at least one trainer"):
            resolve_megatron_config(_make_args(path))

    def test_getting_a_model_id_answers_its_first_trainer(self, tmp_path):
        """Callers ask by model id and expect the actor: the critic of that policy is addressed by role."""
        path = _write_yaml({"trainers": [{"model_id": "a"}]}, tmp_path)

        config = resolve_megatron_config(_make_args(path, use_critic=True))

        assert [trainer.role for trainer in config.trainers] == ["actor", "critic"]
        assert config.get("a").role == "actor"

    def test_a_run_without_the_flag_has_no_leader_model_id(self):
        """A single policy run has no leader to index the trainers by, and must answer None rather than invent one."""
        assert resolve_megatron_config(_make_args()).leader_model_id is None
