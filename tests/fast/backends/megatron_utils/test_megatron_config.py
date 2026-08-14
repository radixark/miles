import sys
from argparse import Namespace
from types import SimpleNamespace

import pydantic
import pytest
import yaml
from tests.fast.fixtures.args_fixtures import parser_defaults
from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config

from miles.backends.megatron_utils import megatron_config as megatron_config_module
from miles.backends.megatron_utils.megatron_config import (
    _has_megatron_checkpoint,
    _resolve_overrides,
    get_megatron_arg_parser,
    resolve_args_checkpoint_load,
    resolve_megatron_config,
)


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

    def test_duplicate_trainer_ids_are_refused(self, tmp_path):
        """Two entries sharing a trainer id would land in one controller and one engine pool."""
        path = _write_yaml({"trainers": [{"model_id": "a"}, {"model_id": "a"}]}, tmp_path)

        with pytest.raises(pydantic.ValidationError, match="trainer ids must be unique"):
            resolve_megatron_config(_make_args(path))

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

    def test_an_explicit_trainer_id_colliding_with_a_derived_one_is_refused(self, tmp_path):
        """Uniqueness has to hold across both spellings, or two trainers would share one engine pool."""
        path = _write_yaml({"trainers": [{"model_id": "a"}, {"model_id": "b", "trainer_id": "a-actor"}]}, tmp_path)

        with pytest.raises(pydantic.ValidationError, match="trainer ids must be unique"):
            resolve_megatron_config(_make_args(path))

    def test_a_trainer_id_that_is_not_a_dns_label_is_refused(self, tmp_path):
        """A trainer id is embedded in Kubernetes pool names, which must be lowercase DNS labels."""
        path = _write_yaml({"trainers": [{"model_id": "a", "trainer_id": "Legacy_Actor"}]}, tmp_path)

        with pytest.raises(pydantic.ValidationError, match="trainer ids"):
            resolve_megatron_config(_make_args(path))

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


class TestDerivedPerPolicyArgs:
    def test_a_model_id_that_escapes_its_checkpoint_directory_is_refused(self, tmp_path):
        """A model id is pasted into --save and --load, so it must stay one path component."""
        path = _write_yaml({"trainers": [{"model_id": "../evil"}, {"model_id": "b"}]}, tmp_path)

        with pytest.raises(pydantic.ValidationError, match="not usable as Kubernetes pool names"):
            resolve_megatron_config(_make_args(path))

    @pytest.mark.parametrize("model_id", ["policy_a", "PolicyA", "-policy", "policy-", "policy.a"])
    def test_a_model_id_that_is_not_a_dns_label_is_refused(self, tmp_path, model_id):
        """A model id is embedded in Kubernetes pool names, which must be lowercase DNS labels."""
        path = _write_yaml({"trainers": [{"model_id": model_id}, {"model_id": "b"}]}, tmp_path)

        with pytest.raises(pydantic.ValidationError, match="not usable as Kubernetes pool names"):
            resolve_megatron_config(_make_args(path))

    @pytest.mark.parametrize("model_id", ["default", "policy-a", "a1", "a-b-c"])
    def test_lowercase_dns_labels_are_accepted(self, tmp_path, model_id):
        """The ids the docs and examples use must survive validation."""
        path = _write_yaml({"trainers": [{"model_id": model_id}, {"model_id": "other"}]}, tmp_path)

        assert resolve_megatron_config(_make_args(path)).model_ids == [model_id, "other"]


class TestOverrideCoercion:
    def test_a_value_is_typed_by_the_declared_argument_not_by_the_yaml_scalar(self, tmp_path):
        """YAML reads `5e-7` as a string, so an untyped overlay would train against a string learning rate."""
        path = _write_yaml(
            {"trainers": [{"model_id": "a", "overrides": {"lr": "5e-7", "global_batch_size": "128"}}]}, tmp_path
        )

        overrides = resolve_megatron_config(_make_args(path)).get("a").overrides

        assert overrides == {"lr": 5e-7, "global_batch_size": 128}

    def test_a_boolean_argument_given_a_non_boolean_is_refused(self, tmp_path):
        """`sequence_parallel: yes-please` would otherwise become a truthy string."""
        path = _write_yaml(
            {"trainers": [{"model_id": "a", "overrides": {"sequence_parallel": "yes-please"}}]}, tmp_path
        )

        with pytest.raises(AssertionError, match="not a boolean"):
            resolve_megatron_config(_make_args(path))

    def test_an_override_without_a_value_is_refused(self, tmp_path):
        """A key written with an empty YAML value reads as None, which no argument can be set to."""
        path = _write_yaml({"trainers": [{"model_id": "a", "overrides": {"eps_clip_high": None}}]}, tmp_path)

        with pytest.raises(AssertionError, match="no value"):
            resolve_megatron_config(_make_args(path))

    def test_an_argument_outside_the_per_policy_whitelist_is_refused(self, tmp_path):
        """Rhythm arguments are read from the base command line, so accepting them here would do nothing."""
        path = _write_yaml({"trainers": [{"model_id": "a", "overrides": {"num_rollout": 3}}]}, tmp_path)

        with pytest.raises(AssertionError, match="num_rollout"):
            resolve_megatron_config(_make_args(path))


class TestResolveOverrides:
    def test_an_empty_override_map_never_builds_the_parser(self, monkeypatch):
        """Building the parser imports and runs megatron's whole argument stack, per trainer that overrides nothing."""
        monkeypatch.setattr(
            megatron_config_module,
            "get_megatron_arg_parser",
            lambda: pytest.fail("the parser was built for a trainer that overrides nothing"),
        )

        assert _resolve_overrides({}, model_id="a") == {}


class TestGetMegatronArgParser:
    def test_a_parser_that_never_reaches_the_provider_fails_loudly(self, monkeypatch):
        """Silently answering an empty parser would type every override as a string."""
        monkeypatch.setitem(
            sys.modules,
            "miles.backends.megatron_utils.arguments",
            SimpleNamespace(parse_args=lambda extra_args_provider: Namespace()),
        )

        with pytest.raises(AssertionError, match="returned without calling the extra args provider"):
            get_megatron_arg_parser()


def _make_checkpoint_args(tmp_path, **overrides) -> Namespace:
    defaults = dict(
        megatron_to_hf_mode="core",
        load=str(tmp_path / "save"),
        ref_load=str(tmp_path / "ref"),
        hf_checkpoint=str(tmp_path / "hf"),
        ref_ckpt_step=None,
        ckpt_step=None,
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        start_rollout_id=None,
    )
    defaults.update(overrides)
    return Namespace(**{**parser_defaults(), **defaults})


def _write_megatron_checkpoint(tmp_path) -> str:
    """A directory megatron recognizes: it exists and carries the iteration tracker file."""
    path = tmp_path / "save"
    path.mkdir()
    (path / "latest_checkpointed_iteration.txt").write_text("10")
    return str(path)


class TestResolveArgsCheckpointLoad:
    def test_a_fresh_bridge_run_falls_back_to_the_reference_weights(self, tmp_path):
        """--load points at a directory the first run has not written yet, which load_checkpoint asserts on."""
        args = _make_checkpoint_args(tmp_path, megatron_to_hf_mode="bridge")

        resolve_args_checkpoint_load(args)

        assert args.load == str(tmp_path / "ref")
        assert args.start_rollout_id == 0

    def test_a_bridge_run_without_reference_weights_falls_back_to_the_hf_checkpoint(self, tmp_path):
        """Without --ref-load the HF weights are the only thing the bridge can start from."""
        args = _make_checkpoint_args(tmp_path, megatron_to_hf_mode="bridge", ref_load=None)

        resolve_args_checkpoint_load(args)

        assert args.load == str(tmp_path / "hf")

    def test_a_bridge_run_with_a_real_checkpoint_resumes_from_it(self, tmp_path):
        """A bridge resume keeps --load, and start_rollout_id staying None is what says 'resume'."""
        load = _write_megatron_checkpoint(tmp_path)
        args = _make_checkpoint_args(tmp_path, megatron_to_hf_mode="bridge", load=load)

        resolve_args_checkpoint_load(args)

        assert args.load == load
        assert args.start_rollout_id is None

    def test_a_fresh_core_run_finetunes_from_the_reference_weights(self, tmp_path):
        """Loading an optimizer state and rng that were never written aborts the very first step."""
        args = _make_checkpoint_args(tmp_path)

        resolve_args_checkpoint_load(args)

        assert (args.no_load_optim, args.no_load_rng, args.finetune) == (True, True, True)
        assert args.load == str(tmp_path / "ref")
        assert args.start_rollout_id == 0

    def test_a_fresh_core_run_takes_the_step_of_the_reference_checkpoint(self, tmp_path):
        """--ref-ckpt-step names which iteration of the reference weights to read."""
        args = _make_checkpoint_args(tmp_path, ref_ckpt_step=7)

        resolve_args_checkpoint_load(args)

        assert args.ckpt_step == 7

    def test_a_core_run_with_a_real_checkpoint_is_left_untouched(self, tmp_path):
        """A resume must keep its optimizer state, and start_rollout_id staying None is what says 'resume'."""
        load = _write_megatron_checkpoint(tmp_path)
        args = _make_checkpoint_args(tmp_path, load=load, ref_ckpt_step=7)

        resolve_args_checkpoint_load(args)

        assert args.load == load
        assert (args.no_load_optim, args.no_load_rng, args.finetune) == (False, False, False)
        assert (args.ckpt_step, args.start_rollout_id) == (None, None)


class TestHasMegatronCheckpoint:
    def test_a_directory_holding_the_tracker_file_is_a_checkpoint(self, tmp_path):
        """This is the one shape both branches treat as a resume."""
        assert _has_megatron_checkpoint(_write_megatron_checkpoint(tmp_path)) is True

    def test_a_directory_without_the_tracker_file_is_not_a_checkpoint(self, tmp_path):
        """A --save directory exists from the moment the run starts, long before it holds a checkpoint."""
        (tmp_path / "save").mkdir()

        assert _has_megatron_checkpoint(str(tmp_path / "save")) is False

    def test_a_missing_directory_is_not_a_checkpoint(self, tmp_path):
        """The first run of a job passes a --load path nothing has created yet."""
        assert _has_megatron_checkpoint(str(tmp_path / "nope")) is False

    def test_no_load_directory_at_all_is_not_a_checkpoint(self):
        """--load is optional, and None must not reach os.path.exists."""
        assert _has_megatron_checkpoint(None) is False
