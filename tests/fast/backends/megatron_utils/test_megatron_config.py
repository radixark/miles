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
    PER_POLICY_ARGS,
    _compute_trainer_checkpoint_dir,
    _has_megatron_checkpoint,
    _resolve_overrides,
    compute_trainer_args,
    get_megatron_arg_parser,
    resolve_args_checkpoint_load,
    resolve_megatron_config,
)
from miles.utils.external_utils.model_args_utils import load_model_args


def _write_yaml(data: dict, tmp_path) -> str:
    path = tmp_path / "megatron.yaml"
    path.write_text(yaml.dump(data))
    return str(path)


def _model_args(args: Namespace, *, model_id: str) -> Namespace:
    return compute_trainer_args(args, resolve_megatron_config(args).get(model_id))


def _make_args(megatron_config: str | None = None, **overrides) -> Namespace:
    defaults = dict(
        megatron_config=megatron_config,
        lr=1e-6,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        hf_checkpoint="/models/base",
        global_batch_size=None,
        eps_clip_high=None,
        save=None,
        load=None,
        advantage_estimator="grpo",
        use_critic=False,
        num_steps_per_rollout=None,
        rollout_batch_size=8,
        n_samples_per_prompt=4,
        megatron_to_hf_mode="core",
        ref_load=None,
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        ref_ckpt_step=None,
        ckpt_step=None,
        start_rollout_id=None,
        optimizer="adam",
        use_distributed_optimizer=True,
        world_size=8,
        debug_disable_optimizer=False,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        kl_coef=0.1,
        use_opd=True,
        disable_param_buffers_cpu_backup=True,
        lr_warmup_iters=10,
        critic_load=None,
        critic_save=None,
        critic_lr=None,
        critic_lr_warmup_iters=None,
        fp16=False,
        seq_length=4096,
        vocab_size=None,
        padded_vocab_size=None,
        tokenizer_model="/models/base",
        tokenizer_type="HuggingFaceTokenizer",
        multi_lora_n_adapters=0,
    )
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


class TestComputeTrainerArgs:
    def test_each_policy_overlays_its_own_args_on_the_base_arguments(self, tmp_path):
        """Per-policy megatron args are the whole point of the flag; the base args stay untouched."""
        path = _write_yaml(
            {
                "megatron": [
                    {"model_id": "a", "overrides": {"lr": 5e-7, "tensor_model_parallel_size": 2}},
                    {"model_id": "b"},
                ]
            },
            tmp_path,
        )
        args = _make_args(path)

        model_a = _model_args(args, model_id="a")
        model_b = _model_args(args, model_id="b")

        assert (model_a.lr, model_a.tensor_model_parallel_size) == (5e-7, 2)
        assert (model_b.lr, model_b.tensor_model_parallel_size) == (1e-6, 1)
        assert (args.lr, args.tensor_model_parallel_size) == (1e-6, 1)

    def test_a_boolean_override_is_kept_as_a_boolean(self, tmp_path):
        """store_true arguments are booleans in the overlay, not the strings a command line would carry."""
        path = _write_yaml(
            {"trainers": [{"model_id": "a", "overrides": {"sequence_parallel": True}}, {"model_id": "b"}]},
            tmp_path,
        )

        assert _model_args(_make_args(path), model_id="a").sequence_parallel is True

    def test_an_unknown_argument_is_refused(self, tmp_path):
        """A per-policy typo would otherwise be dropped and the policy would train with base settings."""
        path = _write_yaml({"trainers": [{"model_id": "a", "overrides": {"no_such_flag": 3}}]}, tmp_path)

        with pytest.raises(AssertionError, match="no_such_flag"):
            resolve_megatron_config(_make_args(path))

    def test_a_whitelisted_argument_this_run_does_not_declare_is_refused(self, tmp_path):
        """A whitelist entry is not a promise that every backend's parser declares it."""
        path = _write_yaml(
            {"trainers": [{"model_id": "a", "overrides": {"min_lr": 1e-8}}, {"model_id": "b"}]}, tmp_path
        )

        with pytest.raises(AssertionError, match="does not know"):
            _model_args(_make_args(path), model_id="a")

    def test_an_overlaid_advantage_estimator_does_not_fabricate_a_critic(self, tmp_path):
        """use_critic is settled from the command line before the overlay, so an override must not flip it."""
        path = _write_yaml(
            {"trainers": [{"model_id": "a", "overrides": {"advantage_estimator": "ppo"}}, {"model_id": "b"}]}, tmp_path
        )
        args = _make_args(path, advantage_estimator="grpo")

        assert _model_args(args, model_id="a").use_critic is False


class TestTrainerCheckpointDirs:
    def test_a_multi_policy_run_gives_every_trainer_its_own_checkpoint_dir(self, tmp_path):
        """A shared --save makes two policies write the same iter_* directory and overwrite each other."""
        old = tmp_path / "old"
        for trainer_id in ("a-actor", "b-actor"):
            trainer_dir = old / "trainers" / trainer_id
            trainer_dir.mkdir(parents=True)
            (trainer_dir / "latest_checkpointed_iteration.txt").write_text("7")
        path = _write_yaml({"trainers": [{"model_id": "a"}, {"model_id": "b"}]}, tmp_path)
        args = _make_args(path, save="/ckpt/run", load=str(old))

        model_a = _model_args(args, model_id="a")
        model_b = _model_args(args, model_id="b")

        assert (model_a.save, model_a.load) == ("/ckpt/run/trainers/a-actor", str(old / "trainers" / "a-actor"))
        assert (model_b.save, model_b.load) == ("/ckpt/run/trainers/b-actor", str(old / "trainers" / "b-actor"))

    def test_a_checkpoint_dir_is_keyed_by_the_trainer_id_and_not_by_the_model_id(self, tmp_path):
        """A trainer id is unique where a model id is not, so keying the directory by the model would collide."""
        path = _write_yaml({"trainers": [{"model_id": "a", "trainer_id": "a-second"}, {"model_id": "b"}]}, tmp_path)
        args = _make_args(path, save="/ckpt/run")

        saves = [compute_trainer_args(args, trainer).save for trainer in resolve_megatron_config(args).trainers]

        assert saves == ["/ckpt/run/trainers/a-second", "/ckpt/run/trainers/b-actor"]

    def test_an_explicitly_named_policy_gets_its_own_checkpoint_dir(self, tmp_path):
        """A split trainer must not share its checkpoint paths with another policy's release."""
        old = tmp_path / "old"
        trainer_dir = old / "trainers" / "a-actor"
        trainer_dir.mkdir(parents=True)
        (trainer_dir / "latest_checkpointed_iteration.txt").write_text("7")
        path = _write_yaml({"trainers": [{"model_id": "a"}]}, tmp_path)
        args = _make_args(path, save="/ckpt/run", load=str(old), save_hf="/ckpt/hf")

        model = _model_args(args, model_id="a")

        assert (model.save, model.load, model.save_hf) == (
            "/ckpt/run/trainers/a-actor",
            str(trainer_dir),
            "/ckpt/hf/trainers/a-actor",
        )

    def test_a_named_critic_derives_its_dir_from_the_critic_checkpoint(self, tmp_path):
        """A named critic must namespace its own checkpoint base rather than the policy's base."""
        old = tmp_path / "critic-old"
        trainer_dir = old / "trainers" / "alpha-critic"
        trainer_dir.mkdir(parents=True)
        (trainer_dir / "latest_checkpointed_iteration.txt").write_text("7")
        path = _write_yaml({"trainers": [{"model_id": "alpha"}]}, tmp_path)
        args = _make_args(
            path,
            use_critic=True,
            save="/ckpt/actor",
            load="/old/actor",
            critic_save="/ckpt/critic",
            critic_load=str(old),
        )
        [_, critic] = resolve_megatron_config(args).trainers

        model = compute_trainer_args(args, critic)

        assert (model.save, model.load) == ("/ckpt/critic/trainers/alpha-critic", str(trainer_dir))

    def test_an_unset_checkpoint_dir_stays_unset(self, tmp_path):
        """A run without --save must not grow a derived path out of None."""
        path = _write_yaml({"trainers": [{"model_id": "a"}, {"model_id": "b"}]}, tmp_path)

        assert _model_args(_make_args(path), model_id="a").save is None

    def test_the_derived_dir_is_the_trainer_id_under_a_trainers_directory(self):
        """The layout is a user visible contract: it is where a resume looks for a trainer's checkpoints."""
        assert (
            _compute_trainer_checkpoint_dir(base_dir="/ckpt/run", trainer_id="policy-b-actor")
            == "/ckpt/run/trainers/policy-b-actor"
        )

    def test_a_policy_cannot_name_its_own_checkpoint_directory(self, tmp_path):
        """The per trainer directory is derived from the base --load after the overlay, which would drop an override."""
        path = _write_yaml(
            {"trainers": [{"model_id": "a", "overrides": {"load": "/ckpt/a"}}, {"model_id": "b"}]}, tmp_path
        )

        with pytest.raises(AssertionError, match="sets 'load', which it may not override"):
            resolve_megatron_config(_make_args(path))


class TestPerPolicyCheckpointResolution:
    def test_a_fresh_policy_falls_back_to_the_reference_weights_without_the_policy_subdirectory(self, tmp_path):
        """The --ref-load fallback holds shared reference weights, not a per policy checkpoint tree."""
        path = _write_yaml({"trainers": [{"model_id": "a"}, {"model_id": "b"}]}, tmp_path)
        args = _make_args(path, save="/ckpt/run", load="/ckpt/run", ref_load="/models/ref")

        model_args = _model_args(args, model_id="a")

        assert model_args.save == "/ckpt/run/trainers/a-actor"
        assert model_args.load == "/models/ref"
        assert (model_args.finetune, model_args.start_rollout_id) == (True, 0)

    def test_a_policy_with_its_own_tracker_resumes_from_its_own_directory(self, tmp_path):
        """The tracker of a policy lives under its own subdirectory, so the root never looks resumable."""
        root = tmp_path / "run"
        trainer_dir = root / "trainers" / "a-actor"
        trainer_dir.mkdir(parents=True)
        (trainer_dir / "latest_checkpointed_iteration.txt").write_text("7")
        path = _write_yaml({"trainers": [{"model_id": "a"}, {"model_id": "b"}]}, tmp_path)
        args = _make_args(path, save=str(root), load=str(root), ref_load="/models/ref")

        model_args = _model_args(args, model_id="a")

        assert model_args.load == str(trainer_dir)
        assert (model_args.finetune, model_args.start_rollout_id) == (False, None)

    def test_a_fresh_bridge_policy_falls_back_to_its_own_hf_checkpoint(self, tmp_path):
        """In bridge mode a policy starts from its own hugging face checkpoint, not from another policy's."""
        path = _write_yaml(
            {
                "trainers": [
                    {"model_id": "a", "overrides": {"hf_checkpoint": "/models/a"}},
                    {"model_id": "b", "overrides": {"hf_checkpoint": "/models/b"}},
                ]
            },
            tmp_path,
        )
        args = _make_args(path, megatron_to_hf_mode="bridge", save="/ckpt/run", load="/ckpt/run")

        assert _model_args(args, model_id="a").load == "/models/a"
        assert _model_args(args, model_id="b").load == "/models/b"


class TestPerPolicyDerivedDefaults:
    def test_a_policy_checkpoint_override_repoints_the_tokenizer(self, tmp_path):
        """The tokenizer latched onto the base checkpoint at parse time, so a policy of its own needs its own."""
        path = _write_yaml(
            {"trainers": [{"model_id": "a", "overrides": {"hf_checkpoint": "/models/a"}}, {"model_id": "b"}]},
            tmp_path,
        )
        args = _make_args(path)

        assert _model_args(args, model_id="a").tokenizer_model == "/models/a"
        assert _model_args(args, model_id="b").tokenizer_model == "/models/base"

    def test_a_tokenizer_named_on_the_command_line_is_left_alone(self, tmp_path):
        """That tokenizer was chosen rather than derived, so no policy may re-point it at its own checkpoint."""
        path = _write_yaml(
            {"trainers": [{"model_id": "a", "overrides": {"hf_checkpoint": "/models/a"}}, {"model_id": "b"}]},
            tmp_path,
        )
        args = _make_args(path, tokenizer_model="/models/shared")

        assert _model_args(args, model_id="a").tokenizer_model == "/models/shared"


class TestMultiPolicyIds:
    def test_an_explicitly_named_policy_keeps_its_trainer_model_id(self, tmp_path):
        """A split trainer keeps its policy identity even when its release carries no neighbouring policy."""
        path = _write_yaml({"trainers": [{"model_id": "a"}]}, tmp_path)

        assert _model_args(_make_args(path), model_id="a").trainer_model_id == "a"

    def test_a_run_without_a_megatron_config_carries_no_trainer_model_id(self):
        """The unnamed legacy actor keeps using the unnamespaced single-policy key."""
        args = _make_args()
        [trainer] = resolve_megatron_config(args).trainers

        assert compute_trainer_args(args, trainer).trainer_model_id is None

    def test_each_policy_of_a_multi_policy_run_carries_its_own_id(self, tmp_path):
        """Metrics, routers and checkpoints are all namespaced by this value."""
        path = _write_yaml({"trainers": [{"model_id": "a"}, {"model_id": "b"}]}, tmp_path)

        assert _model_args(_make_args(path), model_id="b").trainer_model_id == "b"


class TestSynthesizedCriticTrainer:
    def test_arguments_that_do_not_carry_use_critic_yet_still_resolve(self):
        """use_critic is derived while the arguments are validated, and the config is resolved before that."""
        args = _make_args()
        del args.use_critic

        assert [trainer.role for trainer in resolve_megatron_config(args).trainers] == ["actor"]

    def test_a_run_without_the_flag_synthesizes_the_critic_beside_the_actor(self):
        """The critic used to be assembled in specs and in the worker; the config is now the only source."""
        config = resolve_megatron_config(_make_args(use_critic=True))

        assert [(t.trainer_id, t.role, t.model_id) for t in config.trainers] == [
            ("actor", "actor", None),
            ("critic", "critic", None),
        ]

    def test_training_several_policies_with_a_critic_is_refused(self, tmp_path):
        """A critic is synthesized beside one policy, so several policies plus a critic has no defined meaning."""
        path = _write_yaml({"trainers": [{"model_id": "a"}, {"model_id": "b"}]}, tmp_path)

        with pytest.raises(AssertionError, match="does not support --use-critic"):
            resolve_megatron_config(_make_args(path, use_critic=True))

    def test_a_config_that_declares_its_own_critic_is_refused(self, tmp_path):
        """The critic checkpoint, learning rate and neutralized knobs only reach the synthesized critic, so a
        declared one would run as a plain trainer under a critic's name."""
        path = _write_yaml(
            {
                "trainers": [
                    {"model_id": "a"},
                    {"model_id": "a", "role": "critic", "overrides": {"lr": 5e-7}},
                ]
            },
            tmp_path,
        )

        with pytest.raises(AssertionError, match="declares a critic for"):
            resolve_megatron_config(_make_args(path, use_critic=True))

    def test_the_critic_of_a_named_policy_inherits_its_id_and_its_overlay(self, tmp_path):
        """The critic trains the same policy, so it must be addressed by that policy and see its settings."""
        path = _write_yaml({"trainers": [{"model_id": "alpha", "overrides": {"eps_clip": 0.3}}]}, tmp_path)

        [_, critic] = resolve_megatron_config(_make_args(path, use_critic=True)).trainers

        assert (critic.trainer_id, critic.model_id, critic.role) == ("alpha-critic", "alpha", "critic")
        assert critic.overrides["eps_clip"] == 0.3

    def test_the_synthesized_critic_reproduces_the_legacy_worker_swap(self):
        """The worker no longer remaps critic_* onto the standard fields, so the overlay must do it."""
        args = _make_args(
            use_critic=True,
            save="/ckpt/run",
            load="/ckpt/run",
            critic_load="/ckpt/critic",
            critic_save="/ckpt/run_critic",
            critic_lr=2e-6,
            critic_lr_warmup_iters=3,
        )

        critic_args = compute_trainer_args(args, resolve_megatron_config(args).trainers[1])

        assert (critic_args.kl_coef, critic_args.use_opd, critic_args.disable_param_buffers_cpu_backup) == (
            0,
            False,
            False,
        )
        assert (critic_args.load, critic_args.save, critic_args.lr, critic_args.lr_warmup_iters) == (
            "/ckpt/critic",
            "/ckpt/run_critic",
            2e-6,
            3,
        )

    def test_the_actor_of_a_critic_run_keeps_its_own_checkpoint_and_schedule(self):
        """The two trainers share one command line, so a leaked critic override would retrain the actor."""
        args = _make_args(use_critic=True, save="/ckpt/run", load="/ckpt/run", critic_load="/ckpt/critic")

        actor_args = compute_trainer_args(args, resolve_megatron_config(args).trainers[0])

        assert (actor_args.load, actor_args.save, actor_args.lr, actor_args.kl_coef) == (
            "/ckpt/run",
            "/ckpt/run",
            1e-6,
            0.1,
        )

    def test_an_internally_synthesized_override_bypasses_the_user_whitelist(self):
        """kl_coef and load are not per-policy yaml arguments, yet the critic must still be able to set them."""
        args = _make_args(use_critic=True, critic_load="/ckpt/critic")

        overrides = set(resolve_megatron_config(args).trainers[1].overrides)

        assert {"kl_coef", "load"} <= overrides
        assert not {"kl_coef", "load"} & set(PER_POLICY_ARGS)

    def test_a_critic_without_its_own_schedule_inherits_the_unset_values(self, tmp_path):
        """--critic-lr is unset by default, and the critic takes it as it is: the policy's own lr does not apply."""
        path = _write_yaml({"trainers": [{"model_id": "alpha", "overrides": {"lr": 5e-7}}]}, tmp_path)
        args = _make_args(path, use_critic=True)

        critic_args = compute_trainer_args(args, resolve_megatron_config(args).trainers[1])

        assert (critic_args.lr, critic_args.lr_warmup_iters) == (None, None)

    def test_the_critic_overlay_names_exactly_the_fields_the_worker_used_to_swap(self):
        """A new critic_* argument that nobody wires in here would be read from the command line and ignored."""
        overrides = resolve_megatron_config(_make_args(use_critic=True)).trainers[1].overrides

        assert set(overrides) == {
            "kl_coef",
            "use_opd",
            "disable_param_buffers_cpu_backup",
            "load",
            "save",
            "lr",
            "lr_warmup_iters",
        }

    def test_a_policy_override_of_a_critic_neutralized_field_loses_to_the_neutralization(self, tmp_path):
        """The overlay order is what neutralizes the critic, so a policy override of the same field must not win."""
        path = _write_yaml({"trainers": [{"model_id": "alpha", "overrides": {"lr": 5e-7, "eps_clip": 0.3}}]}, tmp_path)

        overrides = resolve_megatron_config(_make_args(path, use_critic=True, critic_lr=2e-6)).trainers[1].overrides

        assert (overrides["lr"], overrides["eps_clip"]) == (2e-6, 0.3)


class TestBaseArgumentsAreNotMutated:
    def test_the_base_arguments_are_never_mutated_by_an_overlay(self, tmp_path):
        """A shallow copy would let one trainer's overlay reach every other trainer through a shared dict."""
        path = _write_yaml({"trainers": [{"model_id": "a", "overrides": {"lr": 5e-7}}, {"model_id": "b"}]}, tmp_path)
        args = _make_args(path, train_env_vars={"NCCL_DEBUG": "WARN"})

        model = _model_args(args, model_id="a")
        model.train_env_vars["NCCL_DEBUG"] = "INFO"

        assert args.train_env_vars == {"NCCL_DEBUG": "WARN"}


class TestPerPolicyArgsCoverage:
    @pytest.mark.parametrize("model_type", ["qwen2.5-0.5B", "qwen3-0.6B"])
    def test_the_whitelist_admits_every_argument_of_a_model_script(self, model_type):
        """A policy that cannot override one of its own architecture arguments would train another model's shape."""
        options = get_megatron_arg_parser()._option_string_actions
        flags = [token for token in load_model_args(model_type).split() if token.startswith("--")]
        unknown = [flag for flag in flags if flag not in options]
        assert not unknown, f"{model_type} passes {unknown}, which the megatron parser does not declare"

        assert {options[flag].dest for flag in flags} <= PER_POLICY_ARGS

    def test_the_whitelist_names_arguments_the_parser_actually_produces(self):
        """The override keys are compared against this set and then looked up by the same name in the
        parser, so a flag spelled the way it appears on a command line admits nothing at all: a config
        carrying the argument's real name is refused, and the name in the set can never be reached."""
        parsed = {action.dest for action in get_megatron_arg_parser()._actions}

        assert PER_POLICY_ARGS <= parsed
