from pathlib import Path

import pytest
from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config, parse_megatron_test_config
from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config

from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.utils.args.runtime import OrchestratorConfig, TrainerConfig
from miles.utils.args.trainer_utils import compute_trainer_ckpt_load_config, compute_trainer_config
from miles.utils.run_uuid import RUN_UUID_LENGTH


class TestTrainerCheckpointConfiguration:
    def test_checkpoint_creation_preserves_the_complete_trainer_config(self, tmp_path: Path) -> None:
        """Identical launch arguments must keep trainer Pod configuration unchanged after saving."""
        load = tmp_path / "run"
        argv = ("--load", str(load), "--ref-load", str(tmp_path / "reference"), "--run-uuid", "a" * RUN_UUID_LENGTH)
        before = parse_megatron_test_config(*argv)
        [trainer] = before.raw_megatron.trainers
        original = compute_trainer_config(before, trainer).model_dump(mode="json")

        load.mkdir()
        (load / "latest_checkpointed_iteration.txt").write_text("3")
        after = parse_megatron_test_config(*argv)
        [trainer] = after.raw_megatron.trainers
        repeated = compute_trainer_config(after, trainer).model_dump(mode="json")

        assert original == repeated

    @pytest.mark.parametrize("field", ["load", "finetune", "no_load_optim", "no_load_rng", "ckpt_step"])
    def test_runtime_loading_fields_are_rejected_in_trainer_configuration(self, field: str) -> None:
        """A serialized loading decision must not silently become part of a trainer Pod template."""
        args = parse_megatron_test_config()
        [trainer] = args.raw_megatron.trainers
        config = compute_trainer_config(args, trainer)
        assert field not in vars(config.backend)
        values = config.model_dump(mode="python")
        values["backend"] = MegatronArgsNamespace(**(vars(config.backend) | {field: None}))
        with pytest.raises(ValueError, match="Checkpoint load inputs"):
            TrainerConfig.model_validate(values)


class TestOrchestratorCheckpointRequests:
    @pytest.mark.parametrize(
        "extra",
        [(), ("--advantage-estimator", "ppo"), ("--megatron-config", encode_megatron_config("a", "b"))],
    )
    def test_loading_uses_only_fields_owned_by_the_orchestrator(self, extra: tuple[str, ...]) -> None:
        """Checkpoint selection must work after deployment-only scaling fields have been removed."""
        args = parse_megatron_test_config(*extra, "--load", "/run", "--ref-load", "/reference")
        orchestrator = OrchestratorConfig.slice_from(args)

        assert "actor_num_nodes" not in type(orchestrator).model_fields
        for trainer in args.raw_megatron.trainers:
            request = compute_trainer_ckpt_load_config(orchestrator, trainer)
            assert request == compute_trainer_ckpt_load_config(args, trainer)
            assert request.load == "/reference"

    def test_requested_load_is_not_transported_to_trainers(self) -> None:
        """The requested destination remains with orchestration while trainers receive loading inputs per call."""
        args = parse_megatron_test_config("--load", "/run", "--ref-load", "/reference")
        orchestrator = OrchestratorConfig.slice_from(args)
        trainer = compute_trainer_config(args, args.raw_megatron.trainers[0])

        assert orchestrator.requested_load == "/run"
        assert "requested_load" not in trainer.model_dump()
        assert "requested_load" not in vars(trainer.backend)


class TestSinglePolicyCriticCheckpointRequests:
    @pytest.mark.parametrize("actor_tracked", [False, True])
    @pytest.mark.parametrize("critic_source", ["unset", "hf", "tracked", "same_as_actor"])
    def test_each_trainer_resolves_its_own_checkpoint_without_replacing_explicit_weights(
        self, tmp_path: Path, actor_tracked: bool, critic_source: str
    ) -> None:
        """Each PPO trainer resolves its own resume state and retains an explicit weight source."""
        actor = tmp_path / "actor"
        actor.mkdir()
        if actor_tracked:
            (actor / "latest_checkpointed_iteration.txt").write_text("7")
        reference = tmp_path / "reference"
        critic = actor if critic_source == "same_as_actor" else tmp_path / "critic"
        critic.mkdir(exist_ok=True)
        if critic_source == "tracked":
            (critic / "latest_checkpointed_iteration.txt").write_text("7")
        extra = [] if critic_source == "unset" else ["--critic-load", str(critic)]
        args = parse_megatron_test_config(
            "--advantage-estimator", "ppo", "--load", str(actor), "--ref-load", str(reference), *extra
        )
        orchestrator = OrchestratorConfig.slice_from(args)
        [actor_config, critic_config] = args.raw_megatron.trainers
        before = orchestrator.model_dump(mode="python")

        actor_request = compute_trainer_ckpt_load_config(orchestrator, actor_config)
        critic_request = compute_trainer_ckpt_load_config(orchestrator, critic_config)

        assert actor_request.load == str(actor if actor_tracked else reference)
        assert critic_request.load == (actor_request.load if critic_source == "unset" else str(critic))
        critic_tracked = critic_source == "tracked" or (actor_tracked and critic_source in {"unset", "same_as_actor"})
        assert actor_request.resume_from_ckpt is actor_tracked
        assert critic_request.resume_from_ckpt is critic_tracked
        for request, tracked in [(actor_request, actor_tracked), (critic_request, critic_tracked)]:
            assert bool(request.finetune) is not tracked
            assert bool(request.no_load_optim) is not tracked
            assert bool(request.no_load_rng) is not tracked
        assert orchestrator.model_dump(mode="python") == before
        assert args.critic_load == (None if critic_source == "unset" else str(critic))
        trainer = compute_trainer_config(args, critic_config)
        assert not {"load", "finetune", "no_load_optim", "no_load_rng", "ckpt_step"} & vars(trainer.backend).keys()

    def test_resuming_actor_keeps_critic_checkpoint_and_optimizer_restore_flags(self, tmp_path: Path) -> None:
        """Two resumable PPO checkpoints independently retain their paths and optimizer restore flags."""
        actor = tmp_path / "actor"
        critic = tmp_path / "critic"
        for directory in (actor, critic):
            directory.mkdir()
            (directory / "latest_checkpointed_iteration.txt").write_text("7")
        args = parse_megatron_test_config(
            "--advantage-estimator", "ppo", "--load", str(actor), "--critic-load", str(critic)
        )
        [actor_config, critic_config] = args.raw_megatron.trainers

        actor_request = compute_trainer_ckpt_load_config(args, actor_config)
        critic_request = compute_trainer_ckpt_load_config(args, critic_config)

        assert actor_request.load == str(actor)
        assert critic_request.load == str(critic)
        assert critic_request.resume_from_ckpt is True
        assert not critic_request.finetune
        assert not critic_request.no_load_optim
        assert not critic_request.no_load_rng
        assert critic_request.model_dump(exclude={"load"}) == actor_request.model_dump(exclude={"load"})

    @pytest.mark.parametrize("tracked", [False, True])
    def test_named_critic_resolves_its_own_checkpoint_or_explicit_initial_weights(
        self, tmp_path: Path, tracked: bool
    ) -> None:
        """Named critics resume their own checkpoint or initialize from their explicit reference weights."""
        critic = tmp_path / "critic"
        critic.mkdir()
        directory = critic / "trainers" / "policy-critic"
        if tracked:
            directory.mkdir(parents=True)
            (directory / "latest_checkpointed_iteration.txt").write_text("7")
        args = parse_megatron_test_config(
            "--advantage-estimator",
            "ppo",
            "--megatron-config",
            encode_megatron_config("policy"),
            "--load",
            str(tmp_path / "actor"),
            "--ref-load",
            str(tmp_path / "reference"),
            "--critic-load",
            str(critic),
        )
        [actor_config, critic_config] = args.raw_megatron.trainers

        actor_request = compute_trainer_ckpt_load_config(args, actor_config)
        critic_request = compute_trainer_ckpt_load_config(args, critic_config)

        assert actor_request.resume_from_ckpt is False
        assert actor_request.load == str(tmp_path / "reference")
        assert critic_request.resume_from_ckpt is tracked
        assert critic_request.load == str(directory if tracked else critic)
        assert bool(critic_request.finetune) is not tracked
        assert bool(critic_request.no_load_optim) is not tracked
        assert bool(critic_request.no_load_rng) is not tracked


class TestCriticLoadDefaultBoundaries:
    def test_non_megatron_default_keeps_the_requested_actor_source(self, tmp_path: Path) -> None:
        """FSDP parsing preserves the requested load instead of applying Megatron fallback resolution."""
        reference = str(tmp_path / "reference")
        args = parse_fsdp_test_config("--load", str(tmp_path / "missing"), "--ref-load", reference)

        assert args.critic_load == args.load == str(tmp_path / "missing")
        assert args.ref_load == reference

    def test_named_megatron_preserves_an_unset_critic_source_and_inherits_the_checkpoint_root(
        self, tmp_path: Path
    ) -> None:
        """An unset critic source inherits the common root while retaining its own checkpoint subdirectory."""
        root = tmp_path / "run"
        directory = root / "trainers" / "policy-critic"
        directory.mkdir(parents=True)
        (directory / "latest_checkpointed_iteration.txt").write_text("7")
        args = parse_megatron_test_config(
            "--advantage-estimator", "ppo", "--megatron-config", encode_megatron_config("policy"), "--load", str(root)
        )
        [_, critic] = args.raw_megatron.trainers
        request = compute_trainer_ckpt_load_config(args, critic)

        assert args.critic_load is None
        assert "load" not in critic.overrides
        assert "ref_load" not in critic.overrides
        assert request.load == str(directory)
        assert request.resume_from_ckpt is True
