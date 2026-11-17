from pathlib import Path

import pytest
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config

from miles.backends.megatron_utils.checkpoint_request import MegatronCkptLoadConfig
from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.utils.args.trainer_utils import compute_trainer_ckpt_load_config
from miles.utils.run_uuid import RUN_UUID_LENGTH


class TestCheckpointLoadInputs:
    def test_checkpoint_creation_changes_runtime_loading_inputs(self, tmp_path: Path) -> None:
        """A fresh launch loads reference weights while a repeated launch restores saved state."""
        load = tmp_path / "run"
        ref = tmp_path / "reference"
        argv = (
            "--load",
            str(load),
            "--ref-load",
            str(ref),
            "--run-uuid",
            "a" * RUN_UUID_LENGTH,
        )
        before = parse_megatron_test_config(*argv)
        [trainer] = before.raw_megatron.trainers
        before_request = compute_trainer_ckpt_load_config(before, trainer)

        load.mkdir()
        (load / "latest_checkpointed_iteration.txt").write_text("3")
        after = parse_megatron_test_config(*argv)
        [trainer] = after.raw_megatron.trainers
        after_request = compute_trainer_ckpt_load_config(after, trainer)

        assert before_request.load == str(ref)
        assert before_request.resume_from_ckpt is False
        assert before_request.no_load_optim is True
        assert before_request.no_load_rng is True
        assert before_request.finetune is True
        assert after_request.load == str(load)
        assert after_request.resume_from_ckpt is True
        assert not after_request.no_load_optim
        assert not after_request.no_load_rng
        assert not after_request.finetune

    @pytest.mark.parametrize("same_path", [False, True])
    def test_reference_fallback_does_not_mutate_the_orchestrator_inputs(self, tmp_path: Path, same_path: bool) -> None:
        """Deriving a fresh load preserves the requested destination and explicit options."""
        reference = str(tmp_path / "reference")
        args = parse_megatron_test_config(
            "--load",
            reference if same_path else str(tmp_path / "missing"),
            "--ref-load",
            reference,
            "--ref-ckpt-step",
            "7",
        )
        args = args.model_copy(update={"hf_checkpoint": "hf"})
        original = args.model_dump()
        [trainer] = args.raw_megatron.trainers
        request = compute_trainer_ckpt_load_config(args, trainer)
        assert args.model_dump() == original
        assert request.load == reference
        assert request.ckpt_step == 7
        assert not request.resume_from_ckpt

    def test_loading_restores_the_same_namespace_even_when_the_loader_raises(self) -> None:
        """A failed load must not leak temporary fields into the shared Megatron namespace."""
        args = MegatronArgsNamespace()
        original = vars(args).copy()
        request = MegatronCkptLoadConfig(
            load="replacement",
            no_load_optim=True,
            no_load_rng=True,
            finetune=True,
            ckpt_step=4,
            resume_from_ckpt=False,
        )
        with pytest.raises(RuntimeError, match="loader failed"):
            with request.apply(args):
                assert args.load == "replacement"
                assert args.ckpt_step == 4
                assert args.no_load_optim is True
                raise RuntimeError("loader failed")
        assert vars(args) == original


class TestCheckpointLoadScope:
    def test_nested_load_is_rejected_without_destroying_the_outer_request(self) -> None:
        """A nested checkpoint load cannot replace the active request or remove its fields."""
        args = MegatronArgsNamespace()
        outer = MegatronCkptLoadConfig.for_peer_recovery()
        inner = outer.model_copy(update={"load": "other"})
        original = vars(args).copy()

        with outer.apply(args):
            with pytest.raises(AssertionError, match="runtime-only"):
                with inner.apply(args):
                    pytest.fail("Nested checkpoint scope unexpectedly entered")
            assert args.load is None
            assert args.no_load_optim is False
            assert args.no_load_rng is False
        assert vars(args) == original

    @pytest.mark.parametrize(
        "field", ["load", "finetune", "no_load_optim", "no_load_rng", "ckpt_step", "requested_load"]
    )
    def test_preexisting_runtime_fields_are_rejected_without_mutation(self, field: str) -> None:
        """A loading request must not overwrite fields leaked from an earlier call."""
        args = MegatronArgsNamespace(**{field: "existing"})
        original = vars(args).copy()

        with pytest.raises(AssertionError):
            with MegatronCkptLoadConfig.for_peer_recovery().apply(args):
                pytest.fail("Invalid checkpoint scope unexpectedly entered")

        assert vars(args) == original

    def test_sequential_requests_do_not_reuse_previous_load_inputs(self) -> None:
        """Each call receives its own checkpoint inputs and leaves the shared backend clean."""
        args = MegatronArgsNamespace()
        original = vars(args).copy()
        request = MegatronCkptLoadConfig.for_peer_recovery()

        with request.model_copy(update={"load": "first", "ckpt_step": 11}).apply(args):
            assert args.load == "first"
            assert args.ckpt_step == 11
        with request.model_copy(update={"load": "second", "ckpt_step": 22}).apply(args):
            assert args.load == "second"
            assert args.ckpt_step == 22

        assert vars(args) == original

    def test_requested_load_leaked_by_loader_is_rejected_on_exit(self) -> None:
        """A loader cannot silently introduce a second checkpoint source into the shared backend."""
        args = MegatronArgsNamespace()

        with pytest.raises(AssertionError, match="requested_load belongs to the orchestrator"):
            with MegatronCkptLoadConfig.for_peer_recovery().apply(args):
                vars(args)["requested_load"] = "unexpected"

        assert not {"load", "finetune", "no_load_optim", "no_load_rng", "ckpt_step"} & vars(args).keys()


class TestBridgeReferenceInitialization:
    def test_reference_checkpoint_does_not_claim_resume_from_the_new_destination(self, tmp_path: Path) -> None:
        """A bridge reference checkpoint supplies weights without restoring a new run's rollout state."""
        reference = tmp_path / "reference"
        reference.mkdir()
        (reference / "latest_checkpointed_iteration.txt").write_text("100")
        destination = tmp_path / "new-run"
        args = parse_megatron_test_config(
            "--megatron-to-hf-mode", "bridge", "--load", str(destination), "--ref-load", str(reference)
        )
        [trainer] = args.raw_megatron.trainers

        request = compute_trainer_ckpt_load_config(args, trainer)

        assert request.load == str(reference)
        assert request.resume_from_ckpt is False
        assert args.load == str(destination)
        assert args.start_rollout_id is None


class TestExplicitCheckpointSelection:
    @pytest.mark.parametrize("tracked", [False, True])
    @pytest.mark.parametrize("finetune", [False, True])
    def test_step_selection_retains_source_flags_and_tracker_provenance(
        self, tmp_path: Path, tracked: bool, finetune: bool
    ) -> None:
        """An explicit step stays distinct from tracker-driven resume and reference initialization."""
        load = tmp_path / "checkpoint"
        (load / "iter_0000007").mkdir(parents=True)
        if tracked:
            (load / "latest_checkpointed_iteration.txt").write_text("9")
        args = parse_megatron_test_config(
            "--load",
            str(load),
            "--ckpt-step",
            "7",
            "--ref-load",
            str(tmp_path / "reference"),
            *(["--finetune"] if finetune else []),
        )
        original = args.model_dump()
        [trainer] = args.raw_megatron.trainers

        request = compute_trainer_ckpt_load_config(args, trainer)

        assert request.load == str(load)
        assert request.ckpt_step == 7
        assert request.resume_from_ckpt is tracked
        assert bool(request.finetune) is finetune
        assert args.model_dump() == original

    @pytest.mark.parametrize("critic_has_step", [False, True])
    def test_critic_hf_initialization_does_not_inherit_an_actor_only_step(
        self, tmp_path: Path, critic_has_step: bool
    ) -> None:
        """The critic drops an actor-only selector unless its own checkpoint contains that step."""
        actor = tmp_path / "actor"
        (actor / "iter_0000007").mkdir(parents=True)
        critic = tmp_path / "critic"
        critic.mkdir()
        (critic / "config.json").write_text("{}")
        if critic_has_step:
            (critic / "iter_0000007").mkdir()
        args = parse_megatron_test_config(
            "--advantage-estimator",
            "ppo",
            "--load",
            str(actor),
            "--critic-load",
            str(critic),
            "--ckpt-step",
            "7",
        )
        actor_config, critic_config = args.raw_megatron.trainers

        actor_request = compute_trainer_ckpt_load_config(args, actor_config)
        critic_request = compute_trainer_ckpt_load_config(args, critic_config)

        assert actor_request.ckpt_step == 7
        assert critic_request.load == str(critic)
        assert critic_request.ckpt_step == (7 if critic_has_step else None)
        assert critic_request.resume_from_ckpt is False
