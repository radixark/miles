from argparse import Namespace
from pathlib import Path

import pytest
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config

from miles.backends.megatron_utils.checkpoint_request import MegatronCheckpointLoad
from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.utils.args.trainer_utils import compute_trainer_checkpoint_load
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
        before_request = compute_trainer_checkpoint_load(before, trainer)

        load.mkdir()
        (load / "latest_checkpointed_iteration.txt").write_text("3")
        after = parse_megatron_test_config(*argv)
        [trainer] = after.raw_megatron.trainers
        after_request = compute_trainer_checkpoint_load(after, trainer)

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

    def test_reference_fallback_does_not_mutate_the_orchestrator_inputs(self, tmp_path: Path) -> None:
        """Deriving a fresh load preserves the requested destination and explicit options."""
        args = Namespace(
            load=str(tmp_path / "missing"),
            ref_load="reference",
            hf_checkpoint="hf",
            megatron_to_hf_mode="core",
            ref_ckpt_step=7,
            ckpt_step=None,
            no_load_optim=False,
            no_load_rng=False,
            finetune=False,
        )
        original = vars(args).copy()
        request = MegatronCheckpointLoad.from_args(args)
        assert vars(args) == original
        assert request.load == "reference"
        assert request.ckpt_step == 7
        assert not request.resume_from_ckpt

    @pytest.mark.parametrize("existing", [False, True])
    def test_loading_restores_the_same_namespace_even_when_the_loader_raises(self, existing: bool) -> None:
        """A failed load must not leak temporary fields into the shared Megatron namespace."""
        prior = dict(load="original", no_load_optim=False, no_load_rng=False, finetune=False, ckpt_step=1)
        args = MegatronArgsNamespace(**(prior if existing else {}))
        original = vars(args).copy()
        request = MegatronCheckpointLoad(
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
