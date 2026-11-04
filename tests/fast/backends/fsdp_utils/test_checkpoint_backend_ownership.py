from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tests.fast.backends.fsdp_utils.checkpoint_fixtures import cpu_checkpoint_actor
from torch.distributed.checkpoint.api import CheckpointException

from miles.backends.fsdp_utils import checkpoint
from miles.backends.fsdp_utils.config import FsdpArgsNamespace

__all__ = ["cpu_checkpoint_actor"]


class TestBackendCheckpointOwnership:
    @pytest.mark.parametrize(
        "save_optimizer",
        [
            pytest.param(
                False,
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=CheckpointException,
                    reason="Existing checkpoint load requires absent optimizer or scheduler metadata after no-save-optim",
                ),
            ),
            True,
        ],
    )
    @pytest.mark.parametrize("load_optimizer", [False, True])
    def test_checkpoint_roundtrip_obeys_backend_optimizer_options_and_restores_progress(
        self, cpu_checkpoint_actor: SimpleNamespace, tmp_path: Path, save_optimizer: bool, load_optimizer: bool
    ) -> None:
        """Actual checkpoint roundtrips restore weights and progress while honoring backend optimizer flags."""
        actor = cpu_checkpoint_actor
        actor.args.backend = FsdpArgsNamespace(
            load=str(tmp_path),
            save=str(tmp_path),
            no_save_optim=not save_optimizer,
            no_load_optim=not load_optimizer,
            no_load_rng=True,
            ckpt_step=None,
        )
        saved_weight = actor.model.weight.detach().clone()
        try:
            checkpoint.save(actor=actor, iteration=6)
        except CheckpointException as error:
            raise AssertionError(
                "Checkpoint save must succeed before exercising missing optimizer metadata"
            ) from error
        step = tmp_path / "iter_0000007"
        assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "7"
        assert (step / "optimizer" / ".metadata").exists() is save_optimizer
        assert (step / "lr_scheduler" / ".metadata").exists() is save_optimizer
        with torch.no_grad():
            actor.model.weight.fill_(4)
        actor.optimizer.state.clear()
        actor.optimizer.zero_grad(set_to_none=True)
        actor.optimizer.param_groups[0]["lr"] = 0.9
        actor.global_step = 0
        actor.micro_step = 0

        try:
            payload = checkpoint.load(actor=actor)
        except CheckpointException as error:
            assert not save_optimizer
            assert set(error.failures) == {0}
            failure, _ = error.failures[0]
            assert type(failure) is AssertionError
            assert failure.args == ("metadata is None",)
            raise
        checkpoint.finalize_load(actor=actor, checkpoint_payload=payload)

        torch.testing.assert_close(actor.model.weight, saved_weight)
        assert actor.optimizer.param_groups[0]["lr"] == (0.05 if save_optimizer and load_optimizer else 0.9)
        if save_optimizer or not load_optimizer:
            assert bool(actor.optimizer.state) is (save_optimizer and load_optimizer)
        assert actor.global_step == 11
        assert actor.micro_step == 13
        assert actor.args.start_rollout_id == 7

    @pytest.mark.parametrize("restore_rng", [False, True])
    def test_checkpoint_finalization_honors_backend_rng_restore_switch(
        self, cpu_checkpoint_actor: SimpleNamespace, restore_rng: bool
    ) -> None:
        """Checkpoint finalization restores the saved CPU RNG only when its backend option permits it."""
        actor = cpu_checkpoint_actor
        original = torch.get_rng_state()
        actor.args.backend = FsdpArgsNamespace(no_load_rng=not restore_rng)
        saved = torch.Generator().manual_seed(2718).get_state()
        payload = {"rng": {"torch": saved}, "metadata": {}, "iteration": 3}
        try:
            checkpoint.finalize_load(actor=actor, checkpoint_payload=payload)
            assert torch.equal(torch.get_rng_state(), saved if restore_rng else original)
            assert actor.args.start_rollout_id == 3
        finally:
            torch.set_rng_state(original)
