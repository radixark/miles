from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tests.fast.fixtures.args_fixtures import make_trainer_args
from tests.fast.fixtures.fsdp_checkpoint_config_fixtures import fsdp_checkpoint_sdk

from miles.backends.fsdp_utils import checkpoint

__all__ = ["fsdp_checkpoint_sdk"]


class TestFsdpCheckpointConfigFields:
    @pytest.mark.parametrize("omit_optimizer", [False, True])
    def test_saving_uses_the_declared_optimizer_persistence_switch(
        self, fsdp_checkpoint_sdk: SimpleNamespace, tmp_path: Path, omit_optimizer: bool
    ) -> None:
        """Explicit no-save-optim suppresses optimizer and scheduler files while preserving model state."""
        worker = fsdp_checkpoint_sdk
        worker.args = make_trainer_args(train_backend="fsdp", save=str(tmp_path), no_save_optim=omit_optimizer)

        checkpoint.save(worker, iteration=3)

        saved = tmp_path / "iter_0000004"
        assert (saved / "model" / "state.pt").exists()
        assert (saved / "optimizer" / "state.pt").exists() is (not omit_optimizer)
        assert (saved / "lr_scheduler" / "state.pt").exists() is (not omit_optimizer)
        assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "4"

    @pytest.mark.parametrize("omit_optimizer", [False, True])
    def test_loading_restores_model_weights_independently_of_optimizer_selection(
        self, fsdp_checkpoint_sdk: SimpleNamespace, tmp_path: Path, omit_optimizer: bool
    ) -> None:
        """Explicit no-load-optim preserves live optimizer settings while still loading requested model weights."""
        worker = fsdp_checkpoint_sdk
        expected = worker.model.weight.detach().clone()
        checkpoint.save(worker, iteration=3)
        worker.args = make_trainer_args(
            train_backend="fsdp", load=str(tmp_path), ckpt_step=4, no_load_optim=omit_optimizer
        )
        with torch.no_grad():
            worker.model.weight.fill_(99)
        worker.optimizer.param_groups[0]["lr"] = 2.0

        payload = checkpoint.load(worker)

        assert payload["iteration"] == 4
        torch.testing.assert_close(worker.model.weight, expected)
        assert worker.optimizer.param_groups[0]["lr"] == (2.0 if omit_optimizer else 0.125)

    @pytest.mark.parametrize("no_load_rng", [False, True])
    @pytest.mark.parametrize("requested_start", [None, 9])
    def test_finalizing_respects_rng_selection_and_preserves_an_explicit_start(
        self,
        fsdp_checkpoint_sdk: SimpleNamespace,
        no_load_rng: bool,
        requested_start: int | None,
    ) -> None:
        """Restoring RNG is optional and iteration fallback cannot overwrite an explicit rollout start."""
        worker = fsdp_checkpoint_sdk
        worker.args = make_trainer_args(
            train_backend="fsdp", no_load_rng=no_load_rng, start_rollout_id=requested_start
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(101)
            saved = torch.get_rng_state()
            torch.manual_seed(202)
            current = torch.get_rng_state()

            checkpoint.finalize_load(worker, {"rng": {"torch": saved}, "metadata": {}, "iteration": 4})

            assert torch.equal(torch.get_rng_state(), current if no_load_rng else saved)
            assert worker.args.start_rollout_id == (4 if requested_start is None else requested_start)
