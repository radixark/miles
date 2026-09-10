from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from miles.backends.fsdp_utils import checkpoint
from miles.backends.training_utils.weight_version_checkpoint import read_weight_version


class TestWeightVersionCheckpoint:
    def test_save_and_load_restore_the_selected_step_counter(
        self, checkpoint_actor: SimpleNamespace, tmp_path: Path
    ) -> None:
        """FSDP saves rollout r under iteration r+1 and resumes the explicitly selected counter."""
        checkpoint.save(actor=checkpoint_actor, iteration=2)
        checkpoint_actor.weight_updater.weight_version = 20
        checkpoint.save(actor=checkpoint_actor, iteration=4)
        checkpoint_actor.args.ckpt_step = 3

        payload = checkpoint.load(actor=checkpoint_actor)
        assert payload is not None
        checkpoint.finalize_load(actor=checkpoint_actor, checkpoint_payload=payload)

        assert payload["iteration"] == 3
        assert checkpoint_actor.weight_updater.weight_version == 9
        assert read_weight_version(checkpoint_dir=tmp_path, iteration=5) == 20
        assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "5"

    def test_only_the_tracker_writer_saves_the_counter(
        self, checkpoint_actor: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Other FSDP ranks must not race the tracker writer's version file."""
        monkeypatch.setattr(checkpoint.dist, "get_rank", Mock(return_value=1))

        checkpoint.save(actor=checkpoint_actor, iteration=2)

        assert not (tmp_path / "iter_0000003" / "weight_version.txt").exists()
