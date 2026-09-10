from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from miles.backends.fsdp_utils import checkpoint
from miles.backends.training_utils.weight_version_checkpoint import read_weight_version


def test_finalize_load_restores_weight_updater_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """FSDP applies the counter selected with the restored weight checkpoint."""
    monkeypatch.setattr(checkpoint.torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(checkpoint.dist, "barrier", Mock())
    actor = SimpleNamespace(
        args=SimpleNamespace(no_load_rng=True, start_rollout_id=None),
        global_step=0,
        micro_step=0,
        weight_updater=SimpleNamespace(weight_version=20),
    )

    checkpoint.finalize_load(
        actor=actor,
        checkpoint_payload={"rng": None, "metadata": {}, "iteration": 3, "weight_version": 9},
    )

    assert actor.weight_updater.weight_version == 9


def test_save_records_the_published_weight_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FSDP stores the updater counter beside the exact checkpoint weights."""
    monkeypatch.setattr(checkpoint.torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(checkpoint.torch.cuda, "get_rng_state_all", Mock(return_value=[]))
    monkeypatch.setattr(checkpoint.dist, "barrier", Mock())
    monkeypatch.setattr(checkpoint.dist, "get_rank", Mock(return_value=0))
    monkeypatch.setattr(checkpoint.dist, "get_world_size", Mock(return_value=1))
    monkeypatch.setattr(checkpoint.dcp, "save", Mock())
    actor = SimpleNamespace(
        args=SimpleNamespace(save=str(tmp_path), no_save_optim=True),
        model=object(),
        global_step=0,
        micro_step=0,
        weight_updater=SimpleNamespace(weight_version=9),
    )

    checkpoint.save(actor=actor, iteration=2)

    assert read_weight_version(checkpoint_dir=tmp_path, iteration=3) == 9
