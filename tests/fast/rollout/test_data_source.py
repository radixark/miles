from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from miles.rollout.data_source import RolloutDataSource


def _make_args(**overrides) -> SimpleNamespace:
    defaults = dict(rollout_global_dataset=False, save=None, load=None, rollout_shuffle=False)
    return SimpleNamespace(**{**defaults, **overrides})


def test_save_writes_nothing_without_a_global_dataset(tmp_path: Path) -> None:
    """The built-in source guards itself, so the executor needs no outer guard to keep it silent."""
    source = RolloutDataSource(_make_args(save=str(tmp_path)))

    source.save(rollout_id=3)

    assert list(tmp_path.iterdir()) == []


def test_load_reads_nothing_without_a_global_dataset(tmp_path: Path) -> None:
    """The load side has always been called unconditionally and relies on the same internal guard."""
    source = RolloutDataSource(_make_args(load=str(tmp_path)))

    source.load(rollout_id=3)

    assert source.sample_offset == 0
    assert source.epoch_id == 0


def _global_dataset_source(**overrides) -> RolloutDataSource:
    """Enable the cursor without building a dataset; load() only touches the dataset to shuffle it."""
    source = RolloutDataSource(_make_args(**overrides))
    source.args.rollout_global_dataset = True
    return source


def test_lora_resume_loads_the_cursor_of_the_resumed_run(tmp_path: Path) -> None:
    """A LoRA resume keeps --load on the base model, so the cursor comes from the adapter's run."""
    cursor = tmp_path / "run" / "rollout" / "global_dataset_state_dict_7.pt"
    cursor.parent.mkdir(parents=True)
    torch.save({"sample_offset": 64, "epoch_id": 2}, cursor)
    source = _global_dataset_source(load=str(tmp_path / "base"), lora_resume_root=str(tmp_path / "run"))

    source.load(rollout_id=7)

    assert (source.sample_offset, source.epoch_id) == (64, 2)


def test_lora_resume_without_a_cursor_fails(tmp_path: Path) -> None:
    source = _global_dataset_source(lora_resume_root=str(tmp_path / "run"))

    with pytest.raises(FileNotFoundError, match="global_dataset_state_dict_7.pt"):
        source.load(rollout_id=7)
    source.load(rollout_id=-1)  # a run that starts at rollout 0 has no cursor to restore
