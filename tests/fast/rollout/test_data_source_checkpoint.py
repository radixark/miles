from argparse import Namespace

import pytest
import torch

from miles.rollout.data_source import RolloutDataSource


def make_source(tmp_path, **overrides):
    args = {
        "rollout_global_dataset": True,
        "rollout_data_load": None,
        "load": str(tmp_path / "base-model"),
        "save": str(tmp_path / "run"),
        "rollout_shuffle": False,
    }
    args.update(overrides)

    source = RolloutDataSource.__new__(RolloutDataSource)
    source.args = Namespace(**args)
    source.sample_offset = 0
    source.epoch_id = 0
    source.sample_group_index = 0
    source.sample_index = 0
    source.metadata = {}
    return source


def test_lora_resume_loads_dataset_cursor_from_adapter_run_root(tmp_path):
    run_root = tmp_path / "run"
    state_path = run_root / "rollout" / "global_dataset_state_dict_7.pt"
    state_path.parent.mkdir(parents=True)
    torch.save(
        {
            "sample_offset": 64,
            "epoch_id": 2,
            "sample_group_index": 11,
            "sample_index": 22,
            "metadata": {"source": "checkpoint"},
        },
        state_path,
    )

    source = make_source(
        tmp_path,
        rollout_data_load=str(run_root),
        load=str(tmp_path / "base-model"),
    )

    source.load(rollout_id=7)

    assert source.sample_offset == 64
    assert source.epoch_id == 2
    assert source.sample_group_index == 11
    assert source.sample_index == 22
    assert source.metadata == {"source": "checkpoint"}

def test_missing_resume_cursor_fails_loudly(tmp_path):
    source = make_source(tmp_path, rollout_data_load=str(tmp_path / "missing-run"))

    with pytest.raises(FileNotFoundError, match="global_dataset_state_dict_7.pt"):
        source.load(rollout_id=7)

def test_explicit_rollout_id_without_resume_root_keeps_existing_behavior(tmp_path):
    source = make_source(tmp_path)

    source.load(rollout_id=7)

    assert source.sample_offset == 0

def test_fresh_run_does_not_require_negative_rollout_cursor(tmp_path):
    source = make_source(tmp_path, rollout_data_load=str(tmp_path / "new-run"))

    source.load(rollout_id=-1)

    assert source.sample_offset == 0
