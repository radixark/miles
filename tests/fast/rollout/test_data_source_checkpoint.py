from argparse import Namespace

import torch

from miles.rollout.data_source import RolloutDataSource


def make_source(tmp_path, **overrides):
    args = {
        "rollout_global_dataset": True,
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
    source._rollout_state_snapshots = {}
    return source


def test_save_uses_immutable_rollout_snapshot_and_prunes_old_snapshots(tmp_path):
    source = make_source(tmp_path)
    source.snapshot(2)

    source.sample_offset = 64
    source.epoch_id = 2
    source.sample_group_index = 11
    source.sample_index = 22
    source.metadata = {"nested": {"source": "rollout-3"}}
    source.snapshot(3)

    source.sample_offset = 96
    source.epoch_id = 3
    source.sample_group_index = 15
    source.sample_index = 30
    source.metadata["nested"]["source"] = "rollout-4"
    source.snapshot(4)

    source.save(3)

    state = torch.load(
        tmp_path / "run" / "rollout" / "global_dataset_state_dict_3.pt",
        weights_only=True,
    )
    assert state == {
        "sample_offset": 64,
        "epoch_id": 2,
        "sample_group_index": 11,
        "sample_index": 22,
        "metadata": {"nested": {"source": "rollout-3"}},
    }
    assert set(source._rollout_state_snapshots) == {4}

def test_save_without_a_matching_snapshot_falls_back_to_the_live_cursor(tmp_path):
    """An eval-only or externally triggered save has no generate behind it. Writing the live
    cursor keeps that path working; it used to raise."""
    source = make_source(tmp_path)
    source.sample_offset = 16

    source.save(0)

    state = torch.load(tmp_path / "run" / "rollout" / "global_dataset_state_dict_0.pt", weights_only=True)
    assert state["sample_offset"] == 16
