"""Samples the rollout executor drops or restores hand their output store refs over, or refuse to load."""

import pytest
import torch
from tests.fast.ray.rollout.conftest import make_args, make_sample
from tests.fast.utils.output_store.test_dropped import sample_owning

from miles.ray.rollout.debug_data import _load_rollout_data_file
from miles.ray.rollout.output_snapshotter import _RolloutExecutorOutputSnapshotter
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.utils.output_store import dropped
from miles.utils.output_store.dropped import take_dropped_refs
from miles.utils.types import Sample


@pytest.fixture(autouse=True)
def _empty_dropped(monkeypatch):
    monkeypatch.setattr(dropped, "_dropped", {})


def test_trimmed_samples_hand_over_their_refs():
    data = [sample_owning(f"key-{index}") for index in range(3)]
    for index, sample in enumerate(data):
        sample.index = index

    kept, _ = postprocess_rollout_data(make_args(global_batch_size=2), data, train_parallel_config=None)

    assert kept == data[:2]
    assert list(take_dropped_refs()) == ["key-2"]


def test_a_restored_batch_with_refs_is_dropped_and_regenerated(tmp_path):
    args = make_args()
    saved = _RolloutExecutorOutputSnapshotter(args=args)
    saved.capture(trainer_model_id=None, rollout_id=4, data=[sample_owning("stored")], metadata={})
    saved.capture(trainer_model_id=None, rollout_id=5, data=[make_sample(index=1)], metadata={})
    saved.save(tmp_path)

    restored = _RolloutExecutorOutputSnapshotter(args=args)
    restored.load(tmp_path)

    assert restored.get(trainer_model_id=None, rollout_id=4) is None
    assert restored.get(trainer_model_id=None, rollout_id=5) is not None
    assert list(take_dropped_refs()) == ["stored"]


def test_debug_rollout_data_with_refs_is_refused(tmp_path):
    path = tmp_path / "rollout_0.pt"
    torch.save(dict(rollout_id=0, samples=[sample_owning("stored").to_dict()]), path)
    with pytest.raises(ValueError, match="saved with --use-output-store"):
        _load_rollout_data_file(path)

    torch.save(dict(rollout_id=0, samples=[Sample(index=0).to_dict()]), path)
    [sample], _ = _load_rollout_data_file(path)
    assert sample.rollout_store_plan is None
