import random
from argparse import Namespace

import pytest

from examples.mopd_puzzles.data_source import BalancedPuzzleDataSource, _BalancedDataset
from miles.rollout.data_source import pop_first
from miles.utils.types import Sample


class _Dataset:
    def __init__(self):
        self.original = [
            Sample(prompt=f"{domain}-{i}", metadata={"domain": domain})
            for domain in ["countdown", "graph_color"]
            for i in range(6)
        ]
        self.shuffle(0)

    def shuffle(self, epoch):
        self.samples = list(self.original)
        random.Random(42 + epoch).shuffle(self.samples)


def test_balanced_batches_survive_epoch_boundary_and_buffer():
    source = BalancedPuzzleDataSource.__new__(BalancedPuzzleDataSource)
    source.args = Namespace(rollout_shuffle=True, n_samples_per_prompt=1)
    source.dataset = _BalancedDataset(_Dataset())
    source.sample_offset = source.sample_index = source.sample_group_index = source.epoch_id = 0
    source.buffer = []
    source.buffer_filter = pop_first
    batches = [source.get_samples(8) for _ in range(3)]
    for batch in batches:
        assert [group[0].metadata["domain"] for group in batch] == ["countdown", "graph_color"] * 4
    flat = [group[0] for batch in batches for group in batch]
    assert len({s.prompt for s in flat[:12]}) == 12
    assert len({s.prompt for s in flat[12:]}) == 12
    assert [s.index for s in flat] == list(range(24))
    source.add_samples(batches[-1][:2])
    returned = source.get_samples(2)
    assert returned == batches[-1][:2]
    assert source.sample_index == 24


def test_balanced_shuffle_is_reproducible_for_resume():
    dataset = _BalancedDataset(_Dataset())
    dataset.shuffle(3)
    expected = [s.prompt for s in dataset.samples]
    dataset.shuffle(4)
    assert [s.prompt for s in dataset.samples] != expected
    dataset.shuffle(3)
    assert [s.prompt for s in dataset.samples] == expected


def test_balanced_dataset_rejects_missing_domain():
    dataset = _Dataset()
    dataset.samples = [s for s in dataset.samples if s.metadata["domain"] == "countdown"]
    with pytest.raises(ValueError, match="equally sized"):
        _BalancedDataset(dataset)
