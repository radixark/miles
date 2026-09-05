"""Alternate puzzle domains while preserving Miles' buffering and resume state."""

from collections import defaultdict

from miles.rollout.data_source import RolloutDataSourceWithBuffer


class _BalancedDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self._interleave()

    def _interleave(self):
        groups = defaultdict(list)
        for sample in self.dataset.samples:
            groups[sample.metadata["domain"]].append(sample)
        if set(groups) != {"countdown", "graph_color"} or len({len(g) for g in groups.values()}) != 1:
            raise ValueError("The puzzle student requires equally sized Countdown and graph-coloring datasets")
        self.samples = [
            sample for pair in zip(groups["countdown"], groups["graph_color"], strict=True) for sample in pair
        ]

    def shuffle(self, epoch_id):
        self.dataset.shuffle(epoch_id)
        self._interleave()

    def __len__(self):
        return len(self.samples)


class BalancedPuzzleDataSource(RolloutDataSourceWithBuffer):
    def __init__(self, args):
        super().__init__(args)
        if self.dataset is None or args.n_samples_per_prompt != 1:
            raise ValueError("Balanced puzzle OPD requires a global dataset and one response per prompt")
        self.dataset = _BalancedDataset(self.dataset)
