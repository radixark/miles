from argparse import Namespace

import pytest

from miles.backends.training_utils import data as data_utils
from miles.backends.training_utils.parallel import GroupInfo, ParallelState


def _parallel_state(*, dp_size: int) -> ParallelState:
    trivial_group = GroupInfo(rank=0, size=1, group=None)
    dp_group = GroupInfo(rank=0, size=dp_size, group=None)
    return ParallelState(
        intra_dp=dp_group,
        intra_dp_cp=dp_group,
        cp=trivial_group,
        tp=trivial_group,
        pp=trivial_group,
        ep=trivial_group,
        etp=trivial_group,
        indep_dp=trivial_group,
    )


def _static_args() -> Namespace:
    return Namespace(
        qkv_format="thd",
        global_batch_size=256,
        use_dynamic_global_batch_size=False,
        use_dynamic_batch_size=False,
        micro_batch_size=8,
    )


class TestGetDataIteratorTrainingSideSchedule:
    def test_rejects_a_dp_size_that_does_not_divide_the_global_batch(self, monkeypatch):
        """Batches the rollout side cannot schedule still need the live cell count to divide the global batch."""
        monkeypatch.setattr(data_utils, "get_parallel_state", lambda: _parallel_state(dp_size=3))

        with pytest.raises(AssertionError, match="must be divisible by dp_size"):
            data_utils.get_data_iterator(_static_args(), model=None, rollout_data={"total_lengths": [4] * 86})

    def test_a_dividing_dp_size_keeps_the_fixed_size_micro_batches(self, monkeypatch):
        """The divisibility check must not disturb the legacy static path it guards."""
        monkeypatch.setattr(data_utils, "get_parallel_state", lambda: _parallel_state(dp_size=4))

        data_iterators, num_microbatches = data_utils.get_data_iterator(
            _static_args(), model=None, rollout_data={"total_lengths": [4] * 64}
        )

        assert num_microbatches == [8]
        assert len(data_iterators) == 1
