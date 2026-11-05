from argparse import Namespace

import pytest

from miles.rollout.reward_shaping import reward_group_segments, rollout_segments
from miles.utils.types import Sample


def test_prompt_identity_groups_interleaved_rows() -> None:
    samples = [Sample(group_index=i) for i in [7, 8, 7]]
    assert reward_group_segments(Namespace(), samples) == [[0, 2], [1]]


def test_explicit_prompt_boundaries_take_priority() -> None:
    samples = [Sample(group_index=7) for _ in range(3)]
    assert reward_group_segments(Namespace(), samples, [0, 2, 1]) == [[0, 1], [2]]
    with pytest.raises(AssertionError, match="prompt group sizes"):
        reward_group_segments(Namespace(), samples, [2])


def test_legacy_full_and_partial_batches() -> None:
    args = Namespace(n_samples_per_prompt=2, rollout_batch_size=2)
    samples = [Sample() for _ in range(4)]
    assert reward_group_segments(args, samples) == [[0, 1], [2, 3]]
    assert reward_group_segments(args, samples[:3]) == [[0, 1, 2]]
    assert reward_group_segments(args, []) == []


def test_sibling_identity_falls_back_to_sample_and_row() -> None:
    samples = [Sample(rollout_id=10), Sample(rollout_id=10), Sample(index=5), Sample(index=5), Sample(), Sample()]
    assert rollout_segments(samples, list(range(6))) == [[0, 1], [2, 3], [4], [5]]
