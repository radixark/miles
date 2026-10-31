from miles.backends.training_utils.data import rollout
from miles.utils.data import get_minimum_num_micro_batch_size
from miles.utils.seqlen_balancing import get_seqlen_balanced_partitions

# per-rank sample lengths where first-fit packs 4 bins under 8192 but the balanced 4-way partition has a 9088 bin
_LENGTHS = [4992, 7296, 3840, 2688, 4224, 2432, 2560, 2816]


def _largest(lengths, num):
    return max(sum(lengths[i] for i in p) for p in get_seqlen_balanced_partitions(lengths, num, equal_size=False))


def test_the_balanced_microbatches_stay_within_the_token_cap():
    """A fixed-shape trainer (TPU) cannot run a microbatch above --max-tokens-per-gpu."""
    assert get_minimum_num_micro_batch_size(_LENGTHS, 8192) == 4 and _largest(_LENGTHS, 4) > 8192
    num = rollout._num_microbatches_within_cap(_LENGTHS, 8192)
    assert _largest(_LENGTHS, num) <= 8192


def test_a_sample_above_the_cap_keeps_a_microbatch_of_its_own():
    lengths = [9000, 1000, 1000]
    num = rollout._num_microbatches_within_cap(lengths, 8192)
    assert num == 2 and _largest(lengths, num) == 9000
