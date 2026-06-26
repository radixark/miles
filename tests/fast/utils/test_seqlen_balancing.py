from __future__ import annotations

import random

from hypothesis import given, settings
from hypothesis import strategies as st

from miles.utils.seqlen_balancing import get_seqlen_bounded_partitions


def _counts(partitions: list[list[int]]) -> list[int]:
    return [len(p) for p in partitions]


class TestBoundedPartitions:
    def test_covers_every_index_exactly_once(self):
        seqlens = [5, 1, 9, 3, 7, 2, 8, 4, 6, 10]  # n=10
        partitions = get_seqlen_bounded_partitions(seqlens, k_partitions=4)
        flat = sorted(i for p in partitions for i in p)
        assert flat == list(range(len(seqlens)))

    def test_returns_k_partitions(self):
        partitions = get_seqlen_bounded_partitions([1, 2, 3, 4, 5], k_partitions=3)
        assert len(partitions) == 3

    def test_sample_count_delta_at_most_one_when_not_divisible(self):
        # n=10, k=4 -> counts must be {3,3,2,2}; max-min == 1
        partitions = get_seqlen_bounded_partitions([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], k_partitions=4)
        counts = _counts(partitions)
        assert max(counts) - min(counts) <= 1
        assert sorted(counts) == [2, 2, 3, 3]

    def test_exact_equal_counts_when_divisible(self):
        # n=12, k=4 -> every rank gets exactly 3 (delta 0)
        partitions = get_seqlen_bounded_partitions(list(range(1, 13)), k_partitions=4)
        assert _counts(partitions) == [3, 3, 3, 3]

    def test_balances_token_sums(self):
        # identical lengths: token sums should track the (bounded) counts, not
        # pile onto one rank. With n=10,k=4 and equal lengths, sums ratio == counts.
        partitions = get_seqlen_bounded_partitions([10] * 10, k_partitions=4)
        sums = sorted(sum(10 for _ in p) for p in partitions)
        assert sums == [20, 20, 30, 30]

    def test_each_partition_nonempty_when_n_at_least_k(self):
        partitions = get_seqlen_bounded_partitions([3, 1, 2, 5, 4], k_partitions=4)  # n=5,k=4
        assert all(len(p) >= 1 for p in partitions)

    def test_extra_item_lands_in_lightest_group(self):
        """The 'which group gets the extra item' choice must follow token weight,
        not a fixed index. With lengths [10, 9, 1, 1, 1] and k=2 the optimal split
        is {10,1} / {9,1,1} -> sums 11/11. A static 'first rem groups get +1'
        capacity instead forces 12/10."""
        lengths = [10, 9, 1, 1, 1]
        partitions = get_seqlen_bounded_partitions(lengths, k_partitions=2)
        sums = sorted(sum(lengths[i] for i in p) for p in partitions)
        assert sums == [11, 11]
        # still bounded
        counts = sorted(_counts(partitions))
        assert counts == [2, 3]


class TestBoundedPartitionsProperties:
    @settings(deadline=None, max_examples=80)
    @given(
        n=st.integers(min_value=1, max_value=64),
        k=st.integers(min_value=1, max_value=16),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_invariants_hold_for_random_inputs(self, n, k, seed):
        # only defined for n >= k (token balancing needs at least one item per rank)
        if n < k:
            return
        rng = random.Random(seed)
        seqlens = [rng.randint(1, 1000) for _ in range(n)]
        partitions = get_seqlen_bounded_partitions(seqlens, k_partitions=k)

        # exactly k partitions
        assert len(partitions) == k
        # every index assigned exactly once
        flat = sorted(i for p in partitions for i in p)
        assert flat == list(range(n))
        # bounded unevenness: counts differ by at most 1
        counts = _counts(partitions)
        assert max(counts) - min(counts) <= 1
        # counts are exactly the floor/ceil split
        base, rem = divmod(n, k)
        assert sorted(counts) == sorted([base + 1] * rem + [base] * (k - rem))
