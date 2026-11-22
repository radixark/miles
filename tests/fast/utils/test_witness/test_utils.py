import random

from miles.utils.audit_utils.witness.utils import compute_id_ranges, exclude_id_ranges


class TestComputeIdRanges:
    def test_empty_ids_give_no_ranges(self) -> None:
        """No stale ids log no ranges."""
        assert compute_id_ranges([]) == []

    def test_a_consecutive_run_becomes_one_range(self) -> None:
        """The ring buffer's stale window collapses into a single half-open pair."""
        assert compute_id_ranges(range(3, 1_000_000)) == [(3, 1_000_000)]

    def test_a_wrapped_run_becomes_two_sorted_ranges(self) -> None:
        """A stale window crossing the buffer end splits at zero."""
        assert compute_id_ranges([8, 9, 0, 1]) == [(0, 2), (8, 10)]

    def test_unordered_and_duplicate_ids_are_covered_exactly(self) -> None:
        """Any id list maps to disjoint sorted ranges holding exactly its ids."""
        ids = [5, 3, 4, 4, 10, 11, 7, 3]

        ranges = compute_id_ranges(ids)

        assert ranges == [(3, 6), (7, 8), (10, 12)]
        assert {x for start, stop in ranges for x in range(start, stop)} == set(ids)


class TestExcludeIdRanges:
    def test_ids_inside_any_range_are_removed(self) -> None:
        """Both a short and a long range drop exactly the ids they cover."""
        assert exclude_id_ranges([0, 1, 2, 5, 9, 10**6], [(1, 3), (9, 2 * 10**6)]) == {0, 5}

    def test_matches_a_plain_set_difference(self) -> None:
        """The range form excludes the same ids as the stale id list it came from."""
        rng = random.Random(0)
        for _ in range(50):
            stale = rng.sample(range(100), k=rng.randint(0, 100))
            ids = rng.sample(range(120), k=rng.randint(0, 120))

            assert exclude_id_ranges(ids, compute_id_ranges(stale)) == set(ids) - set(stale)
