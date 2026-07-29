"""Tests for miles.utils.audit_utils.witness.allocator: WitnessIdAllocator, _compute_stale_ids."""

import json

import pytest

from miles.utils.audit_utils.witness.allocator import (
    WitnessIdAllocator,
    WitnessInfo,
    _compute_stale_ids,
    read_persisted_witness_counter,
)


class TestWitnessIdAllocator:
    def test_monotonic_and_wraps(self) -> None:
        allocator = WitnessIdAllocator(buffer_size=5)

        info1 = allocator.allocate(3)
        assert info1.witness_ids == [0, 1, 2]

        info2 = allocator.allocate(4)
        assert info2.witness_ids == [3, 4, 0, 1]

    def test_allocate_returns_correct_count(self) -> None:
        allocator = WitnessIdAllocator(buffer_size=100)
        info = allocator.allocate(7)
        assert len(info.witness_ids) == 7

    def test_allocate_returns_witness_info(self) -> None:
        allocator = WitnessIdAllocator(buffer_size=10)
        info = allocator.allocate(3)
        assert isinstance(info, WitnessInfo)
        assert len(info.witness_ids) == 3
        assert isinstance(info.stale_ids, list)

    def test_stale_ids_computed_on_allocate(self) -> None:
        """Allocating 8 from buffer_size=10 should produce stale IDs (keep 70% = 7)."""
        allocator = WitnessIdAllocator(buffer_size=10)
        info = allocator.allocate(8)
        assert info.witness_ids == [0, 1, 2, 3, 4, 5, 6, 7]
        assert set(info.stale_ids) == {8, 9, 0}

    def test_allocate_zero_ids(self) -> None:
        """allocate(num_ids=0) should return empty witness_ids and reasonable stale_ids."""
        allocator = WitnessIdAllocator(buffer_size=5)
        allocator.allocate(3)
        info = allocator.allocate(0)
        assert info.witness_ids == []
        assert isinstance(info.stale_ids, list)

    def test_non_positive_buffer_size_raises(self) -> None:
        """buffer_size <= 0 raises ValueError at construction."""
        with pytest.raises(ValueError, match="must be positive"):
            WitnessIdAllocator(buffer_size=0)
        with pytest.raises(ValueError, match="must be positive"):
            WitnessIdAllocator(buffer_size=-1)

    def test_negative_num_ids_raises(self) -> None:
        """allocate(num_ids<0) raises ValueError and leaves the counter untouched."""
        allocator = WitnessIdAllocator(buffer_size=5)
        with pytest.raises(ValueError, match="must be non-negative"):
            allocator.allocate(num_ids=-1)
        assert allocator.counter == 0

    def test_allocate_exceeds_buffer_size_raises(self) -> None:
        """num_ids > buffer_size raises AssertionError."""
        allocator = WitnessIdAllocator(buffer_size=5)
        with pytest.raises(AssertionError, match="exceeds buffer_size"):
            allocator.allocate(num_ids=10)

    def test_consecutive_allocations_stale_ids_evolve(self) -> None:
        """Consecutive allocate calls should produce evolving stale_ids as counter grows."""
        allocator = WitnessIdAllocator(buffer_size=10)

        info1 = allocator.allocate(3)
        stale1 = set(info1.stale_ids)

        info2 = allocator.allocate(3)
        stale2 = set(info2.stale_ids)

        info3 = allocator.allocate(3)
        stale3 = set(info3.stale_ids)

        assert stale1 != stale2 or stale2 != stale3, "stale_ids should evolve across allocations"
        all_ids = set(range(10))
        for stale in [stale1, stale2, stale3]:
            assert stale.issubset(all_ids)


class TestComputeStaleIds:
    """Direct tests for _compute_stale_ids module-level function."""

    def test_counter_zero_returns_empty(self) -> None:
        assert _compute_stale_ids(keep_count=5, counter=0, buffer_size=10) == []

    def test_counter_zero_keep_zero_returns_empty(self) -> None:
        assert _compute_stale_ids(keep_count=0, counter=0, buffer_size=10) == []

    def test_counter_zero_large_buffer_returns_empty(self) -> None:
        assert _compute_stale_ids(keep_count=100, counter=0, buffer_size=1000) == []

    def test_counter_less_than_keep_count_returns_all_unused(self) -> None:
        # counter=3, buffer=10, keep=7 → active=min(7,3,10)=3 → stale=7 slots
        result = _compute_stale_ids(keep_count=7, counter=3, buffer_size=10)
        assert set(result) == {3, 4, 5, 6, 7, 8, 9}

    def test_keep_count_equals_buffer_size_returns_empty(self) -> None:
        # All slots are active
        assert _compute_stale_ids(keep_count=10, counter=15, buffer_size=10) == []

    def test_keep_count_exceeds_buffer_size_returns_empty(self) -> None:
        assert _compute_stale_ids(keep_count=20, counter=15, buffer_size=10) == []

    def test_basic_no_wrap(self) -> None:
        # counter=8, buffer=10, keep=5 → stale=5, head=8 → stale=[8,9,0,1,2]
        result = _compute_stale_ids(keep_count=5, counter=8, buffer_size=10)
        assert result == [8, 9, 0, 1, 2]

    def test_basic_wrap(self) -> None:
        # counter=3, buffer=10, keep=5 → active=min(5,3,10)=3 → stale=7, head=3 → stale=[3,4,5,6,7,8,9]
        result = _compute_stale_ids(keep_count=5, counter=3, buffer_size=10)
        assert result == [3, 4, 5, 6, 7, 8, 9]

    def test_head_at_zero(self) -> None:
        # counter=10, buffer=10, keep=3 → stale=7, head=0 → stale=[0,1,2,3,4,5,6]
        result = _compute_stale_ids(keep_count=3, counter=10, buffer_size=10)
        assert result == [0, 1, 2, 3, 4, 5, 6]

    def test_keep_one(self) -> None:
        # counter=5, buffer=10, keep=1 → stale=9, head=5 → stale=[5,6,7,8,9,0,1,2,3]
        result = _compute_stale_ids(keep_count=1, counter=5, buffer_size=10)
        assert len(result) == 9
        assert 4 not in result

    def test_keep_zero(self) -> None:
        # All slots stale
        result = _compute_stale_ids(keep_count=0, counter=5, buffer_size=10)
        assert len(result) == 10
        assert set(result) == set(range(10))

    def test_stale_and_active_are_disjoint_and_cover_buffer(self) -> None:
        for counter in [0, 1, 5, 10, 13, 20, 100]:
            for keep in [0, 1, 3, 7, 10, 15]:
                stale = _compute_stale_ids(keep_count=keep, counter=counter, buffer_size=10)
                if counter == 0:
                    assert stale == [], "counter=0 → no stale IDs"
                    continue
                active_count = min(keep, counter, 10)
                assert len(stale) == 10 - active_count, f"counter={counter}, keep={keep}"
                assert len(set(stale)) == len(stale), "no duplicates"
                assert all(0 <= x < 10 for x in stale), "all in range"

    def test_buffer_size_one(self) -> None:
        assert _compute_stale_ids(keep_count=1, counter=5, buffer_size=1) == []
        assert _compute_stale_ids(keep_count=0, counter=5, buffer_size=1) == [0]


class TestWitnessIdAllocatorResume:
    def test_resume_continues_allocation_without_reusing_ids(self) -> None:
        """Resuming from a persisted counter issues fresh ids, as if the run never stopped."""
        saved = WitnessIdAllocator(buffer_size=100)
        saved.allocate(10)
        assert saved.counter == 10

        resumed = WitnessIdAllocator(buffer_size=100)
        resumed.resume(saved.counter)
        info = resumed.allocate(5)

        assert info.witness_ids == [10, 11, 12, 13, 14]
        assert resumed.counter == 15

    def test_resume_backwards_is_rejected(self) -> None:
        """resume() must never move the counter backwards."""
        allocator = WitnessIdAllocator(buffer_size=100)
        allocator.allocate(10)

        with pytest.raises(AssertionError):
            allocator.resume(3)

    def test_counter_matches_uninterrupted_run(self) -> None:
        """A save/resume sequence allocates the same ids as one uninterrupted allocator."""
        uninterrupted = WitnessIdAllocator(buffer_size=7)
        expected = [uninterrupted.allocate(3).witness_ids for _ in range(4)]

        first = WitnessIdAllocator(buffer_size=7)
        actual = [first.allocate(3).witness_ids, first.allocate(3).witness_ids]
        second = WitnessIdAllocator(buffer_size=7)
        second.resume(first.counter)
        actual += [second.allocate(3).witness_ids, second.allocate(3).witness_ids]

        assert actual == expected


class TestReadPersistedWitnessCounter:
    def _write_events(self, event_dir, counters) -> None:
        event_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for i, counter in enumerate(counters):
            lines.append(
                json.dumps(
                    {
                        "type": "witness_allocate_id",
                        "timestamp": "2026-06-12T00:00:00Z",
                        "source": {"component": "main"},
                        "rollout_id": i,
                        "attempt": 0,
                        "witness_id_to_sample_index": {},
                        "counter_after": counter,
                    }
                )
            )
        (event_dir / "main.jsonl").write_text("\n".join(lines) + "\n")

    def test_reads_latest_counter_from_events(self, tmp_path) -> None:
        """The max counter_after across allocate events is the resume point."""
        self._write_events(tmp_path / "events", [3, 6, 9])

        assert read_persisted_witness_counter(tmp_path / "events") == 9

    def test_empty_or_missing_dir_resumes_from_zero(self, tmp_path) -> None:
        """A fresh run (no events yet) starts allocation at zero."""
        assert read_persisted_witness_counter(tmp_path / "missing") == 0
