import pytest

from miles.utils.workers.naming import compute_worker_name, parse_worker_name


class TestParseWorkerName:
    def test_preserves_a_hyphenated_pool_id(self):
        """Only the last two hyphens separate indices, so a hyphenated pool id survives intact."""
        assert parse_worker_name("trainer-pool-a-3-7") == ("trainer-pool-a", 3, 7)

    def test_returns_integer_indices(self):
        """Both index suffixes come back as integers, not as the strings they were parsed from."""
        pool_id, cell_index, worker_in_cell_index = parse_worker_name("rollout-12-5")

        assert pool_id == "rollout"
        assert (cell_index, worker_in_cell_index) == (12, 5)
        assert isinstance(cell_index, int) and isinstance(worker_in_cell_index, int)

    def test_round_trips_a_computed_worker_name(self):
        """A name produced by compute_worker_name parses back to the values it was built from."""
        worker_name = compute_worker_name(pool_id="trainer-pool-a", cell_index=2, worker_in_cell_index=4)

        assert parse_worker_name(worker_name) == ("trainer-pool-a", 2, 4)

    @pytest.mark.parametrize("worker_name", ["trainer", "trainer-1"])
    def test_rejects_missing_index_suffixes(self, worker_name: str):
        """A name without both index suffixes cannot be unpacked into pool, cell, and worker."""
        with pytest.raises(ValueError):
            parse_worker_name(worker_name)

    @pytest.mark.parametrize("worker_name", ["trainer-abc-7", "trainer-3-abc"])
    def test_rejects_non_numeric_index_suffixes(self, worker_name: str):
        """A non-numeric cell or worker suffix is rejected instead of flowing on as a string."""
        with pytest.raises(ValueError):
            parse_worker_name(worker_name)
