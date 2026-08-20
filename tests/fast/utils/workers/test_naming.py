import pytest

from miles.utils.workers.naming import (
    NAME_INDEX_PAD_WIDTH,
    _worker_name_of_cell,
    cell_id_of_worker,
    compute_cell_id,
    compute_worker_name,
    format_name_index,
    parse_cell_id,
    parse_worker_name,
)


class TestFormatNameIndex:
    def test_pads_to_the_declared_width(self):
        """Every index is rendered at the same width, so names sort lexicographically by index."""
        assert format_name_index(3) == "00003"
        assert len(format_name_index(0)) == NAME_INDEX_PAD_WIDTH

    def test_accepts_the_largest_five_digit_index(self):
        """The largest representable index must retain the fixed-width naming contract."""
        assert format_name_index(99999) == "99999"

    @pytest.mark.parametrize("index", [-1, 100000])
    def test_rejects_an_index_outside_the_five_digit_range(self, index: int):
        """Every emitted index must fit the fixed-width range used for sortable names."""
        with pytest.raises(AssertionError, match="name index must be"):
            format_name_index(index)


class TestComputeCellId:
    def test_pads_the_cell_index(self):
        """A cell id carries a padded index so a pool's cell ids sort in index order."""
        assert compute_cell_id(pool_id="trainer-engine-actor", cell_index=3) == "trainer-engine-actor-00003"

    def test_orders_cell_ids_lexicographically(self):
        """Sorted cell ids of one pool must run 0, 1, ..., 10 rather than 0, 1, 10, 2."""
        cell_ids = [compute_cell_id(pool_id="engine", cell_index=index) for index in [10, 2, 1]]

        assert sorted(cell_ids) == [compute_cell_id(pool_id="engine", cell_index=index) for index in [1, 2, 10]]


class TestComputeWorkerName:
    def test_pads_both_indices(self):
        """A worker name pads the cell index and the in-cell index alike."""
        assert compute_worker_name(pool_id="engine", cell_index=3, worker_in_cell_index=7) == "engine-00003-00007"

    def test_defaults_to_the_single_worker_of_the_single_cell(self):
        """A one-worker pool still gets padded indices from the defaults."""
        assert compute_worker_name(pool_id="rollout-executor") == "rollout-executor-00000-00000"


class TestParseCellId:
    def test_round_trips_a_computed_cell_id(self):
        """A cell id produced here parses back to the values it was built from."""
        cell_id = compute_cell_id(pool_id="trainer-pool-a", cell_index=12)

        assert parse_cell_id(cell_id) == ("trainer-pool-a", 12)


class TestParseWorkerName:
    def test_preserves_a_hyphenated_pool_id(self):
        """Only the last two hyphens separate indices, so a hyphenated pool id survives intact."""
        assert parse_worker_name("trainer-pool-a-00003-00007") == ("trainer-pool-a", 3, 7)

    def test_returns_integer_indices(self):
        """Both index suffixes come back as integers, not as the strings they were parsed from."""
        pool_id, cell_index, worker_in_cell_index = parse_worker_name("rollout-00012-00005")

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


class TestCellIdOfWorker:
    def test_names_the_cell_the_worker_belongs_to(self):
        """The cell of a worker is the padded cell id, whatever the worker name's own padding."""
        assert cell_id_of_worker("engine-00003-00007") == "engine-00003"


class TestWorkerNameOfCell:
    def test_round_trips_with_cell_id_of_worker(self):
        """Naming a worker of a cell and asking which cell it belongs to are inverses."""
        cell_id = compute_cell_id(pool_id="engine", cell_index=3)

        assert cell_id_of_worker(_worker_name_of_cell(cell_id, worker_in_cell_index=7)) == cell_id
