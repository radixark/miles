import pytest

from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_provider.utils import apply_cell_observation, single_worker_name_of

pytestmark = pytest.mark.asyncio

_CELL_ID = "spec-0"


def _make_cell_info(workers_hash: str = "hash-1") -> CellInfo:
    return CellInfo(
        cell_id=_CELL_ID,
        pool_id="spec",
        alive=True,
        worker_names=["spec-0-0"],
        workers_hash=workers_hash,
        meta={},
    )


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def add(self, cell_id: str, observed: CellInfo) -> None:
        self.calls.append(("add", cell_id, observed.workers_hash))

    async def remove(self, cell_id: str) -> None:
        self.calls.append(("remove", cell_id))


class TestApplyCellObservation:
    async def test_an_unknown_observed_cell_is_added(self):
        """A newly reported cell must enter the bookkeeping."""
        recorder = _Recorder()

        await apply_cell_observation(
            cell_id=_CELL_ID,
            observed=_make_cell_info(),
            actual_workers_hash=None,
            add=recorder.add,
            remove=recorder.remove,
        )

        assert recorder.calls == [("add", _CELL_ID, "hash-1")]

    async def test_a_disappeared_known_cell_is_removed(self):
        """A cell the provider stops reporting must leave the bookkeeping."""
        recorder = _Recorder()

        await apply_cell_observation(
            cell_id=_CELL_ID, observed=None, actual_workers_hash="hash-1", add=recorder.add, remove=recorder.remove
        )

        assert recorder.calls == [("remove", _CELL_ID)]

    async def test_a_disappeared_unknown_cell_is_ignored(self):
        """Removing what was never added would raise in the callbacks."""
        recorder = _Recorder()

        await apply_cell_observation(
            cell_id=_CELL_ID, observed=None, actual_workers_hash=None, add=recorder.add, remove=recorder.remove
        )

        assert recorder.calls == []

    async def test_a_changed_workers_hash_replaces_the_cell(self):
        """A new worker generation must not be served through the old cell object."""
        recorder = _Recorder()

        await apply_cell_observation(
            cell_id=_CELL_ID,
            observed=_make_cell_info("hash-2"),
            actual_workers_hash="hash-1",
            add=recorder.add,
            remove=recorder.remove,
        )

        assert recorder.calls == [("remove", _CELL_ID), ("add", _CELL_ID, "hash-2")]

    async def test_an_unchanged_workers_hash_keeps_the_cell(self):
        """Recreating the cell would throw away its accumulated state."""
        recorder = _Recorder()

        await apply_cell_observation(
            cell_id=_CELL_ID,
            observed=_make_cell_info(),
            actual_workers_hash="hash-1",
            add=recorder.add,
            remove=recorder.remove,
        )

        assert recorder.calls == []


def _single_cell_infos(cell_ids: list[str], *, worker_names: list[str]) -> dict[str, CellInfo]:
    return {
        cell_id: CellInfo(
            cell_id=cell_id,
            pool_id="a-pool",
            alive=True,
            worker_names=list(worker_names),
            workers_hash="hash",
            meta={},
        )
        for cell_id in cell_ids
    }


class TestSingleWorkerNameOf:
    async def test_it_answers_the_one_worker_of_the_one_cell(self):
        """A caller that must not mint names asks the backend which worker its single-cell pool deploys."""
        infos = _single_cell_infos(["a-cell"], worker_names=["a-worker"])

        assert single_worker_name_of(infos, pool_id="a-pool") == "a-worker"

    async def test_a_pool_with_several_cells_is_rejected(self):
        """Picking one of several cells would be numbering them, so the ambiguity is refused instead."""
        infos = _single_cell_infos(["a-cell", "b-cell"], worker_names=["a-worker"])

        with pytest.raises(AssertionError, match="single-cell pool"):
            single_worker_name_of(infos, pool_id="a-pool")

    async def test_a_cell_with_several_workers_is_rejected(self):
        """The same ambiguity one layer down: nothing may guess which rank of the cell is meant."""
        infos = _single_cell_infos(["a-cell"], worker_names=["a-worker", "another-worker"])

        with pytest.raises(AssertionError, match="single-worker cell"):
            single_worker_name_of(infos, pool_id="a-pool")
