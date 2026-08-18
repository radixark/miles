import pytest

from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_provider.utils import apply_cell_observation, build_rpc_handle_of_worker_info
from miles.utils.workers.worker_spec import HostAndPort

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

    async def test_a_failed_remove_aborts_replacement_and_propagates(self):
        """The replacement must not be installed while the old cell is still registered."""
        recorder = _Recorder()

        async def failing_remove(cell_id: str) -> None:
            await recorder.remove(cell_id)
            raise RuntimeError("remove boom")

        with pytest.raises(RuntimeError, match="remove boom"):
            await apply_cell_observation(
                cell_id=_CELL_ID,
                observed=_make_cell_info("hash-2"),
                actual_workers_hash="hash-1",
                add=recorder.add,
                remove=failing_remove,
            )

        assert recorder.calls == [("remove", _CELL_ID)]

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


class _DemoWorker:
    def report(self) -> str:
        return "ok"


def _info(*, generation: int, port: int = 15000) -> WorkerInfo:
    return WorkerInfo(
        name="trainer-engine-actor-0-0",
        generation=generation,
        self_addrs={"rpc": HostAndPort(host="10.0.0.7", port=port)},
        gpu_ids=[],
        worker_class=f"{__name__}._DemoWorker",
    )


class TestBuildRpcHandleFor:
    def test_the_handle_points_at_the_address_the_worker_serves_on(self):
        """A handle aimed anywhere else silently drives another worker on the same node."""
        handle = build_rpc_handle_of_worker_info(_info(generation=1, port=15001))

        assert handle._transport._server_url == "http://10.0.0.7:15001"
