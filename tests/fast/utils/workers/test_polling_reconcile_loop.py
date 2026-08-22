from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import pytest

from miles.utils.workers.polling_reconcile_loop import PollingReconcileLoop
from miles.utils.workers.worker_provider.base import CellInfo

POLL_INTERVAL_SECONDS = 0.001


def _cell_info(cell_id: str, *, workers_hash: str = "hash-0") -> CellInfo:
    return CellInfo(
        cell_id=cell_id,
        pool_id="inference-engine-0-0",
        alive=True,
        worker_names=[f"{cell_id}-0"],
        workers_hash=workers_hash,
        meta={},
    )


@dataclass
class _FakeLister:
    answers: list[Any]
    calls: int = 0

    async def __call__(self) -> dict[str, CellInfo]:
        self.calls += 1
        answer = self.answers[min(self.calls - 1, len(self.answers) - 1)]
        if isinstance(answer, Exception):
            raise answer
        return answer


class _RecordingReconciler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, CellInfo | None]] = []

    async def __call__(self, cell_id: str, info: CellInfo | None) -> None:
        self.calls.append((cell_id, info))


class _FailingOnceReconciler(_RecordingReconciler):
    def __init__(self, *, failing_cell_id: str) -> None:
        super().__init__()
        self._failing_cell_id: str | None = failing_cell_id

    async def __call__(self, cell_id: str, info: CellInfo | None) -> None:
        if cell_id == self._failing_cell_id:
            self._failing_cell_id = None
            raise RuntimeError("reconcile failed")
        await super().__call__(cell_id, info)


class _AlwaysFailingReconciler(_RecordingReconciler):
    async def __call__(self, cell_id: str, info: CellInfo | None) -> None:
        await super().__call__(cell_id, info)
        raise RuntimeError("reconcile rejected the cell")


def _make_loop(*answers: Any) -> tuple[PollingReconcileLoop, _FakeLister]:
    lister = _FakeLister(answers=list(answers))
    loop = PollingReconcileLoop(list_cells=lister, poll_interval_seconds=POLL_INTERVAL_SECONDS)
    return loop, lister


async def _wait_until(predicate, *, timeout_seconds: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting for the reconcile loop"
        await asyncio.sleep(0.001)


class TestPollingReconcileLoopInitialSync:
    async def test_every_initial_cell_is_reconciled_before_start_returns(self):
        """Callers may assume the whole listing is observed once start returns."""
        infos = {"cell-a": _cell_info("cell-a"), "cell-b": _cell_info("cell-b")}
        loop, _lister = _make_loop(infos)
        reconciler = _RecordingReconciler()

        stop = await loop.start(reconciler)
        try:
            assert reconciler.calls == [("cell-a", infos["cell-a"]), ("cell-b", infos["cell-b"])]
        finally:
            await stop()

    async def test_a_failing_initial_listing_propagates_instead_of_starting_the_loop(self):
        """A source we never managed to read must not look like an empty source."""
        loop, lister = _make_loop(RuntimeError("source unreachable"))

        with pytest.raises(RuntimeError, match="source unreachable"):
            await loop.start(_RecordingReconciler())

        await asyncio.sleep(0.02)

        assert lister.calls == 1

    async def test_an_initial_reconcile_failure_prevents_the_loop_from_starting(self):
        """A caller that never learned about the initial cells must not be left with a live loop."""
        loop, lister = _make_loop({"cell-a": _cell_info("cell-a")})

        with pytest.raises(RuntimeError, match="reconcile rejected the cell"):
            await loop.start(_AlwaysFailingReconciler())

        await asyncio.sleep(0.02)

        assert lister.calls == 1


class TestPollingReconcileLoopPolling:
    async def test_an_unchanged_cell_is_not_reconciled_again(self):
        """Re-reconciling every tick would restart cells every interval."""
        info = _cell_info("cell-a")
        loop, lister = _make_loop({"cell-a": info})
        reconciler = _RecordingReconciler()

        stop = await loop.start(reconciler)
        try:
            await _wait_until(lambda: lister.calls >= 3)
            assert reconciler.calls == [("cell-a", info)]
        finally:
            await stop()

    async def test_a_cell_appearing_later_is_reconciled_once(self):
        """A cell created after the initial sync must still reach the consumer, and only once."""
        info = _cell_info("cell-a")
        loop, lister = _make_loop({}, {"cell-a": info})
        reconciler = _RecordingReconciler()

        stop = await loop.start(reconciler)
        try:
            await _wait_until(lambda: lister.calls >= 4)
            assert reconciler.calls == [("cell-a", info)]
        finally:
            await stop()

    async def test_a_disappeared_cell_is_reported_as_gone_exactly_once(self):
        """A vanished cell must be delivered as None, and must not be re-reported afterwards."""
        info = _cell_info("cell-a")
        loop, lister = _make_loop({"cell-a": info}, {})
        reconciler = _RecordingReconciler()

        stop = await loop.start(reconciler)
        try:
            await _wait_until(lambda: lister.calls >= 4)
            assert reconciler.calls == [("cell-a", info), ("cell-a", None)]
        finally:
            await stop()

    async def test_a_changed_cell_info_is_reconciled_again(self):
        """A replaced cell keeps its id, so only its info can reveal the change."""
        first = _cell_info("cell-a", workers_hash="hash-0")
        second = _cell_info("cell-a", workers_hash="hash-1")
        loop, _lister = _make_loop({"cell-a": first}, {"cell-a": second})
        reconciler = _RecordingReconciler()

        stop = await loop.start(reconciler)
        try:
            await _wait_until(lambda: len(reconciler.calls) >= 2)
            assert reconciler.calls[:2] == [("cell-a", first), ("cell-a", second)]
        finally:
            await stop()

    async def test_a_failing_listing_is_retried_instead_of_killing_the_loop(self):
        """One unreachable listing call must not silently end the observation."""
        info = _cell_info("cell-a")
        loop, _lister = _make_loop({}, RuntimeError("transient"), {"cell-a": info})
        reconciler = _RecordingReconciler()

        stop = await loop.start(reconciler)
        try:
            await _wait_until(lambda: reconciler.calls == [("cell-a", info)])
        finally:
            await stop()

    async def test_a_failed_reconcile_is_retried_before_the_cell_is_marked_seen(self):
        """A cell whose reconcile raised must be delivered again instead of being silently forgotten."""
        info = _cell_info("cell-a")
        loop, _lister = _make_loop({}, {"cell-a": info})
        reconciler = _FailingOnceReconciler(failing_cell_id="cell-a")

        stop = await loop.start(reconciler)
        try:
            await _wait_until(lambda: reconciler.calls == [("cell-a", info)])
        finally:
            await stop()

    async def test_a_cell_added_by_a_partially_failed_tick_can_still_disappear(self):
        """A reconcile that raises mid-tick must not lose the bookkeeping of the cells already delivered."""
        info_a = _cell_info("cell-a")
        info_b = _cell_info("cell-b")
        loop, _lister = _make_loop({}, {"cell-a": info_a, "cell-b": info_b}, {"cell-b": info_b})
        reconciler = _FailingOnceReconciler(failing_cell_id="cell-b")

        stop = await loop.start(reconciler)
        try:
            await _wait_until(lambda: ("cell-a", None) in reconciler.calls)
            assert ("cell-a", info_a) in reconciler.calls
        finally:
            await stop()


class TestPollingReconcileLoopStop:
    async def test_stopping_ends_the_polling(self):
        """The returned stop function must actually stop the loop, not just detach from it."""
        loop, lister = _make_loop({})

        stop = await loop.start(_RecordingReconciler())
        await _wait_until(lambda: lister.calls >= 2)
        await stop()
        settled = lister.calls

        await asyncio.sleep(0.02)

        assert lister.calls == settled

    async def test_stopping_twice_is_not_an_error(self):
        """A provider disposed twice is ordinary teardown, and raising there would mask the first failure."""
        loop, _lister = _make_loop({})

        stop = await loop.start(_RecordingReconciler())
        await stop()
        await stop()

    async def test_a_failed_initial_sync_leaves_no_polling_behind(self):
        """start() raises, so the caller never receives a stop function and could not stop a task it left running."""
        loop, lister = _make_loop(RuntimeError("list failed"))

        with pytest.raises(RuntimeError, match="list failed"):
            await loop.start(_RecordingReconciler())
        settled = lister.calls

        await asyncio.sleep(0.02)

        assert lister.calls == settled


class TestTwoLoopsOverOneProvider:
    async def test_each_loop_keeps_its_own_view_of_what_it_has_seen(self):
        """A shared seen-set would make the second loop skip the cells the first one already reported."""
        lister = _FakeLister(answers=[{"cell-0": _cell_info("cell-0")}])
        first = PollingReconcileLoop(list_cells=lister, poll_interval_seconds=POLL_INTERVAL_SECONDS)
        second = PollingReconcileLoop(list_cells=lister, poll_interval_seconds=POLL_INTERVAL_SECONDS)

        first_reconciler, second_reconciler = _RecordingReconciler(), _RecordingReconciler()
        first_stop = await first.start(first_reconciler)
        second_stop = await second.start(second_reconciler)
        await first_stop()
        await second_stop()

        assert [cell_id for cell_id, _ in first_reconciler.calls] == ["cell-0"]
        assert [cell_id for cell_id, _ in second_reconciler.calls] == ["cell-0"]
