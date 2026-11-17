import asyncio

import pytest

from miles.utils.workers.cell_handle import GatherCellHandle


class _FakeWorker:
    def __init__(self, *, fail_after: float | None = None) -> None:
        self._fail_after = fail_after
        self.outcome: str | None = None

    async def work(self, *, value: int) -> int:
        try:
            if self._fail_after is not None:
                await asyncio.sleep(self._fail_after)
                raise RuntimeError(f"worker {value} failed")
            await asyncio.sleep(1.0)
            self.outcome = "finished"
            return value * 10
        except asyncio.CancelledError:
            self.outcome = "cancelled"
            raise


class TestGatherCellHandle:
    async def test_the_results_come_back_in_worker_order(self):
        workers = [_FakeWorker(), _FakeWorker()]

        results = await GatherCellHandle(workers).call_each("work", [{"value": 1}, {"value": 2}])

        assert results == [10, 20]

    async def test_a_failing_worker_cancels_its_siblings_and_raises_its_own_error(self):
        """Siblings left running would keep polling their workers into the cell's next call."""
        failing, slow_a, slow_b = _FakeWorker(fail_after=0.01), _FakeWorker(), _FakeWorker()

        with pytest.raises(RuntimeError, match="worker 0 failed"):
            await GatherCellHandle([failing, slow_a, slow_b]).call_each(
                "work", [{"value": 0}, {"value": 1}, {"value": 2}]
            )

        assert (slow_a.outcome, slow_b.outcome) == ("cancelled", "cancelled")

    async def test_one_set_of_arguments_is_needed_per_worker(self):
        with pytest.raises(AssertionError, match="1 sets of arguments for the 2 workers"):
            await GatherCellHandle([_FakeWorker(), _FakeWorker()]).call_each("work", [{"value": 1}])
