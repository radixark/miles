from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from miles.ray.train.cell import TrainerCell
from miles.ray.train.group import TrainerController
from miles.utils.workers.worker_handle import WorkerUnreachableError


if TYPE_CHECKING:
    from tests.fast.ray.train.conftest import PendingTrainerWorker


class TestCellRemoval:
    async def test_removing_a_cell_unblocks_its_pending_call_without_killing_workers(
        self, pending_trainer_cell: TrainerCell, pending_trainer_worker: PendingTrainerWorker
    ) -> None:
        """A deleted Pod cannot keep the controller waiting on its old RPC endpoint."""
        controller = object.__new__(TrainerController)
        controller._cells_by_id = {pending_trainer_cell.cell_id: pending_trainer_cell}
        task = asyncio.create_task(pending_trainer_cell.execute("train"))
        await pending_trainer_worker.started.wait()

        await controller._remove_cell(pending_trainer_cell.cell_id)

        with pytest.raises(WorkerUnreachableError):
            await asyncio.wait_for(task, timeout=1)
        assert pending_trainer_worker.cancelled.is_set()
        assert pending_trainer_worker.kill_count == 0
        assert controller._cells_by_id == {}

    async def test_a_removed_cell_rejects_new_calls(
        self, pending_trainer_cell: TrainerCell, pending_trainer_worker: PendingTrainerWorker
    ) -> None:
        """Stale snapshots cannot submit another RPC after their cell disappears."""
        controller = object.__new__(TrainerController)
        controller._cells_by_id = {pending_trainer_cell.cell_id: pending_trainer_cell}
        await controller._remove_cell(pending_trainer_cell.cell_id)

        with pytest.raises(WorkerUnreachableError):
            await asyncio.wait_for(pending_trainer_cell.execute("train"), timeout=1)
        assert not pending_trainer_worker.started.is_set()
        assert pending_trainer_worker.kill_count == 0

    async def test_cancelling_the_caller_preserves_cancellation(
        self, pending_trainer_cell: TrainerCell, pending_trainer_worker: PendingTrainerWorker
    ) -> None:
        """Caller cancellation stays distinct from a disappeared cell."""
        task = asyncio.create_task(pending_trainer_cell.execute("train"))
        await pending_trainer_worker.started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert pending_trainer_worker.cancelled.is_set()
        assert pending_trainer_worker.kill_count == 0

    async def test_argument_failure_cancels_calls_already_submitted(
        self,
        pending_trainer_cell: TrainerCell,
        pending_trainer_worker: PendingTrainerWorker,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Preparing a later worker call cannot leave an earlier RPC pending."""
        submitted = asyncio.get_running_loop().create_future()
        monkeypatch.setattr(pending_trainer_worker, "train", lambda: submitted)
        pending_trainer_cell._get_worker_handles().append(pending_trainer_worker)

        def compute_kwargs(index: int) -> dict:
            if index == 1:
                raise ValueError("Invalid worker arguments")
            return {}

        with pytest.raises(ValueError, match="Invalid worker arguments"):
            await pending_trainer_cell._execute_raw("train", compute_kwargs=compute_kwargs, kill_on_failure=False)

        assert submitted.cancelled()
        assert pending_trainer_worker.kill_count == 0
