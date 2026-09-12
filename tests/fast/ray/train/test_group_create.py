import asyncio
from types import SimpleNamespace

import pytest
from tests.fast.ray.train import conftest as train_conftest

from miles.ray.specs.train import compute_trainer_pool_id
from miles.ray.train.group import TrainerController
from miles.utils.workers.worker_provider.base import CellInfo, CellReconcileFn, StopWatchFn
from miles.utils.workers.worker_provider.ray import RayWorkerProvider

pytestmark = pytest.mark.asyncio

_POOL_ID = compute_trainer_pool_id("actor")
_POLL_INTERVAL_SECONDS = 0.01


class _RecordingWorkerProvider(RayWorkerProvider):
    def __init__(self, *, worker_manager_handle: object, pool_ids: list[str] | None = None) -> None:
        super().__init__(
            worker_manager_handle=worker_manager_handle,
            pool_ids=pool_ids,
            poll_interval_seconds=_POLL_INTERVAL_SECONDS,
        )
        self.watch_calls: list[tuple[CellReconcileFn, list[str]]] = []
        self.poll_count: int = 0

    async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
        self.watch_calls.append((reconcile, list(self._watched_pool_ids())))
        return await super().watch_cells(reconcile)

    async def _list_alive_cells(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        self.poll_count += 1
        return await super()._list_alive_cells(pool_ids=pool_ids)


def _make_args(*, num_cells: int) -> SimpleNamespace:
    return SimpleNamespace(
        indep_dp=True,
        enable_witness=False,
        witness_buffer_size=100,
        save_debug_event_data=None,
        trainer_heartbeat_checker_interval=10.0,
        trainer_heartbeat_checker_timeout=10.0,
        trainer_heartbeat_checker_first_wait=300.0,
        trainer_heartbeat_checker_failure_threshold=3,
        ci_ft_test_actions=None,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        actor_num_nodes=1,
        actor_num_gpus_per_node=num_cells,
    )


@pytest.fixture
def provider(monkeypatch) -> _RecordingWorkerProvider:
    recording_provider = _RecordingWorkerProvider(worker_manager_handle=train_conftest.fake_worker_manager)

    def _create(*, pool_ids: list[str] | None = None) -> _RecordingWorkerProvider:
        recording_provider._pool_ids = pool_ids
        return recording_provider

    monkeypatch.setattr(RayWorkerProvider, "create", _create)
    return recording_provider


async def _create_controller(*, num_cells: int) -> TrainerController:
    train_conftest.fake_worker_manager.num_cells = num_cells
    controller = TrainerController(
        _make_args(num_cells=num_cells),
        trainer_id="actor",
        role="actor",
        with_ref=False,
        inference_controller=None,
    )
    await controller.init()
    return controller


class TestCreate:
    async def test_create_subscribes_reconcile_to_the_trainer_spec(self, provider):
        """create() must watch its own trainer spec with the controller's reconcile callback."""
        controller = await _create_controller(num_cells=2)
        try:
            assert len(provider.watch_calls) == 1
            reconcile, pool_ids = provider.watch_calls[0]
            assert reconcile == controller._reconcile
            assert pool_ids == [_POOL_ID]
        finally:
            await controller.dispose()

    async def test_create_populates_cells_from_the_initial_sync(self, provider):
        """The initial watch sync must fill in the cells before create() returns."""
        controller = await _create_controller(num_cells=2)
        try:
            assert sorted(cell.cell_index for cell in controller._cells_by_id.values()) == [0, 1]
            assert [cell.cell_index for cell in controller._cells] == [0, 1]
        finally:
            await controller.dispose()

    async def test_dispose_stops_the_watch_loop(self, provider):
        """Without dispose() the 5-second poll loop outlives training and keeps logging failures."""
        controller = await _create_controller(num_cells=1)
        await asyncio.sleep(_POLL_INTERVAL_SECONDS * 5)

        await controller.dispose()
        polls_after_dispose: int = provider.poll_count
        await asyncio.sleep(_POLL_INTERVAL_SECONDS * 5)

        assert provider.poll_count == polls_after_dispose
        assert controller._watcher_disposer is None

    async def test_dispose_is_idempotent(self, provider):
        """Teardown paths overlap, so a second dispose must not raise."""
        controller = await _create_controller(num_cells=1)

        await controller.dispose()
        await controller.dispose()
