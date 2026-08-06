import asyncio
from types import SimpleNamespace

import pytest
from tests.fast.ray.train import conftest as train_conftest

from miles.ray.specs.train import compute_trainer_spec_name
from miles.ray.train.group import RayTrainGroup
from miles.utils.workers.worker_provider.base import CellInfo, ReconcileFn, StopWatchFn
from miles.utils.workers.worker_provider.ray import RayWorkerProvider

pytestmark = pytest.mark.asyncio

_SPEC_NAME = compute_trainer_spec_name("actor")
_POLL_INTERVAL_SECONDS = 0.01


class _RecordingWorkerProvider(RayWorkerProvider):
    def __init__(self, *, worker_manager_handle: object) -> None:
        super().__init__(worker_manager_handle=worker_manager_handle, poll_interval_seconds=_POLL_INTERVAL_SECONDS)
        self.watch_calls: list[tuple[ReconcileFn, list[str]]] = []
        self.poll_count: int = 0

    async def watch_cells(self, reconcile: ReconcileFn, *, spec_names: list[str]) -> StopWatchFn:
        self.watch_calls.append((reconcile, list(spec_names)))
        return await super().watch_cells(reconcile, spec_names=spec_names)

    async def _poll_once(
        self, reconcile: ReconcileFn, seen_infos: dict[str, CellInfo], *, spec_names: list[str]
    ) -> None:
        self.poll_count += 1
        await super()._poll_once(reconcile, seen_infos=seen_infos, spec_names=spec_names)


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
    monkeypatch.setattr(RayWorkerProvider, "create", lambda: recording_provider)
    return recording_provider


async def _create_group(*, num_cells: int) -> RayTrainGroup:
    train_conftest.fake_worker_manager.num_cells = num_cells
    return await RayTrainGroup.create(
        _make_args(num_cells=num_cells),
        role="actor",
        with_ref=False,
        inference_controller=None,
        rollout_executor=None,
    )


class TestCreate:
    async def test_create_subscribes_reconcile_to_the_trainer_spec(self, provider):
        """create() must watch its own trainer spec with the group's reconcile callback."""
        group = await _create_group(num_cells=2)
        try:
            assert len(provider.watch_calls) == 1
            reconcile, spec_names = provider.watch_calls[0]
            assert reconcile == group._reconcile
            assert spec_names == [_SPEC_NAME]
        finally:
            await group.dispose()

    async def test_create_populates_cells_from_the_initial_sync(self, provider):
        """The initial watch sync must fill in the cells before create() returns."""
        group = await _create_group(num_cells=2)
        try:
            assert sorted(group._cells_by_index) == [0, 1]
            assert [cell.cell_index for cell in group._cells] == [0, 1]
        finally:
            await group.dispose()

    async def test_dispose_stops_the_watch_loop(self, provider):
        """Without dispose() the 5-second poll loop outlives training and keeps logging failures."""
        group = await _create_group(num_cells=1)
        await asyncio.sleep(_POLL_INTERVAL_SECONDS * 5)

        await group.dispose()
        polls_after_dispose: int = provider.poll_count
        await asyncio.sleep(_POLL_INTERVAL_SECONDS * 5)

        assert provider.poll_count == polls_after_dispose
        assert group._watcher_disposer is None

    async def test_dispose_is_idempotent(self, provider):
        """Teardown paths overlap, so a second dispose must not raise."""
        group = await _create_group(num_cells=1)

        await group.dispose()
        await group.dispose()
