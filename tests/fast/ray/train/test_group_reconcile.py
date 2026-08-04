from collections.abc import Awaitable, Callable
from types import SimpleNamespace

import pytest

from miles.ray.specs.train import compute_trainer_pool_id
from miles.ray.train.group import TrainerController
from miles.utils import retry_utils
from miles.utils.workers.worker_provider.base import CellInfo

pytestmark = pytest.mark.asyncio

_POOL_ID = compute_trainer_pool_id("actor")


def _make_controller(*, num_cells: int = 2, indep_dp: bool = False) -> RayTrainGroup:
    group = object.__new__(RayTrainGroup)
    group.args = SimpleNamespace(
        indep_dp=indep_dp,
        actor_num_nodes=1,
        actor_num_gpus_per_node=num_cells,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        train_backend="megatron",
    )
    group._role = "actor"
    group._with_ref = False
    group._with_opd_teacher = False
    group._pool_id = _POOL_ID
    group._rollout_executor = None
    group._health_checker_config = None
    group._health_checker_activeness = True
    group._cells_by_index = {}
    return group


def _make_cell_info(cell_index: int) -> CellInfo:
    return CellInfo(
        cell_id=f"{_POOL_ID}-{cell_index}",
        pool_id=_POOL_ID,
        alive=True,
        worker_names=[f"{_POOL_ID}-{cell_index}-0"],
        workers_hash="pseudo-hash-1",
        meta={"role": "actor"},
    )


class TestReconcile:
    async def test_an_observed_cell_is_added(self):
        """The group learns about its cells from the manager instead of creating them."""
        group = _make_controller()

        await group._reconcile(f"{_POOL_ID}-0", _make_cell_info(0))

        assert [cell.cell_index for cell in group._cells] == [0]

    async def test_reobserving_a_known_cell_keeps_the_same_object(self):
        """Recreating the cell would throw away its state machine and health checker."""
        group = _make_controller()
        await group._reconcile(f"{_POOL_ID}-0", _make_cell_info(0))
        first = group._cells[0]

        await group._reconcile(f"{_POOL_ID}-0", _make_cell_info(0))

        assert group._cells[0] is first

    async def test_cells_are_ordered_by_index_whatever_the_arrival_order(self):
        """Independent DP ranks are derived from position, so order must be stable."""
        group = _make_controller(num_cells=3)

        for cell_index in [2, 0, 1]:
            await group._reconcile(f"{_POOL_ID}-{cell_index}", _make_cell_info(cell_index))

        assert [cell.cell_index for cell in group._cells] == [0, 1, 2]


class _AutoAdvancingClock:
    def __init__(self) -> None:
        self.now: float = 0.0
        self.sleeps: list[float] = []
        self.on_sleep: Callable[[int], Awaitable[None]] | None = None

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        if self.on_sleep is not None:
            await self.on_sleep(len(self.sleeps))


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> _AutoAdvancingClock:
    clock = _AutoAdvancingClock()
    monkeypatch.setattr(retry_utils, "time", SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr(retry_utils, "asyncio", SimpleNamespace(sleep=clock.sleep))
    return clock


class TestWaitExpectedNumCells:
    async def test_waiting_keeps_polling_until_the_late_cells_are_observed(self, fake_clock: _AutoAdvancingClock):
        """Training must not start against half a pool, so the wait retries until the missing cells arrive."""
        group = _make_controller(num_cells=4, indep_dp=True)
        await group._reconcile(f"{_POOL_ID}-0", _make_cell_info(0))

        async def _add_remaining_cells_on_the_second_sleep(sleep_count: int) -> None:
            if sleep_count == 2:
                for cell_index in range(1, 4):
                    await group._reconcile(f"{_POOL_ID}-{cell_index}", _make_cell_info(cell_index))

        fake_clock.on_sleep = _add_remaining_cells_on_the_second_sleep

        await group._wait_expected_num_cells(timeout=600.0)

        assert len(fake_clock.sleeps) == 2
        assert [cell.cell_index for cell in group._cells] == [0, 1, 2, 3]

    async def test_waiting_returns_immediately_when_every_cell_is_already_observed(
        self, fake_clock: _AutoAdvancingClock
    ):
        """A complete pool must not cost a single retry sleep."""
        group = _make_controller(num_cells=4, indep_dp=True)
        for cell_index in range(4):
            await group._reconcile(f"{_POOL_ID}-{cell_index}", _make_cell_info(cell_index))

        await group._wait_expected_num_cells(timeout=600.0)

        assert fake_clock.sleeps == []

    async def test_waiting_gives_up_when_cells_never_appear(self, fake_clock: _AutoAdvancingClock):
        """A silent hang here would look like a stuck first step, so the wait retries and then times out."""
        group = _make_controller(num_cells=4, indep_dp=True)

        with pytest.raises(TimeoutError):
            await group._wait_expected_num_cells(timeout=10.0)

        assert len(fake_clock.sleeps) >= 2
        assert fake_clock.now <= 10.0
