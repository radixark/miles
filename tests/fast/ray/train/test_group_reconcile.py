from collections.abc import Awaitable, Callable
from types import SimpleNamespace

import pytest

from miles.ray.specs.train import compute_trainer_pool_id
from miles.ray.train.group import TrainerController
from miles.utils import retry_utils
from miles.utils.ft_utils.api_server.models import CellStatus, TriState
from miles.utils.workers.worker_provider.base import CellInfo

pytestmark = pytest.mark.asyncio

_POOL_ID = compute_trainer_pool_id("actor")


def _make_controller(*, num_cells: int = 2, indep_dp: bool = False) -> TrainerController:
    group = object.__new__(TrainerController)
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
    group._health_checker_config = None
    group._health_checker_activeness = True
    group._cells_by_id = {}
    return group


def _make_cell_info(cell_index: int, *, workers_hash: str = "pseudo-hash-1") -> CellInfo:
    return CellInfo(
        cell_id=f"{_POOL_ID}-{cell_index}",
        pool_id=_POOL_ID,
        alive=True,
        worker_names=[f"{_POOL_ID}-{cell_index}-0"],
        workers_hash=workers_hash,
        meta={"role": "actor", "cell_index": cell_index},
    )


class TestReconcile:
    async def test_an_observed_cell_is_added(self):
        """The group learns about its cells from the manager instead of creating them."""
        group = _make_controller()

        await group._reconcile(f"{_POOL_ID}-0", _make_cell_info(0))

        assert [cell.cell_index for cell in group._cells] == [0]

    async def test_a_disappeared_cell_is_dropped(self):
        """A cell the manager no longer reports must stop being trained."""
        group = _make_controller()
        await group._reconcile(f"{_POOL_ID}-1", _make_cell_info(1))

        await group._reconcile(f"{_POOL_ID}-1", None)

        assert group._cells == []

    async def test_reobserving_a_known_cell_keeps_the_same_object(self):
        """Recreating the cell would throw away its state machine and health checker."""
        group = _make_controller()
        await group._reconcile(f"{_POOL_ID}-0", _make_cell_info(0))
        first = group._cells[0]

        await group._reconcile(f"{_POOL_ID}-0", _make_cell_info(0))

        assert group._cells[0] is first

    async def test_a_relaunched_cell_is_replaced(self):
        """A new generation hands out new actor handles, so keeping the old object would use dead ones."""
        group = _make_controller()
        await group._reconcile(f"{_POOL_ID}-0", _make_cell_info(0))
        first = group._cells[0]

        await group._reconcile(f"{_POOL_ID}-0", _make_cell_info(0, workers_hash="pseudo-hash-2"))

        assert group._cells[0] is not first
        assert group._cells[0].workers_hash == "pseudo-hash-2"

    async def test_a_dropped_cell_has_its_health_checker_stopped(self):
        """A leaked health checker keeps heartbeating a dead actor and logs a stacktrace every interval."""
        group = _make_controller()
        await group._reconcile(f"{_POOL_ID}-1", _make_cell_info(1))
        health_checker = group._cells[0].health_checker

        await group._reconcile(f"{_POOL_ID}-1", None)

        assert health_checker.stopped

    async def test_a_replaced_cell_has_its_old_health_checker_stopped(self):
        """The replace path removes before adding, so the superseded checker must be stopped too."""
        group = _make_controller()
        await group._reconcile(f"{_POOL_ID}-0", _make_cell_info(0))
        old_health_checker = group._cells[0].health_checker

        await group._reconcile(f"{_POOL_ID}-0", _make_cell_info(0, workers_hash="pseudo-hash-2"))

        assert old_health_checker.stopped
        assert not group._cells[0].health_checker.stopped

    async def test_cells_are_ordered_by_index_whatever_the_arrival_order(self):
        """Independent DP ranks are derived from position, so order must be stable."""
        group = _make_controller(num_cells=3)

        for cell_index in [2, 0, 1]:
            await group._reconcile(f"{_POOL_ID}-{cell_index}", _make_cell_info(cell_index))

        assert [cell.cell_index for cell in group._cells] == [0, 1, 2]


def _healthy_condition(status: CellStatus) -> tuple[TriState, str | None]:
    [healthy] = [condition for condition in status.conditions if condition.type == "Healthy"]
    return healthy.status, healthy.reason


class TestPublicCellInventory:
    async def test_the_inventory_and_statuses_follow_the_reconciled_cells(self):
        """The FT controller heals by these names and statuses, so a stale or mis-ordered view heals the wrong cell."""
        group = _make_controller(num_cells=3, indep_dp=True)
        for cell_index in [2, 0, 1]:
            await group._reconcile(f"{_POOL_ID}-{cell_index}", _make_cell_info(cell_index))
        group._cells[1]._mark_as_errored()

        assert group.pool_id == _POOL_ID
        assert group.expected_num_cells == 3
        assert group.num_cells == 3
        assert group.cell_ids == [f"{_POOL_ID}-{cell_index}" for cell_index in range(3)]

        statuses = group.get_cell_statuses()
        assert sorted(statuses) == group.cell_ids
        assert [status.phase for status in statuses.values()] == ["Running"] * 3
        assert _healthy_condition(statuses[f"{_POOL_ID}-0"]) == (TriState.TRUE, None)
        assert _healthy_condition(statuses[f"{_POOL_ID}-1"]) == (TriState.FALSE, "ExecutionErrored")
        assert _healthy_condition(statuses[f"{_POOL_ID}-2"]) == (TriState.TRUE, None)

    async def test_each_status_carries_the_generation_it_describes(self):
        """The api server joins this with a separately polled cell listing, so an unstamped status can mislead."""
        group = _make_controller(num_cells=1, indep_dp=True)
        await group._reconcile(f"{_POOL_ID}-0", _make_cell_info(0, workers_hash="hash-9"))

        statuses = await group.get_cell_statuses()

        assert statuses[f"{_POOL_ID}-0"].workers_hash == "hash-9"

    async def test_each_cell_is_stamped_with_its_own_generation(self):
        """One cell relaunches without the others, so a stamp taken from a neighbour would clear the wrong verdict."""
        group = _make_controller(num_cells=2, indep_dp=True)
        await group._reconcile(f"{_POOL_ID}-0", _make_cell_info(0, workers_hash="hash-9"))
        await group._reconcile(f"{_POOL_ID}-1", _make_cell_info(1, workers_hash="hash-10"))

        statuses = await group.get_cell_statuses()

        assert (statuses[f"{_POOL_ID}-0"].workers_hash, statuses[f"{_POOL_ID}-1"].workers_hash) == (
            "hash-9",
            "hash-10",
        )

    async def test_a_relaunched_cell_reports_the_new_generation(self):
        """The stamp is what tells the api server the previous process' verdict no longer applies."""
        group = _make_controller(num_cells=1, indep_dp=True)
        await group._reconcile(f"{_POOL_ID}-0", _make_cell_info(0, workers_hash="hash-9"))

        await group._reconcile(f"{_POOL_ID}-0", _make_cell_info(0, workers_hash="hash-10"))

        assert (await group.get_cell_statuses())[f"{_POOL_ID}-0"].workers_hash == "hash-10"


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
