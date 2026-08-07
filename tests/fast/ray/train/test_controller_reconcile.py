from collections.abc import Awaitable, Callable
from types import SimpleNamespace

import pytest
from tests.fast.ray.train.conftest import make_provider

from miles.ray.specs.train import compute_trainer_pool_id
from miles.ray.train.controller import TrainerController
from miles.utils import retry_utils
from miles.utils.ft_utils.health_checker import ActivenessTracker
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
    group._pool = _POOL_ID
    group._health_checker_config = None
    group._health_checker_activeness = ActivenessTracker(active=True)
    group._provider = make_provider()
    group._cells_by_id = {}
    return group


def _make_cell_info_for(cell_id: str, *, workers_hash: str = "pseudo-hash-1") -> CellInfo:
    return CellInfo(
        cell_id=cell_id,
        pool_id=_POOL_ID,
        alive=True,
        worker_names=[f"{cell_id}-0"],
        workers_hash=workers_hash,
        meta={"role": "actor"},
    )


def _make_cell_info(cell_index: int, *, workers_hash: str = "pseudo-hash-1") -> CellInfo:
    return _make_cell_info_for(f"{_POOL_ID}-{cell_index}", workers_hash=workers_hash)


class TestOpaqueCellIds:
    async def test_a_cell_named_without_a_trailing_index_is_driven_like_any_other(self):
        """A platform names its cells however it likes, so nothing may parse an ordinal out of a cell id."""
        group = _make_controller()
        cell_id = f"{_POOL_ID}-west-a"

        await group._reconcile(cell_id, _make_cell_info_for(cell_id))

        assert [cell.cell_id for cell in group._cells] == [cell_id]

    async def test_cells_are_ordered_by_id_rather_than_by_a_parsed_index(self):
        """indep-DP ranks come from this order, so it has to hold for ids that carry no number at all."""
        group = _make_controller()

        for cell_id in [f"{_POOL_ID}-c", f"{_POOL_ID}-a", f"{_POOL_ID}-b"]:
            await group._reconcile(cell_id, _make_cell_info_for(cell_id))

        assert [cell.cell_id for cell in group._cells] == [f"{_POOL_ID}-{suffix}" for suffix in "abc"]

    async def test_the_indep_dp_rank_is_the_position_of_the_cell_id(self):
        """A rank must be dense for torch.distributed, and every participant must agree on the same order."""
        group = _make_controller()
        alive_cell_ids = [f"{_POOL_ID}-{suffix}" for suffix in "abc"]

        ranks = [
            group._compute_indep_dp_info(cell_id=cell_id, alive_cell_ids=alive_cell_ids).alive_rank
            for cell_id in alive_cell_ids
        ]

        assert ranks == [0, 1, 2]


class TestReconcile:
    async def test_an_observed_cell_is_added(self):
        """The group learns about its cells from the manager instead of creating them."""
        group = _make_controller()

        await group._reconcile(f"{_POOL_ID}-0", _make_cell_info(0))

        assert [cell.cell_id for cell in group._cells] == [f"{_POOL_ID}-0"]

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

        assert [cell.cell_id for cell in group._cells] == [f"{_POOL_ID}-{i}" for i in range(3)]


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
        assert [cell.cell_id for cell in group._cells] == [f"{_POOL_ID}-{i}" for i in range(4)]

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
