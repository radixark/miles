from types import SimpleNamespace

import pytest
import ray
from tests.fast.ray.train.conftest import get_raw_actor_handles, make_alive_cell, make_cell

from miles.ray.train.group import TrainerController
from miles.utils.ft_utils.health_checker import ActivenessTracker
from miles.utils.retry_utils import NonRetryableError


async def _noop_run_after_step(**kwargs) -> None:
    return None


pytestmark = pytest.mark.asyncio

_DUMMY_DATA_PACK = {"data_ref": "data", "sample_indices": [0]}


def _make_controller(cells: list) -> RayTrainGroup:
    group = object.__new__(RayTrainGroup)
    group._cells_by_id = {cell.cell_id: cell for cell in cells}
    group.args = SimpleNamespace(enable_event_analyzer=False, save_debug_event_data=None)
    group._witness_allocator = None
    group._indep_dp_quorum_id = 0
    group._health_checker_activeness = ActivenessTracker(active=True)
    group._test_action_executor = SimpleNamespace(run_after_step=_noop_run_after_step)
    return group


def _make_failing_controller(fn_name: str) -> RayTrainGroup:
    cell = make_alive_cell(0, alive_cell_indices=[0])
    for handle in get_raw_actor_handles(cell):
        ray.get(handle.set_fail_methods.remote([fn_name]))
    return _make_controller([cell])


class TestSingleCellFailsFast:
    async def test_train_does_not_retry_when_no_cell_is_left(self):
        """A lone dead cell can never be healed, so retrying only delays the crash."""
        group = _make_failing_controller("train")

        with pytest.raises(NonRetryableError):
            await group.train(3, _DUMMY_DATA_PACK)

    async def test_train_keeps_the_original_failure_as_the_cause(self):
        """Without the cause the driver traceback says nothing about why training died."""
        group = _make_failing_controller("train")

        with pytest.raises(NonRetryableError) as excinfo:
            await group.train(3, _DUMMY_DATA_PACK)

        assert "Injected failure in train" in str(excinfo.value.__cause__)

    async def test_save_model_does_not_retry_when_no_cell_is_left(self):
        """The save path shares the retry wrapper and must fail fast too."""
        group = _make_failing_controller("save_model")

        with pytest.raises(NonRetryableError):
            await group.save_model(3)


class TestLifecycleCallsAreNotSilent:
    @pytest.mark.parametrize(
        ("method_name", "actor_fn_name"),
        [("onload", "wake_up"), ("offload", "sleep"), ("clear_memory", "clear_memory")],
    )
    async def test_a_lost_last_cell_is_reported(self, method_name, actor_fn_name):
        """Swallowing this hides the real error until an unrelated call fails much later."""
        group = _make_failing_controller(actor_fn_name)

        with pytest.raises(NonRetryableError) as excinfo:
            await getattr(group, method_name)()

        assert f"Injected failure in {actor_fn_name}" in str(excinfo.value.__cause__)


class TestUninitializedCellsKeepTheControllerRetryable:
    async def test_a_failed_attempt_is_retryable_while_a_cell_is_still_healing(self):
        """A healing cell can still join the next attempt, so the failure must stay retryable, not fatal."""
        alive_cell = make_alive_cell(0, alive_cell_indices=[0])
        uninitialized_cell = make_cell(1)
        group = _make_controller([alive_cell, uninitialized_cell])
        alive_cell._mark_as_errored()

        with pytest.raises(RuntimeError) as excinfo:
            group._check_train_one_attempt(
                snapshot_alive_cells=[alive_cell],
                results=[ValueError("Injected failure in train")],
            )

        assert not isinstance(excinfo.value, NonRetryableError)
        assert uninitialized_cell.is_uninitialized

    async def test_a_failed_attempt_is_fatal_once_no_cell_can_come_back(self):
        """With every cell errored there is nothing left to heal, so the group must fail fast."""
        alive_cell = make_alive_cell(0, alive_cell_indices=[0])
        group = _make_controller([alive_cell])
        alive_cell._mark_as_errored()

        with pytest.raises(NonRetryableError):
            group._check_train_one_attempt(
                snapshot_alive_cells=[alive_cell],
                results=[ValueError("Injected failure in train")],
            )

    async def test_offload_tolerates_losing_the_last_alive_cell_while_a_cell_is_still_healing(self):
        """A healing cell keeps the group recoverable, so the lifecycle call must not raise at all."""
        alive_cell = make_alive_cell(0, alive_cell_indices=[0])
        for handle in get_raw_actor_handles(alive_cell):
            ray.get(handle.set_fail_methods.remote(["sleep"]))
        uninitialized_cell = make_cell(1)
        group = _make_controller([alive_cell, uninitialized_cell])

        await group.offload()

        assert not alive_cell.is_alive
        assert uninitialized_cell.is_uninitialized


class TestMultipleCellsStillTolerateFailures:
    async def test_one_dead_cell_does_not_stop_the_lifecycle_call(self):
        """Fault tolerance depends on surviving cells carrying on without the dead one."""
        cells = [make_alive_cell(index, alive_cell_indices=[0, 1]) for index in range(2)]
        ray.get(get_raw_actor_handles(cells[0])[0].set_fail_methods.remote(["sleep"]))
        group = _make_controller(cells)

        await group.offload()

        assert not cells[0].is_alive
        assert cells[1].is_alive
