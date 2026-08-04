from types import SimpleNamespace

import pytest
import ray
from tests.fast.ray.train.conftest import make_alive_cell

from miles.ray.train.group import TrainerController
from miles.utils.retry_utils import NonRetryableError

pytestmark = pytest.mark.asyncio

_DUMMY_DATA_PACK = {"data_ref": "data", "sample_indices": [0]}


def _make_controller(cells: list) -> RayTrainGroup:
    group = object.__new__(RayTrainGroup)
    group._cells = cells
    group.args = SimpleNamespace(enable_event_analyzer=False, save_debug_event_data=None)
    group._witness_allocator = None
    group._indep_dp_quorum_id = 0
    group._health_checker_activeness = True
    group._test_action_executor = SimpleNamespace(run_after_step=lambda **kwargs: None)
    return group


def _make_failing_controller(fn_name: str) -> RayTrainGroup:
    cell = make_alive_cell(0, alive_cell_indices=[0])
    for handle in cell._get_actor_handles():
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


class TestPendingCellsDoNotCountAsRecoverable:
    async def test_a_lost_last_alive_cell_is_reported_even_with_a_pending_cell(self):
        """A pending cell has no live source to heal from, so losing the last alive cell must still be reported."""
        cells = [make_alive_cell(index, alive_cell_indices=[0, 1]) for index in range(2)]
        cells[1].stop()
        cells[1].mark_as_pending()
        ray.get(cells[0]._get_actor_handles()[0].set_fail_methods.remote(["sleep"]))
        group = _make_controller(cells)

        with pytest.raises(NonRetryableError) as excinfo:
            await group.offload()

        assert "Injected failure in sleep" in str(excinfo.value.__cause__)
        assert cells[1].is_pending


class TestMultipleCellsStillTolerateFailures:
    async def test_one_dead_cell_does_not_stop_the_lifecycle_call(self):
        """Fault tolerance depends on surviving cells carrying on without the dead one."""
        cells = [make_alive_cell(index, alive_cell_indices=[0, 1]) for index in range(2)]
        ray.get(cells[0]._get_actor_handles()[0].set_fail_methods.remote(["sleep"]))
        group = _make_controller(cells)

        await group.offload()

        assert cells[0].is_stopped
        assert cells[1].is_alive
