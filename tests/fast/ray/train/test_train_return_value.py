from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
import ray
from tests.fast.ray.train.conftest import make_alive_cell

from miles.backends.megatron_utils.ft.types import TrainStepOutcome, TrainStepOutput
from miles.ray.train.group import TrainerController
from miles.utils.ft_utils.health_checker import ActivenessTracker
from miles.utils.ray_utils import Box

pytestmark = pytest.mark.asyncio

_DUMMY_DATA_PACK = {"data_ref": "data", "sample_indices": [0]}


def _make_controller(cells: list) -> RayTrainGroup:
    group = object.__new__(RayTrainGroup)
    group._cells_by_index = dict(enumerate(cells))
    group.args = SimpleNamespace(enable_event_analyzer=False, save_debug_event_data=None)
    group._witness_allocator = None
    group._indep_dp_quorum_id = 0
    group._health_checker_activeness = ActivenessTracker(active=True)
    group._test_action_executor = SimpleNamespace(run_after_step=AsyncMock())
    return group


def _set_train_return_value(cell: Any, value: Any) -> None:
    for handle in cell._get_actor_handles():
        ray.get(handle.set_train_return_value.remote(value))


def _count_train_calls(cell: Any) -> int:
    return sum(
        sum(1 for method, _args, _kwargs in ray.get(handle.get_calls.remote()) if method == "train")
        for handle in cell._get_actor_handles()
    )


class TestTrainReturnValue:
    async def test_one_result_per_worker_reaches_the_caller(self):
        """The critic values leave the group per worker so the driver can feed them to the actor."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        _set_train_return_value(
            cell, TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=Box({"critic_loss": 1.5}))
        )
        group = _make_controller([cell])

        results = await group.train(3, _DUMMY_DATA_PACK)

        assert [result.outcome for result in results] == [TrainStepOutcome.NORMAL] * 2
        assert [result.values.inner for result in results] == [{"critic_loss": 1.5}] * 2

    async def test_results_of_several_cells_are_concatenated_in_cell_order(self):
        """Independent DP ranks are positional, so a reordered result list misroutes values."""
        cells = [make_alive_cell(index, alive_cell_indices=[0, 1]) for index in range(2)]
        for index, cell in enumerate(cells):
            _set_train_return_value(cell, TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=Box(index)))
        group = _make_controller(cells)

        results = await group.train(3, _DUMMY_DATA_PACK)

        assert [result.values.inner for result in results] == [0, 0, 1, 1]

    async def test_a_failed_cell_contributes_no_result(self):
        """A raw exception object in the returned list would be fed straight into the next train call."""
        cells = [make_alive_cell(index, alive_cell_indices=[0, 1]) for index in range(2)]
        ray.get(get_raw_actor_handles(cells[0])[0].set_fail_methods.remote(["train"]))
        _set_train_return_value(cells[1], TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=Box("ok")))
        group = _make_controller(cells)

        results = await group.train(3, _DUMMY_DATA_PACK)

        assert [result.values.inner for result in results] == ["ok", "ok"]


class TestRetryReturnsTheValue:
    async def test_the_value_of_the_successful_attempt_is_returned(self):
        """train() reads its result through retry, so retry must stop swallowing it."""
        from miles.utils.retry_utils import retry

        attempts = []

        async def _fn(attempt: int) -> str:
            attempts.append(attempt)
            if attempt == 0:
                raise RuntimeError("boom")
            return "second"

        async def _no_sleep(_seconds: float) -> None:
            return None

        assert await retry(_fn, sleep_fn=_no_sleep) == "second"
        assert attempts == [0, 1]


class TestWorkerResultShape:
    async def test_a_critic_payload_does_not_break_the_discarded_check(self):
        """A dict payload carried inside TrainStepOutput.values must not be mistaken for a retry request."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        _set_train_return_value(
            cell, TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=Box({"train_step_outcome": "whatever"}))
        )
        group = _make_controller([cell])

        results = await group.train(3, _DUMMY_DATA_PACK)

        assert len(results) == 2
        assert _count_train_calls(cell) == 2

    async def test_a_discarded_outcome_is_seen_by_the_outcome_check(self):
        """The critic's DISCARDED_SHOULD_RETRY must land in the discarded bucket rather than the normal one."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        results = [TrainStepOutput(outcome=TrainStepOutcome.DISCARDED_SHOULD_RETRY)]

        outcomes = RayTrainGroup._compute_attempt_outcomes([cell], [results])

        assert outcomes["discarded"] == [0]
        assert outcomes["normal"] == []

    async def test_a_discarded_outcome_makes_the_group_retry_the_step(self):
        """A worker asking for a retry must cost a whole extra train attempt, not be silently accepted."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        for handle in get_raw_actor_handles(cell):
            ray.get(
                handle.set_train_return_values_per_attempt.remote(
                    [
                        TrainStepOutput(outcome=TrainStepOutcome.DISCARDED_SHOULD_RETRY),
                        TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=Box("after_retry")),
                    ]
                )
            )
        group = _make_controller([cell])

        results = await group.train(3, _DUMMY_DATA_PACK)

        assert [result.values.inner for result in results] == ["after_retry"] * 2
        assert _count_train_calls(cell) == 4

