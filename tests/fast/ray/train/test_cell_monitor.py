from unittest.mock import AsyncMock, MagicMock

import pytest

from miles.ray.train.cell_monitor import compute_cell_status, create_trainer_cell_health_checker
from miles.ray.train.cell_state import StateAllocatedAlive, StateAllocatedErrored, StateAllocatedUninitialized
from miles.utils.ft_utils.api_server.models import TriState
from miles.utils.ft_utils.health_checker import ActiveAndEpoch, SimpleHealthCheckerConfig
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerUnreachableError


def _make_worker_handle_mock() -> MagicMock:
    return MagicMock(spec=BaseWorkerHandle)


def _make_indep_dp_info() -> IndepDPInfo:
    return IndepDPInfo(
        cell_index=0,
        num_cells=1,
        alive_rank=0,
        alive_size=1,
        quorum_id=1,
        alive_cell_indices=[0],
    )


def _make_alive_state() -> StateAllocatedAlive:
    return StateAllocatedAlive(worker_handles=[_make_worker_handle_mock()], indep_dp_info=_make_indep_dp_info())


def _find_condition(status, type_: str):
    matches = [c for c in status.conditions if c.type == type_]
    assert len(matches) == 1, f"expected exactly one {type_!r} condition, got {len(matches)}"
    return matches[0]


class TestComputeCellStatusAlive:
    def test_health_true_reports_healthy_true(self):
        result = compute_cell_status(_make_alive_state(), TriState.TRUE, workers_hash="pseudo-hash-0")

        assert result.phase == "Running"
        healthy = _find_condition(result, "Healthy")
        assert healthy.status == TriState.TRUE
        assert healthy.reason is None

    def test_health_false_reports_healthy_false_with_failed_reason(self):
        result = compute_cell_status(_make_alive_state(), TriState.FALSE, workers_hash="pseudo-hash-0")

        healthy = _find_condition(result, "Healthy")
        assert healthy.status == TriState.FALSE
        assert healthy.reason == "HealthCheckFailed"

    def test_health_unknown_reports_healthy_unknown_not_translated_to_true(self):
        """Regression: paused health checker reports UNKNOWN; previously this was
        silently translated to Healthy=TRUE, hiding the transient state from observers."""
        result = compute_cell_status(_make_alive_state(), TriState.UNKNOWN, workers_hash="pseudo-hash-0")

        healthy = _find_condition(result, "Healthy")
        assert healthy.status == TriState.UNKNOWN
        assert healthy.reason == "HealthCheckUnknown"


class TestComputeCellStatusOtherStates:
    @pytest.mark.parametrize("health_status", [TriState.TRUE, TriState.FALSE, TriState.UNKNOWN])
    def test_uninitialized_ignores_health_checker(self, health_status: TriState):
        state = StateAllocatedUninitialized(worker_handles=[_make_worker_handle_mock()])

        result = compute_cell_status(state, health_status, workers_hash="pseudo-hash-0")

        assert result.phase == "Running"
        healthy = _find_condition(result, "Healthy")
        assert healthy.status == TriState.TRUE

    @pytest.mark.parametrize("health_status", [TriState.TRUE, TriState.FALSE, TriState.UNKNOWN])
    def test_errored_always_reports_unhealthy(self, health_status: TriState):
        state = StateAllocatedErrored(worker_handles=[_make_worker_handle_mock()], indep_dp_info=_make_indep_dp_info())

        result = compute_cell_status(state, health_status, workers_hash="pseudo-hash-0")

        healthy = _find_condition(result, "Healthy")
        assert healthy.status == TriState.FALSE
        assert healthy.reason == "ExecutionErrored"


def _make_cell_mock(*, is_alive: bool, execute: AsyncMock) -> MagicMock:
    cell = MagicMock()
    cell.is_alive = is_alive
    cell.cell_index = 0
    cell.execute = execute
    return cell


class TestTrainerCellHealthCheckLiveness:
    """Cell health is defined as process liveness (RPC reachability), not training
    progress: the check must succeed whenever the heartbeat RPC returns, and fail only
    when the actor is dead/unresponsive."""

    @pytest.mark.asyncio
    async def test_rpc_returns_means_healthy_regardless_of_progress(self):
        """A returned heartbeat (even a stale one) proves liveness, so _check passes."""
        execute = AsyncMock(return_value=[MagicMock(last_active_timestamp=0.0, bump_count=1)])
        cell = _make_cell_mock(is_alive=True, execute=execute)

        checker = create_trainer_cell_health_checker(
            cell=cell,
            config=SimpleHealthCheckerConfig(interval=10.0, timeout=10.0, first_wait=0.0, failure_threshold=3),
            get_activeness=lambda: ActiveAndEpoch(active=True, epoch=0),
        )

        await checker._check_fn()
        execute.assert_awaited_once_with("get_heartbeat_status", kill_on_failure=False)

    @pytest.mark.asyncio
    async def test_rpc_error_propagates_as_unhealthy(self):
        """A dead actor makes the heartbeat RPC raise; _check propagates it so the
        checker reports unhealthy."""
        execute = AsyncMock(side_effect=WorkerUnreachableError("worker gone"))
        cell = _make_cell_mock(is_alive=True, execute=execute)

        checker = create_trainer_cell_health_checker(
            cell=cell,
            config=SimpleHealthCheckerConfig(interval=10.0, timeout=10.0, first_wait=0.0, failure_threshold=3),
            get_activeness=lambda: ActiveAndEpoch(active=True, epoch=0),
        )

        with pytest.raises(WorkerUnreachableError):
            await checker._check_fn()

    @pytest.mark.asyncio
    async def test_not_alive_cell_skips_rpc(self):
        """A not-yet-alive cell is not probed and is not reported unhealthy."""
        execute = AsyncMock()
        cell = _make_cell_mock(is_alive=False, execute=execute)

        checker = create_trainer_cell_health_checker(
            cell=cell,
            config=SimpleHealthCheckerConfig(interval=10.0, timeout=10.0, first_wait=0.0, failure_threshold=3),
            get_activeness=lambda: ActiveAndEpoch(active=True, epoch=0),
        )

        await checker._check_fn()
        execute.assert_not_awaited()


class TestComputeCellStatusGeneration:
    @pytest.mark.parametrize(
        "state_factory",
        [
            _make_alive_state,
            lambda: StateAllocatedUninitialized(worker_handles=[_make_worker_handle_mock()]),
            lambda: StateAllocatedErrored(
                worker_handles=[_make_worker_handle_mock()], indep_dp_info=_make_indep_dp_info()
            ),
        ],
    )
    def test_a_status_is_stamped_with_the_generation_it_describes(self, state_factory):
        """The api server joins this with a separately polled listing, so every branch must name its own process."""
        result = compute_cell_status(state_factory(), TriState.TRUE, workers_hash="hash-7")

        assert result.workers_hash == "hash-7"
