import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
import ray
from tests.fast.ray.train import conftest as train_conftest
from tests.fast.ray.train.conftest import get_raw_actor_handles, make_deployment_identity, make_provider
from tests.fast.train_parallel_config_utils import make_train_parallel_config

import miles.ray.train.group as group_module
import miles.utils.test_utils.fault_injector.controller as fault_hook_module
from miles.backends.megatron_utils.ft.types import TrainStepOutcome, TrainStepOutput
from miles.ray.rollout.inference_controller import UpdatableEngines
from miles.ray.train.group import TrainerController, compute_trainer_health_checker_config
from miles.ray.train_actor import WeightUpdateOutput
from miles.utils import object_store
from miles.utils.audit_utils.event_logger.logger import EventLogger, read_events, set_event_logger
from miles.utils.audit_utils.event_logger.models import (
    CellReconfigureEvent,
    TrainGroupStepEndEvent,
    WeightUpdateResultEvent,
)
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.audit_utils.witness.allocator import WitnessIdAllocator
from miles.utils.data import RolloutDataPack
from miles.utils.dp_schedule import TrainParallelConfig
from miles.utils.object_store import _MooncakeStoreObjectRef
from miles.utils.ray_utils import Box
from miles.utils.retry_utils import NonRetryableError
from miles.utils.test_utils.fault_injector.actions.cell import StopCellAction
from miles.utils.test_utils.fault_injector.controller import _FaultHookController
from miles.utils.test_utils.fault_injector.models import FaultHookName, FaultHookRequest
from miles.utils.test_utils.fault_injector.static_source import render_fault_hooks
from miles.utils.workers.naming import compute_cell_id

pytestmark = pytest.mark.asyncio

_DUMMY_DATA_PACK = RolloutDataPack(sample_indices=[0], data_ref=_MooncakeStoreObjectRef(payload="data"))


def _make_mock_args(
    *,
    indep_dp: bool = True,
    enable_witness: bool = False,
    gpus_per_cell: int = 1,
    num_cells: int = 3,
    ci_fault_hooks: str | None = None,
    ci_fault_hooks_path: str | None = None,
    colocate: bool = True,
    update_weight_transfer_mode: str = "broadcast",
) -> SimpleNamespace:
    # Use SimpleNamespace (not MagicMock) so the args object is picklable. TrainerCell.init
    # passes self.args through Ray to the remote actor; pickling a MagicMock blows the
    # recursion limit because its __getattr__ creates new sub-mocks indefinitely.
    return SimpleNamespace(
        deploy_component="all",
        trainer_controller_addrs=None,
        api_server_port=0,
        indep_dp=indep_dp,
        enable_witness=enable_witness,
        witness_buffer_size=100,
        trainer_heartbeat_checker_interval=10.0,
        trainer_heartbeat_checker_timeout=10.0,
        trainer_heartbeat_checker_first_wait=300.0,
        trainer_heartbeat_checker_failure_threshold=3,
        ci_fault_hooks=ci_fault_hooks,
        ci_fault_hooks_path=ci_fault_hooks_path,
        debug_train_only=False,
        debug_rollout_only=False,
        # compute_megatron_world_size_except_dp(args) = TP * PP * CP. Set CP to
        # gpus_per_cell so TrainerController computes num_cells correctly.
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=gpus_per_cell,
        actor_num_nodes=1,
        actor_num_gpus_per_node=num_cells * gpus_per_cell,
        object_store_backend="ray",
        worker_comm_backend="ray",
        trainer_model_id=None,
        colocate=colocate,
        update_weight_transfer_mode=update_weight_transfer_mode,
        update_weights_timeout=None,
    )


def _make_controller(
    *,
    num_cells: int = 3,
    actor_count_per_cell: int = 1,
    with_ref: bool = False,
    with_opd_teacher: bool = False,
    ci_fault_hooks: str | None = None,
) -> TrainerController:
    """Create a TrainerController and let it observe every cell, as the watcher would."""
    train_conftest.fake_worker_manager.num_cells = num_cells
    train_conftest.fake_worker_manager.actor_count_per_cell = actor_count_per_cell
    group = TrainerController(
        deployment_identity=make_deployment_identity(),
        trainer_id="actor",
        role="actor",
        with_ref=with_ref,
        with_opd_teacher=with_opd_teacher,
        cell_provider=make_provider(),
        cell_operations=AsyncMock(),
    )
    group.args = _make_mock_args(
        indep_dp=True,
        gpus_per_cell=actor_count_per_cell,
        num_cells=num_cells,
        ci_fault_hooks=ci_fault_hooks,
        ci_fault_hooks_path=None,
    )
    group._health_checker_config = compute_trainer_health_checker_config(
        group.args, expected_num_cells=group._expected_num_cells
    )
    if group._expected_num_cells > 1:
        group._indep_dp_store, group._indep_dp_store_addr = group_module.create_tcp_store()
    for cell_index in range(num_cells):
        cell = group._create_cell(
            compute_cell_id(pool_id=group._pool_id, cell_index=cell_index),
            cell_index=cell_index,
            workers_hash="pseudo-hash-1",
        )
        group._cells_by_id[cell.cell_id] = cell
    return group


async def _stop_cell(group: TrainerController, cell_index: int) -> None:
    """Suspension stops the cell in the manager; reconcile then drops it from the bookkeeping."""
    cell_id = compute_cell_id(pool_id=group._pool_id, cell_index=cell_index)
    train_conftest.fake_worker_manager._stop_cells([cell_id])
    await group._reconcile(cell_id, None)


def _cell(group: TrainerController, cell_index: int) -> object:
    return group._cells_by_id[compute_cell_id(pool_id=group._pool_id, cell_index=cell_index)]


def _start_cell(group: TrainerController, cell_index: int) -> None:
    """The manager relaunches the cell, so reconcile hands the controller a fresh object."""
    cell_id = compute_cell_id(pool_id=group._pool_id, cell_index=cell_index)
    group._cells_by_id[cell_id] = group._create_cell(cell_id, cell_index=cell_index, workers_hash="pseudo-hash-2")


def _was_stopped(group: TrainerController, cell_index: int) -> bool:
    return [
        compute_cell_id(pool_id=group._pool_id, cell_index=cell_index)
    ] in train_conftest.fake_worker_manager.stopped_cell_ids


def _was_killed(group: TrainerController, cell_index: int) -> bool:
    for handle in get_raw_actor_handles(_cell(group, cell_index)):
        try:
            ray.get(handle.get_calls.remote())
            return False
        except ray.exceptions.RayActorError:
            pass
    return True


async def _init_controller(group: TrainerController) -> None:
    """Call init and wait for all cells to become alive."""
    await group.init(group.args)


async def _make_alive_controller(*, num_cells: int = 3, **kwargs) -> TrainerController:
    """Create a group and init all cells to alive."""
    group = _make_controller(num_cells=num_cells, **kwargs)
    await _init_controller(group)
    return group


def _output(weight_version: int | None, *failed_cell_ids: str) -> WeightUpdateOutput:
    return WeightUpdateOutput(weight_version=weight_version, failed_cell_ids=failed_cell_ids)


def _outcome_of(output: WeightUpdateOutput) -> tuple[int | None, tuple[str, ...]]:
    return output.weight_version, output.failed_cell_ids


def _make_broadcast_args() -> SimpleNamespace:
    return SimpleNamespace(
        debug_train_only=False,
        debug_rollout_only=False,
        trainer_model_id=None,
        colocate=True,
        update_weight_transfer_mode="broadcast",
        update_weights_timeout=None,
    )


class TestIndepDPStore:
    def test_a_multi_cell_pool_gets_one_quorum_store_from_its_controller(self):
        """The store must be minted once, where every cell can be told the same address."""
        group = _make_controller(num_cells=3)

        assert group._indep_dp_store_addr == train_conftest.FAKE_STORE_ADDR

    def test_a_single_cell_pool_needs_no_quorum_store(self):
        """One cell never renegotiates a quorum, so binding a port for it would be pure waste."""
        group = _make_controller(num_cells=1)

        assert group._indep_dp_store_addr is None


class TestInit:
    def test_the_controller_watches_the_pool_of_its_trainer_id(self):
        """A policy's controller owns the pool named after its trainer id, which the role no longer determines."""
        group = TrainerController(
            deployment_identity=make_deployment_identity(),
            trainer_id="alpha-actor",
            role="actor",
            with_ref=False,
            with_opd_teacher=False,
            cell_provider=make_provider(),
            cell_operations=AsyncMock(),
        )

        assert group._pool_id == "trainer-engine-alpha-actor"

    def test_creates_correct_number_of_cells(self):
        group = _make_controller(num_cells=3)

        assert len(group._cells) == 3
        assert [c.cell_index for c in group._cells] == [0, 1, 2]

    def test_cells_are_allocated_after_init(self):
        group = _make_controller(num_cells=2)

        for cell in group._cells:
            assert cell.is_allocated
            assert not cell.is_alive

    def test_each_cell_has_own_actors(self):
        group = _make_controller(num_cells=3, actor_count_per_cell=2)

        handles_per_cell = [get_raw_actor_handles(cell) for cell in group._cells]
        assert all(len(h) == 2 for h in handles_per_cell)

        all_handles = [h for handles in handles_per_cell for h in handles]
        assert len(set(id(h) for h in all_handles)) == 6

    def test_single_cell_controller(self):
        group = _make_controller(num_cells=1)

        assert len(group._cells) == 1

    async def test_init_gives_the_controller_process_an_object_store(self):
        """The controller frees a failed attempt's outputs itself, which needs a store in its own process."""
        group = _make_controller(num_cells=1)

        await _init_controller(group)

        assert object_store.get_instance() is not None

    async def test_init_marks_all_cells_alive(self):
        group = _make_controller(num_cells=3)

        await _init_controller(group)

        for cell in group._cells:
            assert cell.is_alive
            assert cell.indep_dp_info.alive_cell_indices == [0, 1, 2]
            assert cell.indep_dp_info.alive_size == 3

        assert _cell(group, 0).indep_dp_info.alive_rank == 0
        assert _cell(group, 1).indep_dp_info.alive_rank == 1
        assert _cell(group, 2).indep_dp_info.alive_rank == 2


class TestInitRunsExactlyOnce:
    async def test_a_controller_that_never_ran_init_reports_itself_uninitialized(self):
        """A restarted script asks the controller it found running whether to initialize it or to resume it."""
        group = _make_controller(num_cells=1)

        assert await group.is_initialized() is False

    async def test_a_controller_that_ran_init_reports_itself_initialized(self):
        """The take-over path resumes exactly the controllers that answer this way."""
        group = await _make_alive_controller(num_cells=1)

        assert await group.is_initialized() is True

    async def test_a_second_init_is_refused(self):
        """Initializing trainers a previous script already built would throw away the state they hold."""
        group = await _make_alive_controller(num_cells=1)

        with pytest.raises(AssertionError, match="stale worker"):
            await _init_controller(group)


class TestStopStartCell:
    async def test_stopping_a_cell_reaches_the_worker_manager(self):
        group = await _make_alive_controller(num_cells=2)

        await _stop_cell(group, 1)

        assert _was_stopped(group, 1)
        assert _cell(group, 0).is_alive

    async def test_a_relaunched_cell_is_uninitialized_again(self):
        group = await _make_alive_controller(num_cells=2)
        await _stop_cell(group, 1)

        _start_cell(group, 1)

        assert _cell(group, 1).is_uninitialized


class TestExecuteFirstAlive:
    async def test_picks_first_alive_cell(self):
        group = await _make_alive_controller(num_cells=3)

        await group._execute_first_alive("save_model", rollout_id=42)

        for handle in get_raw_actor_handles(_cell(group, 0)):
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "save_model" for c in calls)

        for cell in group._cells[1:]:
            for handle in get_raw_actor_handles(cell):
                calls = ray.get(handle.get_calls.remote())
                assert not any(c[0] == "save_model" for c in calls)

    async def test_skips_errored_picks_next(self):
        group = await _make_alive_controller(num_cells=2)
        _cell(group, 0)._mark_as_errored()

        await group._execute_first_alive("update_weights")

        for handle in get_raw_actor_handles(_cell(group, 1)):
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "update_weights" for c in calls)


class TestGetTrainParallelConfig:
    @staticmethod
    def _set_configs(cell, configs: list[TrainParallelConfig | None]) -> None:
        handles = get_raw_actor_handles(cell)
        ray.get(
            [handle.set_train_parallel_config.remote(config) for handle, config in zip(handles, configs, strict=True)]
        )

    async def test_returns_config_of_rank_zero_of_the_first_alive_cell(self):
        """The driver reads the config the cell's own rank 0 computed at init."""
        group = await _make_alive_controller(num_cells=2, actor_count_per_cell=2)
        self._set_configs(
            cell=group._cells[0],
            configs=[make_train_parallel_config(dp_size=4), make_train_parallel_config(dp_size=99)],
        )

        assert await group.get_train_parallel_config() == make_train_parallel_config(dp_size=4)

    async def test_skips_stopped_cells(self):
        """A stopped cell 0 must not be asked; the next alive cell answers instead."""
        group = await _make_alive_controller(num_cells=2)
        self._set_configs(cell=_cell(group, 1), configs=[make_train_parallel_config(dp_size=2)])
        await _stop_cell(group, 0)

        assert await group.get_train_parallel_config() == make_train_parallel_config(dp_size=2)


class TestComputeIndepDPInfo:
    def test_all_alive(self):
        group = _make_controller(num_cells=3)

        info = group._compute_indep_dp_info(cell_index=2, alive_cell_indices=[0, 1, 2])

        assert info.alive_rank == 2
        assert info.alive_size == 3
        assert info.cell_index == 2

    def test_with_gap(self):
        group = _make_controller(num_cells=3)

        info = group._compute_indep_dp_info(cell_index=2, alive_cell_indices=[0, 2])

        assert info.alive_rank == 1
        assert info.alive_size == 2


class TestExecuteAllAliveAndCatch:
    async def test_skips_errored_cells(self):
        group = await _make_alive_controller(num_cells=2)
        _cell(group, 1)._mark_as_errored()

        await group._execute_all_alive_and_catch("train")

        for handle in get_raw_actor_handles(_cell(group, 0)):
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "train" for c in calls)

    async def test_refuses_to_retry_when_no_cell_is_alive(self):
        """Retrying without a single live cell can never succeed, so it must fail fast."""
        group = await _make_alive_controller(num_cells=1)
        _cell(group, 0)._mark_as_errored()

        with pytest.raises(NonRetryableError, match="No alive cells"):
            await group._execute_all_alive_and_catch("train")


class TestRefreshCellsReconfigure:
    async def test_reconfigure_triggers_on_alive_change(self):
        """When a cell is stopped, _refresh_cells reconfigures remaining alive cells."""
        group = await _make_alive_controller(num_cells=3)

        # Step 1: Stop cell 1
        await _stop_cell(group, 1)

        # Step 2: Refresh
        await group._refresh_cells(rollout_id=0)

        # Step 3: Quorum bumped (init was quorum 0, this is first reconfigure)
        assert group._indep_dp_quorum_id == 1

        # Step 4: Remaining alive cells have updated indep_dp_info
        assert _cell(group, 0).is_alive
        assert _cell(group, 0).indep_dp_info.alive_cell_indices == [0, 2]
        assert _cell(group, 0).indep_dp_info.alive_rank == 0
        assert _cell(group, 0).indep_dp_info.alive_size == 2

        assert _cell(group, 2).is_alive
        assert _cell(group, 2).indep_dp_info.alive_rank == 1

        # Step 5: Stopped cell untouched
        assert _was_stopped(group, 1)

        # Step 6: Actors received reconfigure_indep_dp
        for cell in [_cell(group, 0), _cell(group, 2)]:
            for handle in get_raw_actor_handles(cell):
                calls = ray.get(handle.get_calls.remote())
                assert any(c[0] == "reconfigure_indep_dp" for c in calls)

    async def test_no_reconfigure_when_unchanged(self):
        group = await _make_alive_controller(num_cells=2)

        await group._refresh_cells(rollout_id=0)

        assert group._indep_dp_quorum_id == 0


class TestRefreshCellsHealing:
    async def test_pending_cell_gets_healed(self):
        """A pending cell goes through allocate + healing with correct alive_rank."""
        group = await _make_alive_controller(num_cells=3)

        # Step 1: Stop cell 2, then start it (pending)
        await _stop_cell(group, 2)
        _start_cell(group, 2)

        # Step 2: Refresh heals the pending cell
        await group._refresh_cells(rollout_id=0)

        # Step 3: All 3 cells are now alive
        assert all(c.is_alive for c in group._cells)

        # Step 4: All cells have consistent indep_dp_info
        for cell in group._cells:
            assert cell.indep_dp_info.alive_cell_indices == [0, 1, 2]
            assert cell.indep_dp_info.alive_size == 3

        # Step 5: Healed cell's actors received init
        for handle in get_raw_actor_handles(_cell(group, 2)):
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "init" for c in calls)

        # Step 6: Source cell sent ckpt to healed cell's alive_rank
        for handle in get_raw_actor_handles(_cell(group, 0)):
            calls = ray.get(handle.get_calls.remote())
            send_calls = [c for c in calls if c[0] == "send_ckpt"]
            assert len(send_calls) == 1
            assert send_calls[0][2]["dst_rank"] == 2

    async def test_multiple_pending_cells_healed(self):
        """Multiple pending cells healed simultaneously."""
        group = await _make_alive_controller(num_cells=3)
        await _stop_cell(group, 1)
        await _stop_cell(group, 2)
        _start_cell(group, 1)
        _start_cell(group, 2)

        await group._refresh_cells(rollout_id=0)

        assert all(c.is_alive for c in group._cells)
        for cell in group._cells:
            assert cell.indep_dp_info.alive_cell_indices == [0, 1, 2]

        # Source (cell 0) sent ckpt to both healed cells
        for handle in get_raw_actor_handles(_cell(group, 0)):
            calls = ray.get(handle.get_calls.remote())
            send_calls = [c for c in calls if c[0] == "send_ckpt"]
            assert len(send_calls) == 2
            dst_ranks = sorted(c[2]["dst_rank"] for c in send_calls)
            assert dst_ranks == [1, 2]

    async def test_pending_cell_with_stopped_cell(self):
        """Pending + stopped: only alive and pending participate, stopped excluded."""
        group = await _make_alive_controller(num_cells=3)

        # cell 1 stopped (not restarted), cell 2 pending
        await _stop_cell(group, 1)
        await _stop_cell(group, 2)
        _start_cell(group, 2)

        await group._refresh_cells(rollout_id=0)

        assert _cell(group, 0).is_alive
        assert _was_stopped(group, 1)
        assert _cell(group, 2).is_alive

        assert _cell(group, 0).indep_dp_info.alive_cell_indices == [0, 2]
        assert _cell(group, 0).indep_dp_info.alive_size == 2
        assert _cell(group, 2).indep_dp_info.alive_rank == 1


class TestRefreshCellsReconfigureEvent:
    @pytest.fixture
    def _event_log_dir(self, tmp_path: Path):
        set_event_logger(EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main")))
        try:
            yield tmp_path
        finally:
            set_event_logger(None)

    @staticmethod
    def _read_reconfigure_events(log_dir: Path) -> list[CellReconfigureEvent]:
        return [e for e in read_events(log_dir) if isinstance(e, CellReconfigureEvent)]

    async def test_healing_emits_event_with_src_and_healed_cells(self, _event_log_dir: Path):
        """A healing reconfigure emits one CellReconfigureEvent naming rollout, src cell, and healed cells."""
        group = await _make_alive_controller(num_cells=3)
        await _stop_cell(group, 2)
        _start_cell(group, 2)

        await group._refresh_cells(rollout_id=7)

        events = self._read_reconfigure_events(_event_log_dir)
        assert len(events) == 1
        assert events[0].rollout_id == 7
        assert events[0].quorum_id == 1
        assert events[0].src_cell_index == 0
        assert events[0].healed_cell_indices == [2]
        assert events[0].alive_cell_indices_after == [0, 1, 2]

    async def test_shrink_emits_event_without_src(self, _event_log_dir: Path):
        """A pure-shrink reconfigure emits one CellReconfigureEvent with no src and no healed cells."""
        group = await _make_alive_controller(num_cells=3)
        await _stop_cell(group, 1)

        await group._refresh_cells(rollout_id=4)

        events = self._read_reconfigure_events(_event_log_dir)
        assert len(events) == 1
        assert events[0].rollout_id == 4
        assert events[0].src_cell_index is None
        assert events[0].healed_cell_indices == []
        assert events[0].alive_cell_indices_after == [0, 2]

    async def test_noop_refresh_emits_no_event(self, _event_log_dir: Path):
        """A refresh that needs no reconfigure emits no CellReconfigureEvent."""
        group = await _make_alive_controller(num_cells=2)

        await group._refresh_cells(rollout_id=1)

        assert self._read_reconfigure_events(_event_log_dir) == []

    async def test_failed_healing_emits_no_event(self, _event_log_dir: Path):
        """When cooperative prepare fails, no CellReconfigureEvent is emitted (witness stays absent)."""
        group = await _make_alive_controller(num_cells=3)
        await _stop_cell(group, 2)
        train_conftest.fake_worker_manager.fail_init_for_cell(2)
        _start_cell(group, 2)

        await group._refresh_cells(rollout_id=5)

        assert self._read_reconfigure_events(_event_log_dir) == []

    async def test_healing_records_the_new_hash_of_the_healed_cell_and_the_kept_hash_of_the_rest(
        self, _event_log_dir: Path
    ) -> None:
        """The event maps exactly the alive cells to the incarnation each one has after the heal."""
        group = await _make_alive_controller(num_cells=3)
        await _stop_cell(group, 2)
        _start_cell(group, 2)

        await group._refresh_cells(rollout_id=7)

        (event,) = self._read_reconfigure_events(_event_log_dir)
        assert event.cell_incarnations_after == {
            compute_cell_id(pool_id=group._pool_id, cell_index=0): "pseudo-hash-1",
            compute_cell_id(pool_id=group._pool_id, cell_index=1): "pseudo-hash-1",
            compute_cell_id(pool_id=group._pool_id, cell_index=2): "pseudo-hash-2",
        }

    async def test_a_shrink_leaves_the_dead_cell_out_of_the_incarnations(self, _event_log_dir: Path) -> None:
        """A cell that is gone after the reconfigure has no incarnation a checker could match against."""
        group = await _make_alive_controller(num_cells=3)
        await _stop_cell(group, 1)

        await group._refresh_cells(rollout_id=4)

        (event,) = self._read_reconfigure_events(_event_log_dir)
        assert event.cell_incarnations_after == {
            compute_cell_id(pool_id=group._pool_id, cell_index=0): "pseudo-hash-1",
            compute_cell_id(pool_id=group._pool_id, cell_index=2): "pseudo-hash-1",
        }


class TestRefreshCellsNoOp:
    async def test_repeated_refresh_without_change_does_not_reconfigure(self):
        """Calling _refresh_cells multiple times without state changes dispatches no actor calls."""
        group = await _make_alive_controller(num_cells=3)

        # Clear init calls by noting current call count
        init_call_counts = {}
        for cell in group._cells:
            for handle in get_raw_actor_handles(cell):
                calls = ray.get(handle.get_calls.remote())
                init_call_counts[id(handle)] = len(calls)

        # Two refreshes — neither should change anything
        await group._refresh_cells(rollout_id=0)
        await group._refresh_cells(rollout_id=0)
        assert group._indep_dp_quorum_id == 0

        # No new calls dispatched
        for cell in group._cells:
            for handle in get_raw_actor_handles(cell):
                calls = ray.get(handle.get_calls.remote())
                assert len(calls) == init_call_counts[id(handle)]

    async def test_refresh_after_reconfigure_is_noop_on_second_call(self):
        group = await _make_alive_controller(num_cells=3)
        await _stop_cell(group, 1)
        await group._refresh_cells(rollout_id=0)
        assert group._indep_dp_quorum_id == 1

        await group._refresh_cells(rollout_id=0)
        assert group._indep_dp_quorum_id == 1


class TestConsecutiveStopStartCycles:
    async def test_stop_train_stop_train_start_train(self):
        """Consecutive: stop 1 → refresh → stop 2 → refresh → start 1 → refresh."""
        group = await _make_alive_controller(num_cells=3)

        # Step 1: Stop cell 1
        await _stop_cell(group, 1)
        await group._refresh_cells(rollout_id=0)
        assert group._indep_dp_quorum_id == 1
        assert _cell(group, 0).indep_dp_info.alive_cell_indices == [0, 2]

        # Step 2: Stop cell 2 (only cell 0 alive)
        await _stop_cell(group, 2)
        await group._refresh_cells(rollout_id=0)
        assert group._indep_dp_quorum_id == 2
        assert _cell(group, 0).indep_dp_info.alive_cell_indices == [0]
        assert _cell(group, 0).indep_dp_info.alive_size == 1

        # Step 3: Start cell 1 (cells 0 and 1 alive)
        _start_cell(group, 1)
        await group._refresh_cells(rollout_id=0)
        assert group._indep_dp_quorum_id == 3
        assert _cell(group, 0).is_alive
        assert _cell(group, 1).is_alive
        assert _was_stopped(group, 2)
        assert _cell(group, 0).indep_dp_info.alive_cell_indices == [0, 1]
        assert _cell(group, 1).indep_dp_info.alive_cell_indices == [0, 1]


class TestTrain:
    async def test_event_analysis_does_not_block_cell_status_requests(self):
        """A growing event log must not make the trainer controller stop answering status requests."""
        group = _make_controller(num_cells=1)
        group._witness_allocator = None
        group._refresh_cells = AsyncMock()
        group._gather_all_alive_and_catch = AsyncMock(return_value=([], []))
        group._check_train_one_attempt = MagicMock()
        group._log_step_end_event = MagicMock()
        group._test_action_executor = AsyncMock()

        started_at = time.monotonic()
        with patch.object(
            group_module.event_analyzer,
            "run_analysis_from_args",
            side_effect=lambda _args: time.sleep(0.2),
        ):
            train_task = asyncio.create_task(group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK))
            await asyncio.sleep(0)

            statuses = await group.get_cell_statuses()

            assert time.monotonic() - started_at < 0.1
            assert set(statuses) == {"trainer-engine-actor-00000"}
            assert await train_task == []

    async def test_event_analysis_failure_still_fails_training(self):
        """Moving event analysis off the event loop must preserve its failure contract."""
        group = _make_controller(num_cells=1)

        with patch.object(
            group_module.event_analyzer,
            "run_analysis_from_args",
            side_effect=ValueError("event analysis failed"),
        ):
            with pytest.raises(ValueError, match="event analysis failed"):
                await group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK)

    async def test_train_refreshes_and_dispatches(self):
        group = await _make_alive_controller(num_cells=2)

        await group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK)

        for cell in group._cells:
            for handle in get_raw_actor_handles(cell):
                calls = ray.get(handle.get_calls.remote())
                assert any(c[0] == "train" for c in calls)

    async def test_train_with_stopped_cell_only_dispatches_to_alive(self):
        group = await _make_alive_controller(num_cells=3)
        await _stop_cell(group, 1)

        await group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK)

        for cell in [_cell(group, 0), _cell(group, 2)]:
            for handle in get_raw_actor_handles(cell):
                calls = ray.get(handle.get_calls.remote())
                assert any(c[0] == "train" for c in calls)

        assert _was_stopped(group, 1)

    async def test_consecutive_train_no_reconfigure_overhead(self):
        """Multiple train calls with no state changes — no reconfigure overhead."""
        group = await _make_alive_controller(num_cells=3)

        # Note init call count
        init_counts = {}
        for cell in group._cells:
            for handle in get_raw_actor_handles(cell):
                init_counts[id(handle)] = len(ray.get(handle.get_calls.remote()))

        for step in range(3):
            await group.train(rollout_id=step, rollout_data_pack=_DUMMY_DATA_PACK)

        assert group._indep_dp_quorum_id == 0

        for cell in group._cells:
            for handle in get_raw_actor_handles(cell):
                calls = ray.get(handle.get_calls.remote())
                new_calls = calls[init_counts[id(handle)] :]
                assert not any(c[0] == "reconfigure_indep_dp" for c in new_calls)
                train_calls = [c for c in new_calls if c[0] == "train"]
                assert len(train_calls) == 3

    async def test_rapid_stop_start_before_train(self):
        """Cell stopped and immediately started before next train — healed in one shot."""
        group = await _make_alive_controller(num_cells=3)

        await _stop_cell(group, 1)
        _start_cell(group, 1)

        await group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK)

        assert all(c.is_alive for c in group._cells)
        for cell in group._cells:
            assert cell.indep_dp_info.alive_cell_indices == [0, 1, 2]

    async def test_full_lifecycle_through_train(self):
        """End-to-end: normal → degraded → steady degraded → healing → full."""
        group = await _make_alive_controller(num_cells=3)

        # Step 1: Normal training (no reconfigure)
        await group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK)
        assert group._indep_dp_quorum_id == 0

        # Step 2: Stop cell 2 → degraded (triggers reconfigure)
        await _stop_cell(group, 2)
        await group.train(rollout_id=1, rollout_data_pack=_DUMMY_DATA_PACK)
        assert group._indep_dp_quorum_id == 1
        assert _cell(group, 0).indep_dp_info.alive_cell_indices == [0, 1]

        # Step 3: Steady degraded (no reconfigure)
        await group.train(rollout_id=2, rollout_data_pack=_DUMMY_DATA_PACK)
        assert group._indep_dp_quorum_id == 1

        # Step 4: Start cell 2 → healing (triggers reconfigure)
        _start_cell(group, 2)
        await group.train(rollout_id=3, rollout_data_pack=_DUMMY_DATA_PACK)
        assert group._indep_dp_quorum_id == 2
        assert all(c.is_alive for c in group._cells)
        assert _cell(group, 2).indep_dp_info.alive_cell_indices == [0, 1, 2]

        # Step 5: Full training again (no reconfigure)
        await group.train(rollout_id=4, rollout_data_pack=_DUMMY_DATA_PACK)
        assert group._indep_dp_quorum_id == 2


class TestPerCellErrorIsolation:
    async def test_one_cell_failure_marks_errored_others_ok(self):
        """One cell's actor fails during broadcast, that cell is killed and stopped, others complete normally."""
        group = await _make_alive_controller(num_cells=3)

        # Step 1: Make cell 1's actors fail on train
        for handle in get_raw_actor_handles(_cell(group, 1)):
            ray.get(handle.set_fail_methods.remote(["train"]))

        # Step 2: Broadcast train
        await group._execute_all_alive_and_catch("train", rollout_id=0, rollout_data_ref="data")

        # Step 3: Cell 1 is errored, others alive
        assert _cell(group, 0).is_alive
        assert _was_killed(group, 1)
        assert _cell(group, 2).is_alive

        # Step 4: Other cells received train call
        for cell_idx in [0, 2]:
            for handle in get_raw_actor_handles(_cell(group, cell_idx)):
                calls = ray.get(handle.get_calls.remote())
                assert any(c[0] == "train" for c in calls)

    async def test_errored_cell_skipped_in_next_broadcast(self):
        """After marking a cell errored, subsequent broadcasts skip it."""
        group = await _make_alive_controller(num_cells=2)

        # Step 1: Make cell 0 fail
        for handle in get_raw_actor_handles(_cell(group, 0)):
            ray.get(handle.set_fail_methods.remote(["train"]))

        await group._execute_all_alive_and_catch("train", rollout_id=0, rollout_data_ref="data")
        assert _was_killed(group, 0)

        # Step 2: Next broadcast only goes to cell 1
        await group._execute_all_alive_and_catch("train", rollout_id=1, rollout_data_ref="data")

        for handle in get_raw_actor_handles(_cell(group, 1)):
            calls = ray.get(handle.get_calls.remote())
            train_calls = [c for c in calls if c[0] == "train"]
            assert len(train_calls) == 2


class TestExecuteFirstAliveFallback:
    async def test_first_cell_fails_retry_falls_back_to_next(self):
        """If the first alive cell fails, retry in save_model kills+stops it and picks the next."""
        group = await _make_alive_controller(num_cells=3)

        # Step 1: Make cell 0 fail on save_model
        for handle in get_raw_actor_handles(_cell(group, 0)):
            ray.get(handle.set_fail_methods.remote(["save_model"]))

        # Step 2: save_model uses retry(lambda _: self._execute_first_alive(...))
        await group.save_model(rollout_id=42)

        # Step 3: Cell 0 errored, cell 1 handled it
        assert _was_killed(group, 0)
        assert _cell(group, 1).is_alive

        for handle in get_raw_actor_handles(_cell(group, 1)):
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "save_model" for c in calls)

    async def test_single_execute_first_alive_raises_on_failure(self):
        """A single _execute_first_alive call raises (no retry) when the first cell fails."""
        group = await _make_alive_controller(num_cells=2)

        for handle in get_raw_actor_handles(_cell(group, 0)):
            ray.get(handle.set_fail_methods.remote(["save_model"]))

        with pytest.raises(Exception):  # noqa: B017
            await group._execute_first_alive("save_model", rollout_id=42)

        assert _was_killed(group, 0)

    async def test_losing_the_last_cell_keeps_the_worker_error_as_the_cause(self):
        """Without the cause the driver traceback says nothing about why the last cell died."""
        group = await _make_alive_controller(num_cells=1)

        for handle in get_raw_actor_handles(_cell(group, 0)):
            ray.get(handle.set_fail_methods.remote(["save_model"]))

        with pytest.raises(NonRetryableError) as excinfo:
            await group._execute_first_alive("save_model", rollout_id=42)

        assert "Injected failure in save_model" in str(excinfo.value.__cause__)

    async def test_losing_the_last_alive_cell_is_fatal_even_while_a_cell_is_still_healing(self):
        """The next attempt runs on alive cells alone, so a cell that is only healing cannot keep it retryable."""
        group = await _make_alive_controller(num_cells=2)
        await _stop_cell(group, 1)
        _start_cell(group, 1)

        for handle in get_raw_actor_handles(_cell(group, 0)):
            ray.get(handle.set_fail_methods.remote(["save_model"]))

        with pytest.raises(NonRetryableError):
            await group._execute_first_alive("save_model", rollout_id=42)

        assert _cell(group, 1).is_uninitialized

    async def test_terminal_failure_does_not_burn_another_backoff(self):
        """Retrying without a single live cell can never succeed, so it must fail fast."""
        group = await _make_alive_controller(num_cells=1)

        for handle in get_raw_actor_handles(_cell(group, 0)):
            ray.get(handle.set_fail_methods.remote(["save_model"]))

        attempts = 0
        execute_first_alive = group._execute_first_alive

        async def _counting_execute_first_alive(fn_name: str, **kwargs: object) -> object:
            nonlocal attempts
            attempts += 1
            return await execute_first_alive(fn_name, **kwargs)

        group._execute_first_alive = _counting_execute_first_alive

        with pytest.raises(NonRetryableError):
            await group.save_model(rollout_id=42)

        assert attempts == 1


class TestRefreshCellsErrorHandling:
    async def test_healing_failure_marks_pending_cell_errored_keeps_alive(self):
        """When healing init fails, the pending cell is killed and stopped (via _execute_raw's
        except path, which marks errored then confirms-dead), alive cells unaffected."""
        group = await _make_alive_controller(num_cells=3)

        # Step 1: Stop cell 2 and start it (pending)
        await _stop_cell(group, 2)

        # Step 2: Replace actor factory so new actors fail on init
        train_conftest.fake_worker_manager.fail_init_for_cell(2)
        _start_cell(group, 2)

        # Step 3: Refresh — healing init fails, cell auto-marks errored
        await group._refresh_cells(rollout_id=0)

        # Step 4: Cell 2 errored, cells 0 and 1 still alive. _was_stopped would already be true
        # from step 1, so it says nothing about the healing failure; the kill does.
        assert _cell(group, 0).is_alive
        assert _cell(group, 1).is_alive
        assert _cell(group, 2).is_errored
        assert _was_killed(group, 2)


class TestHeartbeatMonitor:
    async def test_heartbeat_normal_does_not_mark_errored(self):
        """When heartbeat returns recent timestamp, cells stay alive."""
        group = await _make_alive_controller(num_cells=2)

        for cell in group._cells:
            await cell.health_checker._check_fn()

        assert all(c.is_alive for c in group._cells)

    async def test_heartbeat_stale_timestamp_does_not_mark_errored(self):
        """A stale heartbeat timestamp alone keeps the cell healthy: cell health is
        liveness, not training progress, so a cell legitimately blocked in a cross-cell
        collective (whose training loop stops bumping the heartbeat) must not be reported
        unhealthy as long as the heartbeat RPC still returns."""
        group = await _make_alive_controller(num_cells=2)

        # Drive cell 1's last-active timestamp to the epoch (maximally stale); the
        # liveness check must ignore staleness while the heartbeat RPC keeps returning.
        for handle in get_raw_actor_handles(_cell(group, 1)):
            ray.get(handle.set_last_active_timestamp.remote(0.0))

        # Neither check raises (a returned heartbeat proves the process is alive) and
        # both cells stay alive despite cell 1's stale timestamp.
        await _cell(group, 1).health_checker._check_fn()
        await _cell(group, 0).health_checker._check_fn()
        assert all(c.is_alive for c in group._cells)

    async def test_heartbeat_timeout_marks_errored(self):
        """When heartbeat call fails (actor unresponsive), cell is marked errored."""
        group = await _make_alive_controller(num_cells=2)

        for handle in get_raw_actor_handles(_cell(group, 0)):
            ray.get(handle.set_heartbeat_fail.remote(True))

        with pytest.raises(RuntimeError, match="Injected heartbeat failure"):
            await _cell(group, 0).health_checker._check_fn()

    async def test_the_group_activeness_flag_reaches_every_cell_checker(self):
        """Checkers pull activeness from the group, so one flag governs the whole pool."""
        group = await _make_alive_controller(num_cells=2)

        group._health_checker_activeness.bump_active(False)
        assert not any(c.health_checker._get_activeness().active for c in group._cells)

        group._health_checker_activeness.bump_active(True)
        assert all(c.health_checker._get_activeness().active for c in group._cells)

    async def test_the_paused_context_restores_activeness_after_an_exception(self):
        """A crash inside a reconfigure must not leave health checking off for the rest of the run."""
        group = await _make_alive_controller(num_cells=2)

        with pytest.raises(RuntimeError, match="boom"):
            with group._paused_health_checkers():
                assert not group._health_checker_activeness.get().active
                raise RuntimeError("boom")

        assert group._health_checker_activeness.get().active


NORMAL = TrainStepOutput(outcome=TrainStepOutcome.NORMAL)
DISCARDED = TrainStepOutput(outcome=TrainStepOutcome.DISCARDED_SHOULD_RETRY)


_ERR = RuntimeError("boom")
_ERR2 = ValueError("boom2")


def _alive_cells_for(results) -> list[SimpleNamespace]:
    """Mock alive cells aligned with a `results` list; only `.cell_index` is read."""
    return [SimpleNamespace(cell_index=i) for i in range(len(results))]


class TestCheckTrainOneAttempt:
    """_check_train_one_attempt raises ValueError when any non-exception cell has DISCARDED."""

    @pytest.mark.parametrize(
        "results",
        [
            [[NORMAL]],  # single cell, single actor
            [[NORMAL, NORMAL], [NORMAL]],  # multi cell, multi actor
            [_ERR, [NORMAL, NORMAL]],  # errored + normal → ok
            [[]],  # cell with empty actor list → vacuously ok
        ],
    )
    def test_no_retry_when_no_discarded(self, results):
        _make_controller(num_cells=1)._check_train_one_attempt(_alive_cells_for(results), results)  # should not raise

    @pytest.mark.parametrize(
        "results",
        [
            [[DISCARDED]],  # single cell
            [[DISCARDED], [DISCARDED, DISCARDED]],  # multi cell
            [[NORMAL, DISCARDED]],  # mixed within same cell
            [[NORMAL], [DISCARDED]],  # mixed across cells
            [_ERR, [DISCARDED]],  # errored + discarded → retry
        ],
    )
    def test_retry_when_discarded_exists(self, results):
        with pytest.raises(ValueError, match="DISCARDED_SHOULD_RETRY"):
            _make_controller(num_cells=1)._check_train_one_attempt(_alive_cells_for(results), results)

    @pytest.mark.parametrize(
        "results",
        [
            [_ERR],  # single cell errored
            [_ERR, _ERR2],  # multiple cells all errored
        ],
    )
    def test_raises_when_all_cells_errored(self, results):
        """No cell is alive to carry the next attempt, so the controller must raise the fatal error."""
        with pytest.raises(NonRetryableError, match="All cells failed"):
            _make_controller(num_cells=1)._check_train_one_attempt(_alive_cells_for(results), results)

    @pytest.mark.parametrize(
        "results",
        [
            [_ERR],  # single cell errored
            [_ERR, _ERR2],  # multiple cells all errored
        ],
    )
    def test_raises_a_fatal_error_when_every_cell_is_already_errored(self, results):
        """With every cell errored there is nothing left to heal, so an all-errored attempt is non-retryable."""
        group = _make_controller(num_cells=1)
        for cell in group._cells:
            cell._mark_as_errored()

        with pytest.raises(NonRetryableError, match="All cells failed"):
            group._check_train_one_attempt(_alive_cells_for(results), results)

    def test_compute_attempt_outcomes_buckets_cells_by_index(self):
        """_compute_attempt_outcomes buckets each alive cell into errored / discarded / normal by index."""
        results = [_ERR, [DISCARDED], [NORMAL, NORMAL]]
        outcomes = TrainerController._compute_attempt_outcomes(_alive_cells_for(results), results)
        assert outcomes == {"errored": [0], "discarded": [1], "normal": [2]}

    def test_a_payload_carrying_output_is_bucketed_by_its_outcome(self):
        """The critic ships values alongside its outcome, so the payload must not hide a retry request."""
        results = [[TrainStepOutput(outcome=TrainStepOutcome.DISCARDED_SHOULD_RETRY, values=Box("ref"))]]
        outcomes = TrainerController._compute_attempt_outcomes(_alive_cells_for(results), results)
        assert outcomes == {"errored": [], "discarded": [0], "normal": []}


async def _set_all_train_return(group: TrainerController, value: TrainStepOutput) -> None:
    for cell in group._cells:
        for handle in get_raw_actor_handles(cell):
            ray.get(handle.set_train_return_value.remote(value))


async def _set_all_train_returns_per_attempt(group: TrainerController, values: list[TrainStepOutput]) -> None:
    for cell in group._cells:
        for handle in get_raw_actor_handles(cell):
            ray.get(handle.set_train_return_values_per_attempt.remote(values))


def _count_train_calls(group: TrainerController, cell_index: int) -> int:
    total = 0
    for handle in get_raw_actor_handles(_cell(group, cell_index)):
        calls = ray.get(handle.get_calls.remote())
        total += sum(1 for c in calls if c[0] == "train")
    return total


class TestTrainRetry:
    async def test_no_retry_on_normal(self):
        """All cells return NORMAL → no retry, train called once per cell."""
        group = await _make_alive_controller(num_cells=2)

        await group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK)

        for i in range(2):
            assert _count_train_calls(group, i) == 1

    async def test_retry_on_all_discarded_then_normal(self):
        """First attempt: all DISCARDED. Second attempt: all NORMAL. Train called twice."""
        group = await _make_alive_controller(num_cells=2)
        await _set_all_train_returns_per_attempt(group, [DISCARDED, NORMAL])

        await group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK)

        for i in range(2):
            assert _count_train_calls(group, i) == 2

    async def test_retry_multiple_times_then_succeed(self):
        """DISCARDED 3 times, then NORMAL on 4th attempt."""
        group = await _make_alive_controller(num_cells=2)
        await _set_all_train_returns_per_attempt(group, [DISCARDED, DISCARDED, DISCARDED, NORMAL])

        await group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK)

        for i in range(2):
            assert _count_train_calls(group, i) == 4

    async def test_a_failed_attempt_releases_values_returned_by_its_successful_workers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A retry must release values already returned by successful workers in the failed attempt."""

        class RecordingStore:
            def __init__(self) -> None:
                self.removed: list[Box] = []

            def remove(self, ref: Box) -> None:
                self.removed.append(ref)

        group = await _make_alive_controller(num_cells=2)
        values_ref = Box("failed-attempt-values")
        successful_output = TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=values_ref)
        first_cell_handle = get_raw_actor_handles(_cell(group, 0))[0]
        second_cell_handle = get_raw_actor_handles(_cell(group, 1))[0]
        ray.get(first_cell_handle.set_train_return_values_per_attempt.remote([successful_output, NORMAL]))
        ray.get(second_cell_handle.set_train_return_values_per_attempt.remote([DISCARDED, NORMAL]))
        store = RecordingStore()
        monkeypatch.setattr(object_store, "_INSTANCE", store)

        await group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK)

        assert [ref.inner for ref in store.removed] == [values_ref.inner]

    async def test_cell_errored_does_not_retry_when_others_normal(self):
        """One cell errors during train but others return NORMAL → no retry.

        See _check_train_one_attempt: 'If some cells errors + all other cells claim
        normal, we do *not* retry. This may happen when some cells fails *after*
        exchanging gradients w/ others.' So alive cells get exactly 1 train call.
        """
        group = await _make_alive_controller(num_cells=3)

        # Step 1: Make cell 1 fail (exception)
        for handle in get_raw_actor_handles(_cell(group, 1)):
            ray.get(handle.set_fail_methods.remote(["train"]))

        # Step 2: Train completes without retry (cell 1 errored but others NORMAL)
        await group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK)

        # Step 3: Cell 1 errored, alive cells each got 1 train call (no retry)
        assert _was_killed(group, 1)
        for i in [0, 2]:
            assert _count_train_calls(group, i) == 1


class TestAllocateWitnessInfo:
    def test_returns_none_when_disabled(self):
        """When _witness_allocator is None, _allocate_witness_info returns None."""
        group = _make_controller(num_cells=1)
        group._witness_allocator = None

        result = group._allocate_witness_info(rollout_id=0, attempt=0, sample_indices=[10, 20, 30])

        assert result is None

    def test_returns_witness_info_when_enabled(self):
        """When witness is enabled, _allocate_witness_info returns a WitnessInfo with correct number of ids."""
        group = _make_controller(num_cells=1)
        group._witness_allocator = WitnessIdAllocator(buffer_size=100)

        with patch("miles.ray.train.group.is_event_logger_initialized", return_value=False):
            result = group._allocate_witness_info(rollout_id=0, attempt=0, sample_indices=[10, 20, 30])

        assert result is not None
        assert len(result.witness_ids) == 3
        assert isinstance(result.stale_ids, list)


class TestLogStepEndEvent:
    def test_with_normal_and_error_cells(self):
        """Passes correct cell_outcomes to event logger for a mix of normal and errored cells."""
        group = _make_controller(num_cells=3)

        mock_cell_0 = MagicMock()
        mock_cell_0.cell_index = 0
        mock_cell_1 = MagicMock()
        mock_cell_1.cell_index = 1
        mock_cell_2 = MagicMock()
        mock_cell_2.cell_index = 2

        snapshot_alive_cells = [mock_cell_0, mock_cell_1, mock_cell_2]
        results = [
            [NORMAL, NORMAL],
            RuntimeError("boom"),
            [NORMAL],
        ]

        with patch("miles.ray.train.group.is_event_logger_initialized", return_value=True), patch(
            "miles.ray.train.group.get_event_logger"
        ) as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            group._log_step_end_event(
                rollout_id=42,
                attempt=3,
                snapshot_alive_cells=snapshot_alive_cells,
                results=results,
            )

            mock_logger.log.assert_called_once()
            args = mock_logger.log.call_args[0]
            partial = args[1]
            assert partial["rollout_id"] == 42
            assert partial["attempt"] == 3
            assert partial["role"] == "actor"

            cell_outcomes = partial["cell_outcomes"]
            assert cell_outcomes[0] == [TrainStepOutcome.NORMAL, TrainStepOutcome.NORMAL]
            assert cell_outcomes[1] == "error"
            assert cell_outcomes[2] == [TrainStepOutcome.NORMAL]


class TestCellStatusesUnderConcurrentReconcile:
    async def test_a_cell_removed_while_the_statuses_are_read_does_not_abort_the_read(self):
        """The api server reads this from its own thread while reconcile adds and drops cells,
        and iterating the live dict raises RuntimeError instead of answering the request."""
        controller = _make_controller(num_cells=3)
        victim = compute_cell_id(pool_id=controller._pool_id, cell_index=1)
        real_cell = _cell(controller, 0)

        class _EvictingCell:
            def cell_status(self_inner):
                controller._cells_by_id.pop(victim, None)
                return real_cell.cell_status()

        controller._cells_by_id[compute_cell_id(pool_id=controller._pool_id, cell_index=0)] = _EvictingCell()

        statuses = await controller.get_cell_statuses()

        # The snapshot is taken before the first cell_status() call, so the evicted cell is still
        # answered for. What matters is that the read completes instead of raising.
        assert set(statuses) == {compute_cell_id(pool_id=controller._pool_id, cell_index=i) for i in range(3)}


class TestUpdateWeightsReturnsTheVersion:
    def _make_group(self, *, per_worker_outputs: list[WeightUpdateOutput]) -> TrainerController:
        group = TrainerController.__new__(TrainerController)
        group.args = _make_broadcast_args()
        group._trainer_id = "trainer-0"
        group._debug_trainer_load_state_timestamp = 0.0
        group._execute_first_alive = AsyncMock(return_value=per_worker_outputs)
        return group

    async def test_the_controller_answers_the_version_the_engines_now_serve(self):
        """The driver can only publish the version to the executor if the controller hands it back."""
        group = self._make_group(per_worker_outputs=[_output(1), _output(1)])

        assert _outcome_of(await group.update_weights(info=MagicMock())) == _outcome_of(_output(1))

    async def test_a_trainer_that_skipped_the_broadcast_answers_nothing(self):
        """--debug-skip-weight-update returns None from every worker, which must reach the driver as None."""
        group = self._make_group(per_worker_outputs=[_output(None)])

        assert (await group.update_weights(info=MagicMock())).weight_version is None

    async def test_it_broadcasts_the_window_the_orchestration_script_opened(self):
        """The engines it writes into are the ones the script snapshotted, not a set it fetched for itself."""
        group = self._make_group(per_worker_outputs=[_output(1)])
        info = MagicMock()

        await group.update_weights(info=info)

        group._execute_first_alive.assert_awaited_once_with(
            "update_weights",
            timeout=group.args.update_weights_timeout,
            info=info,
            debug_weight_update_id=ANY,
            rollout_id=None,
        )


class TestModelOwnedWeightVersions:
    @pytest.mark.parametrize("versions", [[7, 7], [4, 9], [8, 2], [None, 5], [0, 0]])
    async def test_controller_returns_model_versions_without_allocating_ordinals(self, versions: list) -> None:
        """Republishing, skipped steps and checkpoint rewinds preserve model versions."""
        controller = TrainerController.__new__(TrainerController)
        controller.args = _make_broadcast_args()
        controller._trainer_id = "trainer-0"
        controller._debug_trainer_load_state_timestamp = 0.0
        controller._execute_first_alive = AsyncMock(
            side_effect=[[_output(version), _output(version)] for version in versions]
        )
        info = MagicMock()

        outputs = [await controller.update_weights(info=info) for _ in versions]

        assert [output.weight_version for output in outputs] == versions
        assert all(call.kwargs["info"] is info for call in controller._execute_first_alive.await_args_list)

    async def test_retry_reads_the_recovered_models_version(self) -> None:
        """A failed trainer does not reserve a version for its replacement."""
        controller = TrainerController.__new__(TrainerController)
        controller.args = _make_broadcast_args()
        controller._trainer_id = "trainer-0"
        controller._debug_trainer_load_state_timestamp = 0.0
        controller._execute_first_alive = AsyncMock(
            side_effect=[RuntimeError("cell died"), [_output(12), _output(12)]]
        )

        assert (await controller.update_weights(info=MagicMock())).weight_version == 12


class TestInitForwardsModelFlags:
    async def test_every_worker_learns_its_role_and_which_extra_models_to_build(self):
        """These flags decide which models a worker allocates, so dropping one silently changes the objective."""
        group = _make_controller(num_cells=2, actor_count_per_cell=2, with_ref=True, with_opd_teacher=True)

        await _init_controller(group)

        for cell in group._cells:
            for handle in get_raw_actor_handles(cell):
                [init_call] = [c for c in ray.get(handle.get_calls.remote()) if c[0] == "init"]
                assert init_call[2]["role"] == "actor"
                assert init_call[2]["with_ref"] is True
                assert init_call[2]["with_opd_teacher"] is True


class TestTrainRunsFaultHooks:
    async def test_train_applies_the_hook_armed_for_that_rollout_before_returning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The declared stop must finish before the caller can begin the next rollout."""
        _isolate_fault_hook_controller(monkeypatch)
        requests = render_fault_hooks(
            [
                FaultHookRequest(
                    request_id="stop-cell-2",
                    hook_name=FaultHookName.TRAINER_CONTROLLER_STEP_END,
                    rollout_id=4,
                    action=StopCellAction(cell_id="trainer-engine-actor-00002"),
                )
            ]
        )
        group = await _make_alive_controller(num_cells=3, ci_fault_hooks=requests)

        await group.train(rollout_id=4, rollout_data_pack=_DUMMY_DATA_PACK)

        group._cell_operations.suspend.assert_awaited_once_with(cell_id="trainer-engine-actor-00002")

    async def test_train_leaves_the_pool_alone_until_the_declared_rollout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unmatched rollout must preserve the pending stop for its declared step."""
        _isolate_fault_hook_controller(monkeypatch)
        requests = render_fault_hooks(
            [
                FaultHookRequest(
                    request_id="stop-cell-2",
                    hook_name=FaultHookName.TRAINER_CONTROLLER_STEP_END,
                    rollout_id=4,
                    action=StopCellAction(cell_id="trainer-engine-actor-00002"),
                )
            ]
        )
        group = await _make_alive_controller(num_cells=3, ci_fault_hooks=requests)

        await group.train(rollout_id=3, rollout_data_pack=_DUMMY_DATA_PACK)

        group._cell_operations.suspend.assert_not_awaited()

        await group.train(rollout_id=4, rollout_data_pack=_DUMMY_DATA_PACK)

        group._cell_operations.suspend.assert_awaited_once_with(cell_id="trainer-engine-actor-00002")


def _isolate_fault_hook_controller(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = _FaultHookController()
    monkeypatch.setattr(group_module, "fault_hook_controller", controller)
    monkeypatch.setattr(fault_hook_module, "fault_hook_controller", controller)


class TestSaveModel:
    async def test_the_selected_cell_is_told_whether_the_save_must_be_synchronous(self):
        """An async save that the caller asked to block on would let training race the checkpoint writer."""
        group = await _make_alive_controller(num_cells=2)

        await group.save_model(rollout_id=9, force_sync=True)

        for handle in get_raw_actor_handles(_cell(group, 0)):
            save_calls = [c for c in ray.get(handle.get_calls.remote()) if c[0] == "save_model"]
            assert [c[2] for c in save_calls] == [{"rollout_id": 9, "force_sync": True}]


class TestExportHf:
    async def test_a_failed_first_cell_hands_the_same_export_to_the_next_alive_cell(self):
        """The exported checkpoint must land at the requested path even when the first cell dies mid-export."""
        group = await _make_alive_controller(num_cells=2)
        for handle in get_raw_actor_handles(_cell(group, 0)):
            ray.get(handle.set_fail_methods.remote(["export_hf"]))

        await group.export_hf(rollout_id=4, path="/ckpt/hf-4")

        assert _was_killed(group, 0)
        for handle in get_raw_actor_handles(_cell(group, 1)):
            export_calls = [c for c in ray.get(handle.get_calls.remote()) if c[0] == "export_hf"]
            assert [c[2] for c in export_calls] == [{"rollout_id": 4, "path": "/ckpt/hf-4"}]


class TestUpdateWeightsReachesTheWorker:
    async def test_the_engine_snapshot_reaches_the_worker_and_its_version_comes_back(self):
        """A worker that never sees the snapshot broadcasts to engines that were not part of the update window."""
        info = SimpleNamespace(engine_cell_ids=["trainer-actor-0"], snapshot_cell_id_to_hashes={"trainer-actor-0": "workers-hash-9"})
        group = await _make_alive_controller(num_cells=1)
        for handle in get_raw_actor_handles(_cell(group, 0)):
            ray.get(handle.set_update_weights_return_value.remote(_output(1)))

        assert _outcome_of(await group.update_weights(info=info, rollout_id=3)) == _outcome_of(_output(1))

        for handle in get_raw_actor_handles(_cell(group, 0)):
            [update_call] = [c for c in ray.get(handle.get_calls.remote()) if c[0] == "update_weights"]
            assert update_call[2]["info"].snapshot_cell_id_to_hashes == {"trainer-actor-0": "workers-hash-9"}
            assert "weight_version" not in update_call[2]

    async def test_reloading_the_trainer_state_does_not_rewind_the_published_version(self):
        """A hot restart reloads the cells while the controller survives, and restarting at version 1 would republish an old ordinal."""
        info = SimpleNamespace(engine_cell_ids=[], snapshot_cell_id_to_hashes={})
        group = await _make_alive_controller(num_cells=1)
        handles = get_raw_actor_handles(_cell(group, 0))
        for handle in handles:
            ray.get(handle.set_update_weights_return_value.remote(_output(1)))
        assert _outcome_of(await group.update_weights(info=info)) == _outcome_of(_output(1))

        await group.load_state()
        for handle in handles:
            ray.get(handle.set_update_weights_return_value.remote(_output(2)))

        assert _outcome_of(await group.update_weights(info=info)) == _outcome_of(_output(2))

        for handle in handles:
            calls = [c for c in ray.get(handle.get_calls.remote()) if c[0] == "update_weights"]
            assert all("weight_version" not in c[2] for c in calls)


class TestUpdateWeightsCarriesTheRollout:
    @pytest.mark.parametrize("rollout_id", [3, None])
    async def test_the_first_alive_dispatch_hands_the_worker_the_rollout(self, rollout_id: int | None):
        """The weight update hooks match on the rollout, so the worker must receive the one the caller named."""
        group = await _make_alive_controller(num_cells=1)
        for handle in get_raw_actor_handles(_cell(group, 0)):
            ray.get(handle.set_update_weights_return_value.remote(_output(1)))

        output = await group.update_weights(info=SimpleNamespace(engine_cell_ids=[], snapshot_cell_id_to_hashes={}), rollout_id=rollout_id)

        for handle in get_raw_actor_handles(_cell(group, 0)):
            [update_call] = [c for c in ray.get(handle.get_calls.remote()) if c[0] == "update_weights"]
            assert update_call[2]["rollout_id"] == rollout_id
            assert update_call[2]["debug_weight_update_id"] == output.debug_weight_update_id

    async def test_every_alive_dispatch_hands_each_sender_the_rollout_and_one_update_id(self):
        """Every sender of a split update must see the same rollout and update ID as the others."""
        cells = [_FakeTrainerCell(cell_index=i) for i in range(2)]
        controller = _make_partial_target_controller(cells)

        output = await controller.update_weights(info=_make_engines(4), rollout_id=3)

        received = [kwargs for cell in cells for kwargs in cell.received_kwargs]
        assert [kwargs["rollout_id"] for kwargs in received] == [3, 3]
        assert {kwargs["debug_weight_update_id"] for kwargs in received} == {output.debug_weight_update_id}


class _FakeTrainerCell:
    def __init__(
        self,
        *,
        cell_index: int,
        output: WeightUpdateOutput | None = None,
        error: Exception | None = None,
        is_alive: bool = True,
    ) -> None:
        self.cell_id = f"trainer-engine-actor-{cell_index:05d}"
        self.cell_index = cell_index
        self.is_alive = is_alive
        self.killed = False
        self.received_infos: list[UpdatableEngines] = []
        self.received_timeouts: list[float | None] = []
        self.received_kwargs: list[dict[str, object]] = []
        self._output = output if output is not None else _output(5)
        self._error = error

    async def execute(self, fn_name: str, *, timeout: float | None = None, **kwargs: object) -> list:
        self.received_infos.append(kwargs["info"])
        self.received_timeouts.append(timeout)
        self.received_kwargs.append(kwargs)
        if self._error is not None:
            self.is_alive = False
            raise self._error
        return [self._output, self._output]

    async def mark_errored_and_kill(self) -> None:
        self.killed = True
        self.is_alive = False


def _make_partial_target_controller(cells: list[_FakeTrainerCell], *, timeout: float | None = 60.0):
    controller = TrainerController.__new__(TrainerController)
    controller.args = SimpleNamespace(
        debug_train_only=False,
        debug_rollout_only=False,
        trainer_model_id=None,
        colocate=False,
        update_weight_transfer_mode="p2p",
        update_weights_timeout=timeout,
    )
    controller._trainer_id = "trainer-0"
    controller._cells_by_id = {cell.cell_id: cell for cell in cells}
    controller._debug_trainer_load_state_timestamp = None
    return controller


def _make_engines(num_engines: int) -> UpdatableEngines:
    cell_ids = [f"rollout-{i}" for i in range(num_engines)]
    return UpdatableEngines(
        rollout_engines=[MagicMock() for _ in cell_ids],
        engine_gpu_counts=[1] * len(cell_ids),
        engine_gpu_offsets=list(range(len(cell_ids))),
        engine_cell_ids=cell_ids,
        snapshot_cell_id_to_hashes={cell_id: "workers-hash" for cell_id in cell_ids},
    )


def _targets_of(cell: _FakeTrainerCell) -> list[list[str]]:
    return [info.engine_cell_ids for info in cell.received_infos]


class TestUpdateWeightsFromEveryAliveCell:
    async def test_the_targets_are_split_disjointly_across_the_alive_trainer_cells(self):
        """Sending every engine from one cell wastes the other senders' links and their bandwidth."""
        cells = [_FakeTrainerCell(cell_index=i) for i in range(2)]
        controller = _make_partial_target_controller(cells)

        await controller.update_weights(info=_make_engines(4))

        assert _targets_of(cells[0]) == [["rollout-0", "rollout-1"]]
        assert _targets_of(cells[1]) == [["rollout-2", "rollout-3"]]

    async def test_an_uneven_share_goes_to_the_earliest_trainer_cells(self):
        """Every share must stay within one of the others so no sender becomes the straggler."""
        cells = [_FakeTrainerCell(cell_index=i) for i in range(2)]
        controller = _make_partial_target_controller(cells)

        await controller.update_weights(info=_make_engines(3))

        assert _targets_of(cells[0]) == [["rollout-0", "rollout-1"]]
        assert _targets_of(cells[1]) == [["rollout-2"]]

    async def test_a_trainer_cell_left_without_a_target_is_not_asked_to_send(self):
        """An update_weights call with an empty engine list would make the worker set up a transfer to nobody."""
        cells = [_FakeTrainerCell(cell_index=i) for i in range(3)]
        controller = _make_partial_target_controller(cells)

        await controller.update_weights(info=_make_engines(2))

        assert _targets_of(cells[0]) == [["rollout-0"]]
        assert _targets_of(cells[1]) == [["rollout-1"]]
        assert _targets_of(cells[2]) == []

    async def test_the_targets_are_shared_only_among_the_cells_that_are_alive(self):
        """A share handed to a dead cell would leave its engines on the previous weights."""
        cells = [
            _FakeTrainerCell(cell_index=0, is_alive=False),
            _FakeTrainerCell(cell_index=1),
            _FakeTrainerCell(cell_index=2),
        ]
        controller = _make_partial_target_controller(cells)

        await controller.update_weights(info=_make_engines(4))

        assert _targets_of(cells[0]) == []
        assert _targets_of(cells[1]) == [["rollout-0", "rollout-1"]]
        assert _targets_of(cells[2]) == [["rollout-2", "rollout-3"]]

    async def test_every_sender_is_given_the_configured_transfer_timeout(self):
        """A transfer that hangs forever would stall the whole run instead of failing its own share."""
        cells = [_FakeTrainerCell(cell_index=i) for i in range(2)]
        controller = _make_partial_target_controller(cells, timeout=123.0)

        await controller.update_weights(info=_make_engines(2))

        assert cells[0].received_timeouts == [123.0]
        assert cells[1].received_timeouts == [123.0]

    async def test_the_reports_of_every_sender_are_merged_into_one_answer(self):
        """The driver acts on a single report, so a failure seen by only one sender must survive the merge."""
        cells = [
            _FakeTrainerCell(cell_index=0, output=_output(7, "rollout-1")),
            _FakeTrainerCell(cell_index=1, output=_output(7)),
        ]
        controller = _make_partial_target_controller(cells)

        output = await controller.update_weights(info=_make_engines(4))

        assert output.weight_version == 7
        assert set(output.failed_cell_ids) == {"rollout-1"}

    async def test_an_update_with_no_trainer_cell_alive_is_not_worth_retrying(self):
        """Retrying inside the controller cannot bring a cell back, and the driver must heal the pool instead."""
        cells = [_FakeTrainerCell(cell_index=i, is_alive=False) for i in range(2)]
        controller = _make_partial_target_controller(cells)

        with pytest.raises(NonRetryableError, match="No alive cells"):
            await controller.update_weights(info=_make_engines(2))

    async def test_an_update_window_with_no_engines_answers_an_empty_report(self):
        """A run whose rollout cells are all down must not be reported as a failed weight update."""
        cells = [_FakeTrainerCell(cell_index=i) for i in range(2)]
        controller = _make_partial_target_controller(cells)

        output = await controller.update_weights(info=_make_engines(0))

        assert output == WeightUpdateOutput(weight_version=None, failed_cell_ids=())
        assert _targets_of(cells[0]) == []

    async def test_a_broadcast_run_still_sends_from_a_single_cell(self):
        """A broadcast reaches every engine at once, so splitting its targets would send the weights twice."""
        cells = [_FakeTrainerCell(cell_index=i) for i in range(2)]
        controller = _make_partial_target_controller(cells)
        controller.args.update_weight_transfer_mode = "broadcast"
        controller._execute_first_alive = AsyncMock(return_value=[_output(3), _output(3)])
        info = _make_engines(4)

        assert await controller.update_weights(info=info) == _output(3)

        controller._execute_first_alive.assert_awaited_once_with(
            "update_weights", timeout=60.0, info=info, debug_weight_update_id=ANY, rollout_id=None
        )
        assert _targets_of(cells[0]) == []

    async def test_each_share_carries_the_layout_and_snapshot_of_its_own_engines(self) -> None:
        """A share whose gpu offsets or hashes belong to other engines would write into the wrong ranks."""
        cells = [_FakeTrainerCell(cell_index=i) for i in range(2)]
        controller = _make_partial_target_controller(cells)
        engines = [MagicMock() for _ in range(3)]
        info = UpdatableEngines(
            rollout_engines=engines,
            engine_gpu_counts=[1, 2, 4],
            engine_gpu_offsets=[0, 1, 3],
            engine_cell_ids=["rollout-0", "rollout-1", "rollout-2"],
            snapshot_cell_id_to_hashes={"rollout-0": "hash-0", "rollout-1": "hash-1", "rollout-2": "hash-2"},
        )

        await controller.update_weights(info=info)

        [first_share] = cells[0].received_infos
        [second_share] = cells[1].received_infos
        assert first_share.rollout_engines == engines[:2]
        assert first_share.engine_gpu_counts == [1, 2]
        assert first_share.engine_gpu_offsets == [0, 1]
        assert first_share.snapshot_cell_id_to_hashes == {"rollout-0": "hash-0", "rollout-1": "hash-1"}
        assert second_share.rollout_engines == engines[2:]
        assert second_share.engine_gpu_counts == [4]
        assert second_share.engine_gpu_offsets == [3]
        assert second_share.snapshot_cell_id_to_hashes == {"rollout-2": "hash-2"}

    async def test_a_colocated_run_still_sends_from_a_single_cell(self):
        """Colocation puts the engines on the trainer's own gpus, where there is nothing to split."""
        cells = [_FakeTrainerCell(cell_index=i) for i in range(2)]
        controller = _make_partial_target_controller(cells)
        controller.args.colocate = True
        controller._execute_first_alive = AsyncMock(return_value=[_output(3)])

        assert await controller.update_weights(info=_make_engines(4)) == _output(3)

        assert _targets_of(cells[0]) == []


class TestUpdateWeightsGivesUpOnADeadTrainersTargets:
    async def test_the_targets_of_a_sender_that_raised_are_all_reported_failed(self):
        """Those engines may hold a half-written model, and serving from them would poison the rollouts."""
        cells = [
            _FakeTrainerCell(cell_index=0, error=RuntimeError("trainer died")),
            _FakeTrainerCell(cell_index=1, output=_output(4)),
        ]
        controller = _make_partial_target_controller(cells)

        output = await controller.update_weights(info=_make_engines(2))

        assert set(output.failed_cell_ids) == {"rollout-0"}

    async def test_a_surviving_sender_still_publishes_the_version_it_reached(self):
        """The engines it did update serve the new weights, which the driver can only publish if it is told."""
        cells = [
            _FakeTrainerCell(cell_index=0, error=RuntimeError("trainer died")),
            _FakeTrainerCell(cell_index=1, output=_output(4)),
        ]
        controller = _make_partial_target_controller(cells)

        assert (await controller.update_weights(info=_make_engines(2))).weight_version == 4

    async def test_a_dead_senders_targets_and_a_survivors_own_failure_are_all_reported(self) -> None:
        """Dropping either set on the merge would leave a stale or half-written engine in service."""
        cells = [
            _FakeTrainerCell(cell_index=0, error=RuntimeError("trainer died")),
            _FakeTrainerCell(cell_index=1, output=_output(4, "rollout-3")),
        ]
        controller = _make_partial_target_controller(cells)

        output = await controller.update_weights(info=_make_engines(4))

        assert output.weight_version == 4
        assert set(output.failed_cell_ids) == {"rollout-0", "rollout-1", "rollout-3"}

    async def test_the_next_update_splits_every_target_among_the_survivors_only(self) -> None:
        """A share left with the dead sender would keep those engines on the old weights on every later update."""
        cells = [
            _FakeTrainerCell(cell_index=0, error=RuntimeError("trainer died")),
            _FakeTrainerCell(cell_index=1, output=_output(4)),
            _FakeTrainerCell(cell_index=2, output=_output(4)),
        ]
        controller = _make_partial_target_controller(cells)
        await controller.update_weights(info=_make_engines(3))

        output = await controller.update_weights(info=_make_engines(4))

        assert (output.weight_version, output.failed_cell_ids) == (4, ())
        assert _targets_of(cells[0]) == [["rollout-0"]]
        assert _targets_of(cells[1]) == [["rollout-1"], ["rollout-0", "rollout-1"]]
        assert _targets_of(cells[2]) == [["rollout-2"], ["rollout-2", "rollout-3"]]

    async def test_losing_every_sender_raises_the_first_failure(self):
        """No engine got the weights and no cell is left to retry on, so the run must surface the error."""
        cells = [
            _FakeTrainerCell(cell_index=0, error=RuntimeError("first failure")),
            _FakeTrainerCell(cell_index=1, error=RuntimeError("second failure")),
        ]
        controller = _make_partial_target_controller(cells)

        with pytest.raises(RuntimeError, match="first failure"):
            await controller.update_weights(info=_make_engines(2))

    async def test_a_healthy_sender_with_no_share_keeps_the_run_from_exiting(self):
        """A trainer cell that was handed no target is still available to heal the pool on the next update."""
        cells = [
            _FakeTrainerCell(cell_index=0, error=RuntimeError("first failure")),
            _FakeTrainerCell(cell_index=1, error=RuntimeError("second failure")),
            _FakeTrainerCell(cell_index=2),
        ]
        controller = _make_partial_target_controller(cells)

        with pytest.raises(NonRetryableError, match="No inference cell received the weights"):
            await controller.update_weights(info=_make_engines(2))

    async def test_no_engine_receiving_the_weights_is_not_worth_retrying(self):
        """Every rollout cell now serves stale or half-written weights, which only a fresh rollout pool fixes."""
        cells = [
            _FakeTrainerCell(cell_index=0, output=_output(4, "rollout-0")),
            _FakeTrainerCell(cell_index=1, output=_output(4, "rollout-1")),
        ]
        controller = _make_partial_target_controller(cells)

        with pytest.raises(NonRetryableError, match="No inference cell received the weights"):
            await controller.update_weights(info=_make_engines(2))

    async def test_one_engine_out_of_several_failing_is_reported_rather_than_raised(self):
        """The rest of the fleet serves the new weights, and the driver only has to drop the one that failed."""
        cells = [
            _FakeTrainerCell(cell_index=0, output=_output(4, "rollout-0")),
            _FakeTrainerCell(cell_index=1, output=_output(4)),
        ]
        controller = _make_partial_target_controller(cells)

        output = await controller.update_weights(info=_make_engines(2))

        assert output == _output(4, "rollout-0")


class TestBlameTheSenderThatReachedNoneOfItsTargets:
    async def test_a_sender_that_missed_all_of_its_several_targets_is_killed(self):
        """A trainer with a dead link would otherwise be handed a fresh share on every update until none are left."""
        cells = [
            _FakeTrainerCell(cell_index=0, output=_output(4, "rollout-0", "rollout-1")),
            _FakeTrainerCell(cell_index=1, output=_output(4)),
        ]
        controller = _make_partial_target_controller(cells)

        await controller.update_weights(info=_make_engines(4))

        assert cells[0].killed
        assert not cells[1].killed

    async def test_a_sender_that_raised_on_all_of_its_several_targets_is_killed(self):
        """A raise is at least as damning as a report of failure, and the cell must not be reused either way."""
        cells = [
            _FakeTrainerCell(cell_index=0, error=RuntimeError("trainer died")),
            _FakeTrainerCell(cell_index=1, output=_output(4)),
        ]
        controller = _make_partial_target_controller(cells)

        await controller.update_weights(info=_make_engines(3))

        assert cells[0].killed

    async def test_a_sender_with_a_single_failed_target_is_left_alone(self):
        """One unreachable engine is far more likely the engine's fault than the sender's."""
        cells = [
            _FakeTrainerCell(cell_index=0, output=_output(4, "rollout-0")),
            _FakeTrainerCell(cell_index=1, output=_output(4)),
        ]
        controller = _make_partial_target_controller(cells)

        await controller.update_weights(info=_make_engines(2))

        assert not cells[0].killed

    async def test_a_sender_that_reached_some_of_its_targets_is_left_alone(self):
        """Its link demonstrably works, so killing it would throw away a healthy trainer cell."""
        cells = [
            _FakeTrainerCell(cell_index=0, output=_output(4, "rollout-0")),
            _FakeTrainerCell(cell_index=1, output=_output(4)),
        ]
        controller = _make_partial_target_controller(cells)

        await controller.update_weights(info=_make_engines(4))

        assert not cells[0].killed

    async def test_a_killed_sender_is_handed_no_share_on_the_next_update(self) -> None:
        """Handing the blamed sender another share would lose those engines again on every update."""
        cells = [
            _FakeTrainerCell(cell_index=0, output=_output(4, "rollout-0", "rollout-1")),
            _FakeTrainerCell(cell_index=1, output=_output(4)),
        ]
        controller = _make_partial_target_controller(cells)
        await controller.update_weights(info=_make_engines(4))

        await controller.update_weights(info=_make_engines(4))

        assert cells[0].killed
        assert _targets_of(cells[0]) == [["rollout-0", "rollout-1"]]
        assert _targets_of(cells[1]) == [
            ["rollout-2", "rollout-3"],
            ["rollout-0", "rollout-1", "rollout-2", "rollout-3"],
        ]

    async def test_the_senders_are_blamed_before_the_update_is_declared_a_total_loss(self):
        """Leaving a broken sender alive because the update failed outright would repeat the failure next time."""
        cells = [
            _FakeTrainerCell(cell_index=0, output=_output(4, "rollout-0", "rollout-1")),
            _FakeTrainerCell(cell_index=1, output=_output(4, "rollout-2", "rollout-3")),
        ]
        controller = _make_partial_target_controller(cells)

        with pytest.raises(NonRetryableError, match="No inference cell received the weights"):
            await controller.update_weights(info=_make_engines(4))

        assert cells[0].killed and cells[1].killed


# ========================= weight update result events ========================


class TestWeightUpdateResultEvent:
    @pytest.fixture
    def _event_log_dir(self, tmp_path: Path):
        set_event_logger(EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main")))
        try:
            yield tmp_path
        finally:
            set_event_logger(None)

    @staticmethod
    def _results(log_dir: Path) -> list[WeightUpdateResultEvent]:
        return [e for e in read_events(log_dir) if isinstance(e, WeightUpdateResultEvent)]

    @staticmethod
    def _controller(cells: list[_FakeTrainerCell]) -> TrainerController:
        controller = _make_partial_target_controller(cells)
        controller._debug_trainer_load_state_timestamp = 123.5
        return controller

    async def test_a_partial_success_logs_the_updated_and_failed_cells_under_one_update_id(self, _event_log_dir: Path):
        """The result names exactly the cells that took the version, with the id every sender and the output used."""
        cells = [
            _FakeTrainerCell(cell_index=0, output=_output(5, "rollout-1")),
            _FakeTrainerCell(cell_index=1, output=_output(5)),
        ]
        info = _make_engines(4)

        output = await self._controller(cells).update_weights(info=info, rollout_id=3)

        [event] = self._results(_event_log_dir)
        assert event.debug_weight_update_id == output.debug_weight_update_id
        assert {kwargs["debug_weight_update_id"] for cell in cells for kwargs in cell.received_kwargs} == {
            event.debug_weight_update_id
        }
        assert event.rollout_id == 3
        assert (event.candidate_version, event.published_version) == (5, 5)
        assert event.updated_cell_ids == ["rollout-0", "rollout-2", "rollout-3"]
        assert event.failed_cell_ids == ["rollout-1"]
        assert event.snapshot_cell_id_to_hashes == info.snapshot_cell_id_to_hashes
        assert event.debug_trainer_load_state_timestamp == output.debug_trainer_load_state_timestamp == 123.5

    async def test_a_sender_that_raised_puts_its_whole_share_in_the_failed_cells(self, _event_log_dir: Path):
        """A dead sender's targets never took the version, so they must not be reported updated."""
        cells = [
            _FakeTrainerCell(cell_index=0, error=RuntimeError("trainer died")),
            _FakeTrainerCell(cell_index=1, output=_output(5)),
        ]

        await self._controller(cells).update_weights(info=_make_engines(4))

        [event] = self._results(_event_log_dir)
        assert event.updated_cell_ids == ["rollout-2", "rollout-3"]
        assert sorted(event.failed_cell_ids) == ["rollout-0", "rollout-1"]
        assert event.published_version == 5

    async def test_an_update_without_targets_publishes_nothing(self, _event_log_dir: Path):
        """No engine took any weights, so a published version would claim coverage that does not exist."""
        await self._controller([_FakeTrainerCell(cell_index=0)]).update_weights(info=_make_engines(0))

        [event] = self._results(_event_log_dir)
        assert event.updated_cell_ids == [] and event.failed_cell_ids == []
        assert event.published_version is None

    async def test_a_skipped_broadcast_logs_no_published_version(self, _event_log_dir: Path):
        """Workers that skipped the broadcast answer no version, which must not be recorded as published."""
        cells = [_FakeTrainerCell(cell_index=0, output=_output(None))]

        await self._controller(cells).update_weights(info=_make_engines(2))

        [event] = self._results(_event_log_dir)
        assert (event.candidate_version, event.published_version) == (None, None)

    async def test_each_update_gets_a_fresh_id(self, _event_log_dir: Path):
        """Reusing an id would let one update's checksum record cover another update."""
        controller = self._controller([_FakeTrainerCell(cell_index=0)])

        first = await controller.update_weights(info=_make_engines(1))
        second = await controller.update_weights(info=_make_engines(1))

        assert [e.debug_weight_update_id for e in self._results(_event_log_dir)] == [
            first.debug_weight_update_id,
            second.debug_weight_update_id,
        ]
        assert first.debug_weight_update_id != second.debug_weight_update_id

    async def test_the_first_alive_dispatch_logs_the_failed_cells_its_worker_reported(self, _event_log_dir: Path):
        """The broadcast path must log the same outcome shape as the split path."""
        group = TrainerController.__new__(TrainerController)
        group.args = _make_broadcast_args()
        group._trainer_id = "trainer-0"
        group._debug_trainer_load_state_timestamp = 7.0
        group._execute_first_alive = AsyncMock(return_value=[_output(2, "rollout-1"), _output(2, "rollout-1")])

        output = await group.update_weights(info=_make_engines(2), rollout_id=1)

        [event] = self._results(_event_log_dir)
        assert event.debug_weight_update_id == output.debug_weight_update_id
        assert group._execute_first_alive.await_args.kwargs["debug_weight_update_id"] == output.debug_weight_update_id
        assert (event.updated_cell_ids, event.failed_cell_ids) == (["rollout-0"], ["rollout-1"])
        assert event.debug_trainer_load_state_timestamp == 7.0

    async def test_no_event_is_logged_without_an_event_logger(self):
        """A run without the event logger still gets its output instead of failing on the log call."""
        output = await self._controller([_FakeTrainerCell(cell_index=0)]).update_weights(info=_make_engines(1))

        assert output.weight_version == 5

    async def test_reloading_the_trainer_state_starts_a_new_lineage(self, _event_log_dir: Path):
        """A load_state rewinds the weights, so later results must not share the earlier lineage."""
        info = SimpleNamespace(engine_cell_ids=["rollout-0"], snapshot_cell_id_to_hashes={"rollout-0": "h"})
        group = await _make_alive_controller(num_cells=1)
        for handle in get_raw_actor_handles(_cell(group, 0)):
            ray.get(handle.set_update_weights_return_value.remote(_output(1)))
        before = await group.update_weights(info=info)

        await group.load_state()
        after = await group.update_weights(info=info)

        first, second = self._results(_event_log_dir)
        assert after.debug_trainer_load_state_timestamp > before.debug_trainer_load_state_timestamp
        assert first.debug_trainer_load_state_timestamp == before.debug_trainer_load_state_timestamp
        assert second.debug_trainer_load_state_timestamp == after.debug_trainer_load_state_timestamp


class TestStepEndRecordsCellIncarnations:
    def test_the_step_end_names_the_incarnation_of_every_participating_cell(self, tmp_path: Path):
        """A peer-progress check needs the exact incarnations that trained, not just cell indices."""
        set_event_logger(EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main")))
        try:
            cells = [
                SimpleNamespace(cell_index=0, cell_id="trainer-0", workers_hash="hash-0"),
                SimpleNamespace(cell_index=2, cell_id="trainer-2", workers_hash="hash-2b"),
            ]
            _make_controller(num_cells=3)._log_step_end_event(
                rollout_id=4, attempt=0, snapshot_alive_cells=cells, results=[[NORMAL], RuntimeError("boom")]
            )
        finally:
            set_event_logger(None)

        [event] = [e for e in read_events(tmp_path) if isinstance(e, TrainGroupStepEndEvent)]
        assert event.cell_incarnations == {"trainer-0": "hash-0", "trainer-2": "hash-2b"}
        assert event.cell_outcomes == {0: [TrainStepOutcome.NORMAL], 2: "error"}
