from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import ray
from tests.fast.ray.train.dummy_actor import DummyTrainActor

from miles.backends.megatron_utils.types import TrainStepOutcome
from miles.ray.train.group import RayTrainGroup
from miles.utils.witness.allocator import WitnessIdAllocator

pytestmark = pytest.mark.asyncio

_DUMMY_DATA_PACK = {"data_ref": "data", "sample_indices": [0]}


def _make_mock_args(
    *,
    indep_dp: bool = True,
    enable_witness: bool = False,
    gpus_per_cell: int = 1,
) -> SimpleNamespace:
    # Use SimpleNamespace (not MagicMock) so the args object is picklable. RayTrainCell.init
    # passes self.args through Ray to the remote actor; pickling a MagicMock blows the
    # recursion limit because its __getattr__ creates new sub-mocks indefinitely.
    return SimpleNamespace(
        indep_dp=indep_dp,
        enable_witness=enable_witness,
        witness_buffer_size=100,
        trainer_heartbeat_checker_interval=10.0,
        trainer_heartbeat_checker_timeout=10.0,
        trainer_heartbeat_checker_first_wait=300.0,
        trainer_heartbeat_checker_failure_threshold=3,
        ci_ft_test_actions=None,
        debug_train_only=False,
        debug_rollout_only=False,
        # compute_megatron_world_size_except_dp(args) = TP * PP * CP. Set CP to
        # gpus_per_cell so RayTrainGroup computes num_cells correctly.
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=gpus_per_cell,
    )


@pytest.fixture(autouse=True)
def _patch_actor_alloc():
    """Persist allocate_gpus_for_actor patch across the whole test (incl. healing path).

    Previously _make_group used `with patch(...)` which expired when _make_group
    returned, so any later `allocate_gpus_for_actor` call during _refresh_cells
    healing hit the real implementation (which dereferences mock args fields).
    """

    def _alloc(*, gpus_per_cell: int, num_gpus_per_actor: float, **_kwargs) -> list:
        actor_count = max(int(gpus_per_cell // num_gpus_per_actor), 1)
        return [DummyTrainActor.remote() for _ in range(actor_count)]

    with patch("miles.ray.train.group.allocate_gpus_for_actor", side_effect=_alloc):
        yield


def _make_group(
    *,
    num_cells: int = 3,
    actor_count_per_cell: int = 1,
    rollout_manager: object | None = None,
) -> RayTrainGroup:
    """Create a RayTrainGroup through real __init__ with mocked pg and actor factory."""
    total_gpus = num_cells * actor_count_per_cell
    return RayTrainGroup(
        args=_make_mock_args(indep_dp=True, gpus_per_cell=actor_count_per_cell),
        num_nodes=1,
        num_gpus_per_node=total_gpus,
        pg=(MagicMock(), list(range(total_gpus)), list(range(total_gpus))),
        role="actor",
        with_ref=False,
        rollout_manager=rollout_manager,
    )


async def _init_group(group: RayTrainGroup) -> None:
    """Call init and wait for all cells to become alive."""
    await group.init()


async def _make_alive_group(*, num_cells: int = 3, **kwargs) -> RayTrainGroup:
    """Create a group and init all cells to alive."""
    group = _make_group(num_cells=num_cells, **kwargs)
    await _init_group(group)
    return group


class TestInit:
    def test_creates_correct_number_of_cells(self):
        group = _make_group(num_cells=3)

        assert len(group._cells) == 3
        assert [c.cell_index for c in group._cells] == [0, 1, 2]

    def test_cells_are_allocated_after_init(self):
        group = _make_group(num_cells=2)

        for cell in group._cells:
            assert cell.is_allocated
            assert not cell.is_alive

    def test_each_cell_has_own_actors(self):
        group = _make_group(num_cells=3, actor_count_per_cell=2)

        handles_per_cell = [cell._get_actor_handles() for cell in group._cells]
        assert all(len(h) == 2 for h in handles_per_cell)

        all_handles = [h for handles in handles_per_cell for h in handles]
        assert len(set(id(h) for h in all_handles)) == 6

    def test_single_cell_no_tcp_store(self):
        # indep_dp=False forces single cell regardless of TP/PP/CP product;
        # the autouse fixture handles allocate_gpus_for_actor.
        group = RayTrainGroup(
            args=_make_mock_args(indep_dp=False),
            num_nodes=1,
            num_gpus_per_node=1,
            pg=(MagicMock(), [0], [0]),
            role="actor",
            with_ref=False,
            rollout_manager=None,
        )

        assert len(group._cells) == 1
        assert group._indep_dp_store is None

    async def test_init_marks_all_cells_alive(self):
        group = _make_group(num_cells=3)

        await _init_group(group)

        for cell in group._cells:
            assert cell.is_alive
            assert cell.indep_dp_info.alive_cell_indices == [0, 1, 2]
            assert cell.indep_dp_info.alive_size == 3

        assert group._cells[0].indep_dp_info.alive_rank == 0
        assert group._cells[1].indep_dp_info.alive_rank == 1
        assert group._cells[2].indep_dp_info.alive_rank == 2


class TestExecuteFirstAlive:
    async def test_picks_first_alive_cell(self):
        group = await _make_alive_group(num_cells=3)

        await group._execute_first_alive("save_model", 42)

        for handle in group._cells[0]._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "save_model" for c in calls)

        for cell in group._cells[1:]:
            for handle in cell._get_actor_handles():
                calls = ray.get(handle.get_calls.remote())
                assert not any(c[0] == "save_model" for c in calls)

    async def test_skips_stopped_picks_next(self):
        group = await _make_alive_group(num_cells=2)
        group._cells[0]._mark_as_errored()

        await group._execute_first_alive("update_weights")

        for handle in group._cells[1]._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "update_weights" for c in calls)


class TestComputeIndepDPInfo:
    def test_all_alive(self):
        group = _make_group(num_cells=3)

        info = group._compute_indep_dp_info(cell_index=2, alive_cell_indices=[0, 1, 2])

        assert info.alive_rank == 2
        assert info.alive_size == 3
        assert info.cell_index == 2

    def test_with_gap(self):
        group = _make_group(num_cells=3)

        info = group._compute_indep_dp_info(cell_index=2, alive_cell_indices=[0, 2])

        assert info.alive_rank == 1
        assert info.alive_size == 2


class TestExecuteAllAliveAndCatch:
    async def test_skips_stopped_cells(self):
        group = await _make_alive_group(num_cells=2)
        group._cells[1]._mark_as_errored()

        await group._execute_all_alive_and_catch("train")

        for handle in group._cells[0]._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "train" for c in calls)

    async def test_asserts_on_no_alive_cells(self):
        group = await _make_alive_group(num_cells=1)
        group._cells[0]._mark_as_errored()

        with pytest.raises(AssertionError, match="No alive cells"):
            await group._execute_all_alive_and_catch("train")


class TestTrain:
    async def test_train_refreshes_and_dispatches(self):
        group = await _make_alive_group(num_cells=2)

        await group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK)

        for cell in group._cells:
            for handle in cell._get_actor_handles():
                calls = ray.get(handle.get_calls.remote())
                assert any(c[0] == "train" for c in calls)

    async def test_consecutive_train_no_reconfigure_overhead(self):
        """Multiple train calls with no state changes — no reconfigure overhead."""
        group = await _make_alive_group(num_cells=3)

        # Note init call count
        init_counts = {}
        for cell in group._cells:
            for handle in cell._get_actor_handles():
                init_counts[id(handle)] = len(ray.get(handle.get_calls.remote()))

        for step in range(3):
            await group.train(rollout_id=step, rollout_data_pack=_DUMMY_DATA_PACK)

        assert group._indep_dp_quorum_id == 0

        for cell in group._cells:
            for handle in cell._get_actor_handles():
                calls = ray.get(handle.get_calls.remote())
                new_calls = calls[init_counts[id(handle)] :]
                assert not any(c[0] == "reconfigure_indep_dp" for c in new_calls)
                train_calls = [c for c in new_calls if c[0] == "train"]
                assert len(train_calls) == 3


class TestPerCellErrorIsolation:
    async def test_one_cell_failure_marks_errored_others_ok(self):
        """One cell's actor fails during broadcast, that cell is errored, others complete normally."""
        group = await _make_alive_group(num_cells=3)

        # Step 1: Make cell 1's actors fail on train
        for handle in group._cells[1]._get_actor_handles():
            ray.get(handle.set_fail_methods.remote(["train"]))

        # Step 2: Broadcast train
        await group._execute_all_alive_and_catch("train", 0, "data")

        # Step 3: Cell 1 is errored, others alive
        assert group._cells[0].is_alive
        assert group._cells[1].is_errored
        assert group._cells[2].is_alive

        # Step 4: Other cells received train call
        for cell_idx in [0, 2]:
            for handle in group._cells[cell_idx]._get_actor_handles():
                calls = ray.get(handle.get_calls.remote())
                assert any(c[0] == "train" for c in calls)

    async def test_errored_cell_skipped_in_next_broadcast(self):
        """After marking a cell errored, subsequent broadcasts skip it."""
        group = await _make_alive_group(num_cells=2)

        # Step 1: Make cell 0 fail
        for handle in group._cells[0]._get_actor_handles():
            ray.get(handle.set_fail_methods.remote(["train"]))

        await group._execute_all_alive_and_catch("train", 0, "data")
        assert group._cells[0].is_errored

        # Step 2: Next broadcast only goes to cell 1
        await group._execute_all_alive_and_catch("train", 1, "data")

        for handle in group._cells[1]._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            train_calls = [c for c in calls if c[0] == "train"]
            assert len(train_calls) == 2


class TestExecuteFirstAliveFallback:
    async def test_first_cell_fails_retry_falls_back_to_next(self):
        """If the first alive cell fails, retry in save_model marks it errored and picks the next."""
        group = await _make_alive_group(num_cells=3)

        # Step 1: Make cell 0 fail on save_model
        for handle in group._cells[0]._get_actor_handles():
            ray.get(handle.set_fail_methods.remote(["save_model"]))

        # Step 2: save_model uses retry(lambda _: self._execute_first_alive(...))
        await group.save_model(rollout_id=42)

        # Step 3: Cell 0 errored, cell 1 handled it
        assert group._cells[0].is_errored
        assert group._cells[1].is_alive

        for handle in group._cells[1]._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "save_model" for c in calls)

    async def test_single_execute_first_alive_raises_on_failure(self):
        """A single _execute_first_alive call raises (no retry) when the first cell fails."""
        group = await _make_alive_group(num_cells=2)

        for handle in group._cells[0]._get_actor_handles():
            ray.get(handle.set_fail_methods.remote(["save_model"]))

        with pytest.raises(Exception):  # noqa: B017
            await group._execute_first_alive("save_model", 42)

        assert group._cells[0].is_errored


NORMAL = TrainStepOutcome.NORMAL
DISCARDED = TrainStepOutcome.DISCARDED_SHOULD_RETRY

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
        RayTrainGroup._check_train_one_attempt(_alive_cells_for(results), results)  # should not raise

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
            RayTrainGroup._check_train_one_attempt(_alive_cells_for(results), results)

    @pytest.mark.parametrize(
        "results",
        [
            [_ERR],  # single cell errored
            [_ERR, _ERR2],  # multiple cells all errored
        ],
    )
    def test_raises_when_all_cells_errored(self, results):
        with pytest.raises(RuntimeError, match="All cells failed"):
            RayTrainGroup._check_train_one_attempt(_alive_cells_for(results), results)

    def test_compute_attempt_outcomes_buckets_cells_by_index(self):
        """_compute_attempt_outcomes buckets each alive cell into errored / discarded / normal by index."""
        results = [_ERR, [DISCARDED], [NORMAL, NORMAL]]
        outcomes = RayTrainGroup._compute_attempt_outcomes(_alive_cells_for(results), results)
        assert outcomes == {"errored": [0], "discarded": [1], "normal": [2]}


async def _set_all_train_return(group: RayTrainGroup, value: TrainStepOutcome) -> None:
    for cell in group._cells:
        for handle in cell._get_actor_handles():
            ray.get(handle.set_train_return_value.remote(value))


def _count_train_calls(group: RayTrainGroup, cell_index: int) -> int:
    total = 0
    for handle in group._cells[cell_index]._get_actor_handles():
        calls = ray.get(handle.get_calls.remote())
        total += sum(1 for c in calls if c[0] == "train")
    return total


class TestTrainRetry:
    async def test_no_retry_on_normal(self):
        """All cells return NORMAL → no retry, train called once per cell."""
        group = await _make_alive_group(num_cells=2)

        await group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK)

        for i in range(2):
            assert _count_train_calls(group, i) == 1

    async def test_retry_on_all_discarded_then_normal(self):
        """First attempt: all DISCARDED. Second attempt: all NORMAL. Train called twice."""
        group = await _make_alive_group(num_cells=2)
        await _set_all_train_return(group, TrainStepOutcome.DISCARDED_SHOULD_RETRY)

        # After first train call, switch to NORMAL so second attempt succeeds
        async def _do_train():
            await group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK)

        import asyncio

        task = asyncio.create_task(_do_train())
        # Give first attempt time to dispatch
        await asyncio.sleep(0.3)
        await _set_all_train_return(group, TrainStepOutcome.NORMAL)
        await task

        for i in range(2):
            assert _count_train_calls(group, i) == 2

    async def test_retry_multiple_times_then_succeed(self):
        """DISCARDED 3 times, then NORMAL on 4th attempt."""
        group = await _make_alive_group(num_cells=2)

        # Use a counter-based actor to track attempts
        await _set_all_train_return(group, TrainStepOutcome.DISCARDED_SHOULD_RETRY)

        async def _do_train():
            await group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK)

        import asyncio

        task = asyncio.create_task(_do_train())

        # Wait for 3 retry rounds, then switch to NORMAL
        for _ in range(3):
            await asyncio.sleep(0.2)

        await _set_all_train_return(group, TrainStepOutcome.NORMAL)
        await task

        for i in range(2):
            assert _count_train_calls(group, i) >= 2

    async def test_cell_errored_does_not_retry_when_others_normal(self):
        """One cell errors during train but others return NORMAL → no retry.

        See _check_train_one_attempt: 'If some cells errors + all other cells claim
        normal, we do *not* retry. This may happen when some cells fails *after*
        exchanging gradients w/ others.' So alive cells get exactly 1 train call.
        """
        group = await _make_alive_group(num_cells=3)

        # Step 1: Make cell 1 fail (exception)
        for handle in group._cells[1]._get_actor_handles():
            ray.get(handle.set_fail_methods.remote(["train"]))

        # Step 2: Train completes without retry (cell 1 errored but others NORMAL)
        await group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK)

        # Step 3: Cell 1 errored, alive cells each got 1 train call (no retry)
        assert group._cells[1].is_errored
        for i in [0, 2]:
            assert _count_train_calls(group, i) == 1


class TestAllocateWitnessInfo:
    def test_returns_none_when_disabled(self):
        """When _witness_allocator is None, _allocate_witness_info returns None."""
        group = _make_group(num_cells=1)
        group._witness_allocator = None

        result = group._allocate_witness_info(rollout_id=0, attempt=0, sample_indices=[10, 20, 30])

        assert result is None

    def test_returns_witness_info_when_enabled(self):
        """When witness is enabled, _allocate_witness_info returns a WitnessInfo with correct number of ids."""
        group = _make_group(num_cells=1)
        group._witness_allocator = WitnessIdAllocator(buffer_size=100)

        with patch("miles.ray.train.group.is_event_logger_initialized", return_value=False):
            result = group._allocate_witness_info(rollout_id=0, attempt=0, sample_indices=[10, 20, 30])

        assert result is not None
        assert len(result.witness_ids) == 3
        assert isinstance(result.stale_ids, list)
