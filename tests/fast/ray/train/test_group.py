from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import ray
from tests.fast.ray.train.dummy_actor import DummyTrainActor

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

        await group._execute_first("save_model", 42)

        for handle in group._cells[0]._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "save_model" for c in calls)

        for cell in group._cells[1:]:
            for handle in cell._get_actor_handles():
                calls = ray.get(handle.get_calls.remote())
                assert not any(c[0] == "save_model" for c in calls)


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
