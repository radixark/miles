from types import SimpleNamespace

import pytest

from miles.ray.specs.train import compute_trainer_pool_id
from miles.ray.train.group import TrainerController
from miles.utils.workers.worker_provider.base import CellInfo

pytestmark = pytest.mark.asyncio

_POOL_ID = compute_trainer_pool_id("actor")


def _make_controller(*, num_cells: int = 2) -> RayTrainGroup:
    group = object.__new__(RayTrainGroup)
    group.args = SimpleNamespace(
        indep_dp=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=num_cells,
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
