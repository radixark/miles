from types import SimpleNamespace

import pytest
from tests.fast.utils.workers.test_ray_worker_manager import _make_spec

from miles.utils.workers.ray_worker_manager import RayWorkerManager, _PoolManager


class TestRayPoolCategoryQueries:
    @pytest.mark.parametrize(
        ("pool_ids", "category", "expected"),
        [
            (None, "inference_engine", ["engine-00000", "engine-00001"]),
            (None, "trainer_engine", ["trainer-00000"]),
            (None, "missing", []),
            (["engine"], "trainer_engine", []),
        ],
    )
    def test_category_queries_intersect_the_selected_pool_scope(
        self, pool_ids: list[str] | None, category: str, expected: list[str]
    ) -> None:
        """Select cells by category without widening an explicit pool scope."""
        manager = object.__new__(RayWorkerManager)
        manager.scaling = SimpleNamespace()
        specs = [
            _make_spec(name="engine", num_cells=2, num_gpus_per_worker=0.25, num_gpu_slots_per_worker=1).model_copy(
                update={"category": "inference_engine"}
            ),
            _make_spec(name="trainer", num_gpus_per_worker=0.4, num_gpu_slots_per_worker=1).model_copy(
                update={"category": "trainer_engine"}
            ),
            _make_spec(name="router"),
        ]
        manager._pools = {spec.name: _PoolManager.initial(spec=spec, manager=manager) for spec in specs}

        cells = manager.get_cell_infos(pool_ids=pool_ids, category=category)

        assert sorted(cells) == expected
        assert all(info.pool_id == cell_id.rsplit("-", 1)[0] for cell_id, info in cells.items())
