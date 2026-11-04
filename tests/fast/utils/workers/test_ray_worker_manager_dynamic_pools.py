from types import SimpleNamespace

from tests.fast.utils.workers.test_ray_worker_manager import _make_spec

from miles.utils.workers.ray_worker_manager import RayWorkerManager, _PoolManager


def test_implicit_pool_queries_follow_live_gpu_pools_without_hiding_explicit_static_pools() -> None:
    """An unspecified scope discovers newly added GPU pools while explicit static lookups remain available."""
    manager = object.__new__(RayWorkerManager)
    manager.scaling = SimpleNamespace()
    specs = [
        _make_spec(name="engine", num_cells=2, num_gpus_per_worker=0.25, num_gpu_slots_per_worker=1),
        _make_spec(name="trainer", num_gpus_per_worker=0.4, num_gpu_slots_per_worker=1),
        _make_spec(name="router"),
    ]
    manager._pools = {spec.name: _PoolManager.initial(spec=spec, manager=manager) for spec in specs}

    initial = manager.get_cell_infos(pool_ids=None)
    late = _make_spec(name="late-engine", num_gpus_per_worker=0.25, num_gpu_slots_per_worker=1)
    manager._pools[late.name] = _PoolManager.initial(spec=late, manager=manager)
    updated = manager.get_cell_infos(pool_ids=None)

    assert sorted(initial) == ["engine-00000", "engine-00001", "trainer-00000"]
    assert sorted(updated) == ["engine-00000", "engine-00001", "late-engine-00000", "trainer-00000"]
    assert list(manager.get_cell_infos(pool_ids=["router"])) == ["router-00000"]
    assert all(not info.alive for info in updated.values())
