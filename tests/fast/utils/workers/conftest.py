import pytest
from tests.fast.utils.workers.fake_ray import FakeRayCluster, FakeRayModule


@pytest.fixture
def fake_ray_cluster(monkeypatch) -> FakeRayCluster:
    """In-process stand-in for Ray, letting the manager's whole launch pipeline run without a cluster."""
    import miles.utils.workers.ray_worker_manager as ray_worker_manager_mod

    cluster = FakeRayCluster()
    fake_ray = FakeRayModule(cluster=cluster)
    monkeypatch.setattr(ray_worker_manager_mod, "ray", fake_ray)
    monkeypatch.setattr(
        ray_worker_manager_mod,
        "compute_ray_pin_head_options",
        lambda _head_node_id=None: {"scheduling_strategy": "head-affinity"},
    )
    monkeypatch.setattr(ray_worker_manager_mod, "get_head_node_id", lambda: "head-node")
    return cluster
