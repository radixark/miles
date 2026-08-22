from types import SimpleNamespace

import pytest
from tests.fast.utils.workers.fake_ray import FakeRayCluster, FakeRayModule


@pytest.fixture
def fake_ray_cluster(monkeypatch) -> FakeRayCluster:
    """In-process stand-in for Ray, letting the manager's whole launch pipeline run without a cluster."""
    import miles.utils.workers.addr_allocator as addr_allocator_mod
    import miles.utils.workers.ray_worker_manager as ray_worker_manager_mod

    cluster = FakeRayCluster()
    fake_ray = FakeRayModule(cluster=cluster)
    monkeypatch.setattr(ray_worker_manager_mod, "ray", fake_ray)
    monkeypatch.setattr(addr_allocator_mod, "ray", fake_ray)
    return cluster


@pytest.fixture
def patch_ray_get(monkeypatch):
    """Make ``ray.get(remote_call(...))`` return the MagicMock's value directly,
    so allocator-driven tests don't need a real Ray cluster."""
    import miles.utils.workers.addr_allocator as mod

    monkeypatch.setattr(mod.ray, "get", lambda x: x)


@pytest.fixture
def patch_ray_get_failure(monkeypatch):
    """Make ``ray.get(...)`` raise, mimicking a probe that is submitted
    successfully but fails while its result is retrieved."""
    import miles.utils.workers.addr_allocator as mod

    def _raise(_object_ref):
        raise RuntimeError("free port probe failed")

    monkeypatch.setattr(mod.ray, "get", _raise)


def worker_manager_args(**overrides) -> SimpleNamespace:
    """The slice of a training run's args the worker manager reads when it configures its logger."""
    return SimpleNamespace(**{"save_debug_event_data": None, "env_report": "", **overrides})
