import pytest


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
