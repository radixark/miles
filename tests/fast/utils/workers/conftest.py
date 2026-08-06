import pytest


@pytest.fixture
def patch_ray_get(monkeypatch):
    """Make ``ray.get(remote_call(...))`` return the MagicMock's value directly,
    so allocator-driven tests don't need a real Ray cluster."""
    import miles.utils.workers.addr_allocator as mod

    monkeypatch.setattr(mod.ray, "get", lambda x: x)
