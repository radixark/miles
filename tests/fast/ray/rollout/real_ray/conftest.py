"""Fixtures for tests that drive ``MockSGLangEngine`` as a real Ray actor."""

from __future__ import annotations

import pytest
import ray

# Production ServerGroup.start_engines hard-codes num_gpus=0.2, num_cpus=0.2 on
# the actor's .options(...) call, so each PG bundle must satisfy that.
_PER_ENGINE_NUM_CPUS = 0.2
_PER_ENGINE_NUM_GPUS = 0.2


@pytest.fixture
def placement_group_factory(ray_local_mode):
    """Yields ``make(num_engines) -> (pg, bundle_indices, gpu_ids)`` matching
    what ``ServerGroup.pg`` expects. PGs are torn down on teardown."""
    created: list = []

    def _make(num_engines: int) -> tuple:
        bundles = [{"CPU": _PER_ENGINE_NUM_CPUS, "GPU": _PER_ENGINE_NUM_GPUS} for _ in range(num_engines)]
        pg = ray.util.placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())
        created.append(pg)
        return (pg, list(range(num_engines)), list(range(num_engines)))

    yield _make

    for pg in created:
        try:
            ray.util.remove_placement_group(pg)
        except Exception:
            pass


@pytest.fixture
def mock_engine_class(ray_local_mode):
    """Unwrapped MockSGLangEngine class.

    Production wraps via ``ray.remote(SGLangEngine)``; substituting the
    already-wrapped class would double-wrap, so callers monkeypatch the
    unwrapped class inside ``miles.ray.rollout.server_cell``."""
    from miles.utils.test_utils.mock_sglang_engine import MockSGLangEngine

    return MockSGLangEngine.__ray_actor_class__


@pytest.fixture
def patched_sglang_engine(monkeypatch, mock_engine_class):
    """Replace SGLangEngine with the mock; the real addr allocator runs, and
    each mock engine serves HTTP on the port it is allocated, so the urls
    ServerGroup derives from the allocator actually serve requests."""
    import miles.ray.rollout.server_cell as cell_mod

    monkeypatch.setattr(cell_mod, "SGLangEngine", mock_engine_class)
