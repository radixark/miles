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
    unwrapped class inside ``miles.ray.rollout.server_group``."""
    from miles.utils.test_utils.mock_sglang_engine import MockSGLangEngine

    return MockSGLangEngine.__ray_actor_class__


@pytest.fixture
def patched_sglang_engine(monkeypatch, mock_engine_class):
    """Replace SGLangEngine with the mock + stub the addr allocator with a
    deterministic dict. The real allocator path is exercised separately by
    ``patched_sglang_engine_real_allocator`` and ``test_addr_allocator.py``."""
    import miles.ray.rollout.server_group as mod

    monkeypatch.setattr(mod, "SGLangEngine", mock_engine_class)

    from miles.ray.rollout.addr_allocator import PortCursors

    def _fake_alloc(*args, **kwargs):
        engines = kwargs["rollout_engines"]
        addr_and_ports = {}
        for rank, _ in engines:
            addr_and_ports[rank] = dict(
                host="127.0.0.1",
                port=30000 + rank,
                nccl_port=31000 + rank,
                engine_info_bootstrap_port=32000 + rank,
                dist_init_addr=f"127.0.0.1:{33000 + rank}",
            )
        return addr_and_ports, PortCursors(_values={0: 34000})

    monkeypatch.setattr(mod, "allocate_rollout_engine_addr_and_ports_normal", _fake_alloc)


@pytest.fixture
def patched_sglang_engine_real_allocator(monkeypatch, mock_engine_class):
    """Replace SGLangEngine with the mock but keep the real addr allocator,
    so the actor → driver port round-trip via
    ``_get_current_node_ip_and_free_port.remote`` runs end-to-end."""
    import miles.ray.rollout.server_group as mod

    monkeypatch.setattr(mod, "SGLangEngine", mock_engine_class)
