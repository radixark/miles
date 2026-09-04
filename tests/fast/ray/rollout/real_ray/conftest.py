"""Fixtures for tests that drive ``MockSGLangEngine`` as a real Ray actor."""

from __future__ import annotations

import pytest
import ray

# Production launch_sglang_ray_actor hard-codes num_gpus=0.2, num_cpus=0.2 on
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


def build_cells(
    *,
    pg_tuple: tuple,
    num_cells: int = 2,
    num_gpus_per_engine: int = 1,
    rank_offset: int = 0,
    gpu_offset: int = 0,
    debug_train_only: bool = False,
    worker_type: str = "regular",
    needs_offload: bool = False,
    update_weights: bool = True,
    model_path: str | None = None,
):
    """Build configured cells for one placement group.

    ``rank_offset`` is a global rank (engines of several groups share it), while
    gpu indices are positions inside this placement group, so the two offsets
    are independent: a group with its own pg still starts at gpu index 0.
    """
    from tests.fast.ray.rollout.conftest import make_args

    from miles.ray.rollout.server_cell import ServerCell, compute_nodes_per_engine

    args = make_args(num_gpus_per_node=8, debug_train_only=debug_train_only)
    nodes_per_engine = compute_nodes_per_engine(num_gpus_per_engine=num_gpus_per_engine, num_gpus_per_node=8)
    num_gpu_per_engine = min(num_gpus_per_engine, 8)
    return [
        ServerCell(
            args=args,
            cell_id=f"cell-{cell_index}",
            num_nodes=nodes_per_engine,
            pg=pg_tuple,
            num_gpus_per_engine=num_gpus_per_engine,
            worker_type=worker_type,
            rank_offset=rank_offset + cell_index * nodes_per_engine,
            gpu_offset=gpu_offset + cell_index * nodes_per_engine * num_gpu_per_engine,
            needs_offload=needs_offload,
            model_path=model_path,
            update_weights=update_weights,
        )
        for cell_index in range(num_cells)
    ]


async def start_cells(cells, allocator=None, *, mark_alive: bool = False):
    """Start every cell's engines through one shared allocator."""
    import asyncio

    from miles.utils.workers.addr_allocator import PortAllocator

    allocator = allocator if allocator is not None else PortAllocator()
    await asyncio.gather(*[cell.start_engines(allocator) for cell in cells])
    if mark_alive:
        for cell in cells:
            cell._mark_alive()


def kill_cells(cells) -> None:
    for cell in cells:
        if cell.is_allocated:
            for actor_handle in cell.actor_handles:
                try:
                    ray.kill(actor_handle)
                except Exception:
                    pass


@pytest.fixture
def mock_engine_class(ray_local_mode):
    """Unwrapped MockSGLangEngine class.

    Production wraps via ``ray.remote(CommandActor)``; substituting the
    already-wrapped class would double-wrap, so callers monkeypatch the
    unwrapped class inside ``miles.ray.rollout.server_cell``."""
    from miles.utils.test_utils.mock_sglang_engine import MockSGLangEngine

    return MockSGLangEngine.__ray_actor_class__


@pytest.fixture
def patched_sglang_engine(monkeypatch, mock_engine_class):
    """Replace the engine CommandActor with the mock; the real addr allocator runs, and
    each mock engine serves HTTP on the port it is allocated, so the urls
    the cell derives from the allocator actually serve requests."""
    import miles.ray.rollout.server_cell as cell_mod

    monkeypatch.setattr(cell_mod, "CommandActor", mock_engine_class)
