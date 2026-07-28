from __future__ import annotations

import pytest
import ray
from tests.fast.ray.rollout.conftest import chunk_engines_into_cells, make_args

from miles.ray.rollout.addr_allocator import PortAllocator
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import flatten_cells
from miles.ray.rollout.server_engine import ServerEngine
from miles.ray.rollout.server_group import ServerGroup


def _build_group(
    *,
    pg_tuple: tuple,
    num_engines: int = 2,
    num_gpus_per_engine: int = 1,
    gpu_offset: int = 0,
    rank_offset: int = 0,
    needs_offload: bool = False,
) -> ServerGroup:
    args = make_args(num_gpus_per_node=8)
    engines = [ServerEngine() for _ in range(num_engines)]
    return ServerGroup(
        args=args,
        cells=chunk_engines_into_cells(
            engines,
            num_gpus_per_engine=num_gpus_per_engine,
            num_gpus_per_node=8,
            args=args,
            pg=pg_tuple,
            rank_offset=rank_offset,
            gpu_offset=gpu_offset,
        ),
        num_gpus_per_engine=num_gpus_per_engine,
        has_new_engines=False,
        update_weights=True,
        needs_offload=needs_offload,
    )


async def _start_group(group: ServerGroup) -> None:
    await group.start_engines(PortAllocator.empty())


def _kill_group(group: ServerGroup) -> None:
    for e in flatten_cells(group.cells):
        if e.is_allocated:
            ray.kill(e.actor_handle)


# ----------------------------- check_weights -----------------------------


@pytest.mark.asyncio
class TestCheckWeightsAggregation:
    async def test_aggregates_across_groups_via_real_asyncio_gather(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """Drives RolloutServer.check_weights through real ``asyncio.gather``
        over real HTTP requests. Verifies every engine in every group was
        actually invoked (read from each mock server's request log)."""
        pg_a = placement_group_factory(2)
        pg_b = placement_group_factory(3)
        a = _build_group(pg_tuple=pg_a, num_engines=2)
        b = _build_group(pg_tuple=pg_b, num_engines=3, rank_offset=2)
        await _start_group(a)
        await _start_group(b)
        a.mark_alive([0, 1])
        b.mark_alive([0, 1, 2])

        srv = RolloutServer(server_groups=[a, b])
        try:
            results = await srv.check_weights(action="report")

            # Outer gather: 2 groups → 2 inner lists; inner: 1 entry per engine
            assert len(results) == 2
            assert len(results[0]) == 2 and len(results[1]) == 3

            all_engines = [e for g in (a, b) for e in g.engines]
            for engine in all_engines:
                payloads = ray.get(engine.actor_handle.get_http_payloads_of.remote("/weights_checker"))
                assert payloads == [{"action": "report", "allow_quant_error": False, "selector": "all"}]
        finally:
            _kill_group(a)
            _kill_group(b)


# ----------------------------- offload / onload -----------------------------


@pytest.mark.asyncio
class TestOffloadOnloadAggregation:
    async def test_offload_and_onload_reach_every_engine_of_every_group(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """Both fan out across groups and return one flat result per engine."""
        pg_a = placement_group_factory(2)
        pg_b = placement_group_factory(3)
        a = _build_group(pg_tuple=pg_a, num_engines=2, needs_offload=True)
        b = _build_group(pg_tuple=pg_b, num_engines=3, rank_offset=2, needs_offload=True)
        await _start_group(a)
        await _start_group(b)
        a.mark_alive([0, 1])
        b.mark_alive([0, 1, 2])

        srv = RolloutServer(server_groups=[a, b])
        try:
            offload_results = await srv.offload(tags=["weights"])
            onload_results = await srv.onload(["weights"])

            assert len(offload_results) == 5
            assert len(onload_results) == 5

            for engine in flatten_cells(a.cells) + flatten_cells(b.cells):
                paths = ray.get(engine.actor_handle.get_http_paths.remote())
                assert [path for path in paths if path.endswith("_memory_occupation")] == [
                    "/release_memory_occupation",
                    "/resume_memory_occupation",
                ]
                assert ray.get(engine.actor_handle.get_http_payloads_of.remote("/release_memory_occupation")) == [
                    {"tags": ["weights"]}
                ]
                assert ray.get(engine.actor_handle.get_http_payloads_of.remote("/resume_memory_occupation")) == [
                    {"tags": ["weights"]}
                ]

            for engine in flatten_cells(a.cells) + flatten_cells(b.cells):
                method_names = {name for name, _args, _kwargs in ray.get(engine.actor_handle.get_calls.remote())}
                assert not {"release_memory_occupation", "resume_memory_occupation"} & method_names
        finally:
            _kill_group(a)
            _kill_group(b)

    async def test_a_group_that_does_not_need_offload_is_skipped(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """Only the groups colocated with megatron give their memory back."""
        pg_a = placement_group_factory(2)
        pg_b = placement_group_factory(2)
        offloading = _build_group(pg_tuple=pg_a, num_engines=2, needs_offload=True)
        resident = _build_group(pg_tuple=pg_b, num_engines=2, rank_offset=2, needs_offload=False)
        await _start_group(offloading)
        await _start_group(resident)
        offloading.mark_alive([0, 1])
        resident.mark_alive([0, 1])

        srv = RolloutServer(server_groups=[offloading, resident])
        try:
            assert len(await srv.offload(tags=None)) == 2

            for engine in flatten_cells(resident.cells):
                assert "/release_memory_occupation" not in ray.get(engine.actor_handle.get_http_paths.remote())
        finally:
            _kill_group(offloading)
            _kill_group(resident)

    async def test_a_dead_engine_is_not_addressed(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """Offload must not block forever on an engine the group already gave up on."""
        pg = placement_group_factory(2)
        group = _build_group(pg_tuple=pg, num_engines=2, needs_offload=True)
        await _start_group(group)
        group.mark_alive([0, 1])
        flatten_cells(group.cells)[1].mark_stopped()

        srv = RolloutServer(server_groups=[group])
        try:
            assert len(await srv.offload(tags=None)) == 1
        finally:
            _kill_group(group)
