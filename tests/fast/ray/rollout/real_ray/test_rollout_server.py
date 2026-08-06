from __future__ import annotations

import pytest
import ray
from tests.fast.ray.rollout.conftest import make_args
from tests.fast.ray.rollout.real_ray.conftest import build_cells, kill_cells, start_cells

from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell


def _make_server(cells: list[ServerCell]) -> RolloutServer:
    return RolloutServer(
        server_cells={f"cell-{i}": cell for i, cell in enumerate(cells)}, args=make_args(num_gpus_per_node=8)
    )


# ----------------------------- check_weights -----------------------------


@pytest.mark.asyncio
class TestCheckWeightsAggregation:
    async def test_aggregates_across_cells_via_real_asyncio_gather(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """Drives RolloutServer.check_weights through real ``asyncio.gather``
        over real HTTP requests. Verifies every engine of every cell was
        actually invoked (read from each mock server's request log)."""
        pg_a = placement_group_factory(2)
        pg_b = placement_group_factory(3)
        a = build_cells(pg_tuple=pg_a, num_cells=2)
        b = build_cells(pg_tuple=pg_b, num_cells=3, rank_offset=2)
        await start_cells(a, mark_alive=True)
        await start_cells(b, mark_alive=True)

        srv = _make_server(a + b)
        try:
            results = await srv.check_weights(action="report")

            # One flat entry per cell's primary engine.
            assert len(results) == 5

            for cell in a + b:
                payloads = ray.get(cell.primary_actor_handle.get_http_payloads_of.remote("/weights_checker"))
                assert payloads == [{"action": "report", "allow_quant_error": False, "selector": "all"}]
        finally:
            kill_cells(a)
            kill_cells(b)


# ----------------------------- offload / onload -----------------------------


@pytest.mark.asyncio
class TestOffloadOnloadAggregation:
    async def test_offload_and_onload_reach_every_engine_of_every_cell(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """Both fan out across cells and return one flat result per engine."""
        pg_a = placement_group_factory(2)
        pg_b = placement_group_factory(3)
        a = build_cells(pg_tuple=pg_a, num_cells=2, needs_offload=True)
        b = build_cells(pg_tuple=pg_b, num_cells=3, rank_offset=2, needs_offload=True)
        await start_cells(a, mark_alive=True)
        await start_cells(b, mark_alive=True)

        srv = _make_server(a + b)
        try:
            offload_results = await srv.offload(tags=["weights"])
            onload_results = await srv.onload(["weights"])

            assert len(offload_results) == 5
            assert len(onload_results) == 5

            for actor_handle in [handle for cell in a + b for handle in cell.actor_handles]:
                paths = ray.get(actor_handle.get_http_paths.remote())
                assert [path for path in paths if path.endswith("_memory_occupation")] == [
                    "/release_memory_occupation",
                    "/resume_memory_occupation",
                ]
                assert ray.get(actor_handle.get_http_payloads_of.remote("/release_memory_occupation")) == [
                    {"tags": ["weights"]}
                ]
                assert ray.get(actor_handle.get_http_payloads_of.remote("/resume_memory_occupation")) == [
                    {"tags": ["weights"]}
                ]

            for cell in a + b:
                method_names = {name for name, _args, _kwargs in ray.get(cell.primary_actor_handle.get_calls.remote())}
                assert not {"release_memory_occupation", "resume_memory_occupation"} & method_names
        finally:
            kill_cells(a)
            kill_cells(b)

    async def test_a_cell_that_does_not_need_offload_is_skipped(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """Only the cells colocated with megatron give their memory back."""
        pg_a = placement_group_factory(2)
        pg_b = placement_group_factory(2)
        offloading = build_cells(pg_tuple=pg_a, num_cells=2, needs_offload=True)
        resident = build_cells(pg_tuple=pg_b, num_cells=2, rank_offset=2, needs_offload=False)
        await start_cells(offloading, mark_alive=True)
        await start_cells(resident, mark_alive=True)

        srv = _make_server(offloading + resident)
        try:
            assert len(await srv.offload(tags=None)) == 2

            for cell in resident:
                assert "/release_memory_occupation" not in ray.get(cell.primary_actor_handle.get_http_paths.remote())
        finally:
            kill_cells(offloading)
            kill_cells(resident)

    async def test_a_dead_engine_is_not_addressed(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """Offload must not block forever on an engine the server already gave up on."""
        pg = placement_group_factory(2)
        cells = build_cells(pg_tuple=pg, num_cells=2, needs_offload=True)
        await start_cells(cells, mark_alive=True)
        cells[1].stop()

        srv = _make_server(cells)
        try:
            assert len(await srv.offload(tags=None)) == 1
        finally:
            kill_cells(cells)
