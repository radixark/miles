"""Real ``ray.kill`` is required so follow-up ``.remote()`` calls surface
``RayActorError``; a MagicMock handle can't simulate that."""

from __future__ import annotations

import asyncio

import pytest
import ray
from tests.fast.ray.rollout.conftest import make_args
from tests.fast.ray.rollout.real_ray.conftest import build_cells, kill_cells, start_cells

from miles.ray.rollout.addr_allocator import PortAllocator
from miles.ray.rollout.rollout_server import RolloutServer


# ----------------------------- single-engine kill + recover -----------------------------


@pytest.mark.asyncio
class TestKillAndRecover:
    async def test_recover_creates_new_actor_after_real_kill(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """Kill cell 0's engine for real, recover, verify a fresh actor replaces
        it and the surviving cell is untouched."""
        pg = placement_group_factory(2)
        cells = build_cells(pg_tuple=pg, num_cells=2)
        await start_cells(cells, mark_alive=True)

        original_handles = [cell.primary_actor_handle for cell in cells]
        # Real fault: kill engine 0 + mark its slot stopped (production code's
        # health monitor would do this; here we simulate it directly).
        ray.kill(original_handles[0])
        cells[0].stop()

        try:
            await cells[0].recover(PortAllocator())
            # New actor for cell 0
            assert cells[0].is_allocated
            assert cells[0].primary_actor_handle is not original_handles[0]
            calls = ray.get(cells[0].primary_actor_handle.get_calls.remote())
            assert "init" in [c[0] for c in calls]

            # Cell 1 untouched, still the same actor
            assert cells[1].primary_actor_handle is original_handles[1]
        finally:
            kill_cells(cells)

    async def test_recover_default_filter_picks_all_dead_cells(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """When ``cell_indices=None``, the server recovers every cell with a
        dead engine. We kill 0 and 2, leave 1 alive, expect only 0 and 2 to
        be re-created."""
        pg = placement_group_factory(3)
        cells = build_cells(pg_tuple=pg, num_cells=3)
        await start_cells(cells, mark_alive=True)
        srv = RolloutServer(server_cells=cells, args=make_args(num_gpus_per_node=8))

        old = [cell.primary_actor_handle for cell in cells]
        for i in (0, 2):
            ray.kill(old[i])
            cells[i].stop()

        try:
            await srv.recover()
            for i in (0, 2):
                assert cells[i].is_allocated
                assert cells[i].primary_actor_handle is not old[i]
            assert cells[1].primary_actor_handle is old[1]
        finally:
            kill_cells(cells)

    async def test_recover_publishes_the_new_url_to_the_router(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """A recovered engine gets a fresh port, so the router must be told the new url."""
        from unittest.mock import patch

        events: list[dict] = []

        class _Recorder:
            async def add_worker(self, **kwargs):
                events.append(kwargs)

            async def remove_worker(self, **kwargs):
                events.append(kwargs)

        pg = placement_group_factory(1)
        cells = build_cells(pg_tuple=pg, num_cells=1)
        srv = RolloutServer(
            server_cells=cells, args=make_args(num_gpus_per_node=8), router_ip="10.0.0.9", router_port=9000
        )
        await start_cells(cells, mark_alive=True)
        ray.kill(cells[0].primary_actor_handle)
        cells[0].stop()

        try:
            with patch.object(RolloutServer, "_router_api_client", property(lambda self: _Recorder())):
                await srv.recover(cell_indices=[0])

            assert [event["worker_url"] for event in events] == [cells[0].addr_info.server_url]
            assert cells[0].is_alive
        finally:
            kill_cells(cells)

    async def test_recover_with_offload_calls_release_then_resume(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """``needs_offload=True`` + ``update_weights=True`` means recover()
        must release_memory_occupation, then resume with WEIGHTS tag.
        Verify by reading the recovered engine's mock HTTP server log."""
        pg = placement_group_factory(2)
        cells = build_cells(pg_tuple=pg, num_cells=2, needs_offload=True, update_weights=True)
        await start_cells(cells, mark_alive=True)
        old = [cell.primary_actor_handle for cell in cells]

        ray.kill(old[0])
        cells[0].stop()

        try:
            await cells[0].recover(PortAllocator())
            recovered_actor = cells[0].primary_actor_handle
            calls = ray.get(recovered_actor.get_calls.remote())
            assert "init" in [c[0] for c in calls]

            paths = ray.get(recovered_actor.get_http_paths.remote())
            assert "/release_memory_occupation" in paths
            assert "/resume_memory_occupation" in paths

            # Ordering claim: release must precede resume — otherwise GPU
            # memory would be re-occupied before being released, defeating
            # the offload. Use the first occurrence of each.
            release_idx = paths.index("/release_memory_occupation")
            resume_idx = paths.index("/resume_memory_occupation")
            assert release_idx < resume_idx, f"release must precede resume; saw order {paths}"
            # The client drains the working queue before releasing.
            assert paths.index("/flush_cache") < release_idx

            from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

            # Recovery releases everything, not just the weights: an engine that kept its kv cache
            # would leave the trainer short of GPU memory when it takes the device back.
            assert ray.get(recovered_actor.get_http_payloads_of.remote("/release_memory_occupation")) == [
                {"tags": None}
            ]
            assert ray.get(recovered_actor.get_http_payloads_of.remote("/resume_memory_occupation")) == [
                {"tags": [GPU_MEMORY_TYPE_WEIGHTS]}
            ]
        finally:
            kill_cells(cells)


# ----------------------------- concurrent recover -----------------------------


@pytest.mark.asyncio
class TestConcurrentRecover:
    async def test_two_cell_batches_recover_in_parallel_completes_without_deadlock(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """Two cell batches recovering simultaneously through real
        ``asyncio.gather`` must both complete — no deadlock, no exception
        leaking out of the gather chain.

        The batches share one PortAllocator, as they do in production: each
        batch's ports are only bound once its engine inits, so concurrent
        recovers with independent allocators could probe the same free port
        twice. The real-ray claim being verified is end-to-end gather
        completion across two batches."""
        pg_a = placement_group_factory(2)
        pg_b = placement_group_factory(2)
        a = build_cells(pg_tuple=pg_a, num_cells=2)
        b = build_cells(pg_tuple=pg_b, num_cells=2)
        await start_cells(a, mark_alive=True)
        await start_cells(b, mark_alive=True)

        # Kill one engine in each batch
        for cells in (a, b):
            old = cells[0].primary_actor_handle
            ray.kill(old)
            cells[0].stop()

        try:
            # Real concurrent recover via asyncio.gather
            shared_allocator = PortAllocator()
            await asyncio.gather(
                a[0].recover(shared_allocator),
                b[0].recover(shared_allocator),
            )
            assert a[0].is_allocated
            assert b[0].is_allocated
        finally:
            kill_cells(a)
            kill_cells(b)


# ----------------------------- simulate_crash at cell level -----------------------------


@pytest.mark.asyncio
class TestSimulateCrashKeepsActorReachable:
    """``MockSGLangEngine.simulate_crash`` self-calls ``shutdown()`` (mirror
    of real SGLangEngine). The actor stays alive at the Ray level; this is
    important because the rollout health monitor uses follow-up ``.remote()``
    calls to determine liveness."""

    async def test_simulate_crash_then_health_check_still_returns(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        pg = placement_group_factory(1)
        cells = build_cells(pg_tuple=pg, num_cells=1)
        await start_cells(cells, mark_alive=True)
        actor = cells[0].primary_actor_handle

        try:
            ray.get(actor.simulate_crash.remote())
            # Actor handle still reachable at Ray level — follow-up returns.
            ray.get(actor.get_calls.remote(), timeout=10.0)
        finally:
            kill_cells(cells)


@pytest.mark.asyncio
class TestRecoverMultiNodeEngine:
    async def test_recover_releases_and_resumes_only_on_node0(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """Recovering a 2-node engine must not send release/resume to node 1."""
        pg = placement_group_factory(16)
        (cell,) = build_cells(pg_tuple=pg, num_cells=1, num_gpus_per_engine=16, needs_offload=True)
        assert cell.num_nodes == 2

        try:
            await cell.recover(PortAllocator())

            node0_actor, node1_actor = cell.actor_handles
            node0_paths = ray.get(node0_actor.get_http_paths.remote())
            node1_paths = ray.get(node1_actor.get_http_paths.remote())

            assert "/release_memory_occupation" in node0_paths
            assert "/resume_memory_occupation" in node0_paths
            assert node1_paths == []
        finally:
            kill_cells([cell])
