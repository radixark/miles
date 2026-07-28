from __future__ import annotations

import asyncio

import pytest
import ray
from tests.fast.ray.rollout.conftest import flatten_cells, make_args
from tests.fast.ray.rollout.real_ray.conftest import build_cells, start_cells

from miles.backends.sglang_utils.sglang_engine import build_server_url
from miles.ray.rollout.addr_allocator import PortAllocator
from miles.ray.rollout.rollout_server import RolloutServer


class TestStartEnginesShortCircuits:
    """Branches that bail before hitting the PG / actor creation path."""

    async def test_debug_train_only_returns_immediately(self, placement_group_factory):
        """In debug_train_only the server schedules no actors at all."""
        pg = placement_group_factory(2)
        cells = build_cells(pg_tuple=pg, num_cells=2, debug_train_only=True)
        srv = RolloutServer(server_cells=cells, args=cells[0].args)
        await srv.start_all_cells(PortAllocator())
        assert srv.has_new_engines is False
        for e in flatten_cells(cells):
            assert not e.is_allocated

    async def test_a_server_without_cells_starts_as_a_noop(self):
        """Placeholder groups produce no cells, so starting the server does nothing."""
        srv = RolloutServer(server_cells=[], args=make_args(num_gpus_per_node=8))
        await srv.start_all_cells(PortAllocator())
        assert srv.has_new_engines is False


class TestStartEnginesRealActors:
    """Drives the actor-creation loop end-to-end. Verifies the actors are
    real Ray actors (via ``get_calls()`` round-trip) and that ``init`` was
    invoked with the addr/port kwargs from the allocator."""

    async def test_creates_real_actors_and_init_runs(self, patched_sglang_engine, placement_group_factory):
        pg = placement_group_factory(2)
        cells = build_cells(pg_tuple=pg, num_cells=2)

        await start_cells(cells)

        for e in flatten_cells(cells):
            assert e.is_allocated
            calls = ray.get(e.actor_handle.get_calls.remote())
            method_names = [name for name, _, _ in calls]
            assert "init" in method_names
            init_kwargs = ray.get(e.actor_handle.get_init_kwargs.remote())
            assert init_kwargs["host"] == "127.0.0.1"
            assert e.addr_info.server_url == build_server_url(host=init_kwargs["host"], port=init_kwargs["port"])

        # Cleanup: kill the actors we created.
        for e in flatten_cells(cells):
            ray.kill(e.actor_handle)

    async def test_starting_a_subset_of_cells_leaves_the_rest_unallocated(
        self, patched_sglang_engine, placement_group_factory
    ):
        pg = placement_group_factory(4)
        cells = build_cells(pg_tuple=pg, num_cells=4)

        allocator = PortAllocator()
        await asyncio.gather(*[cells[i].start_engines(allocator) for i in (1, 3)])

        assert not cells[0].primary_engine.is_allocated
        assert cells[1].primary_engine.is_allocated
        assert not cells[2].primary_engine.is_allocated
        assert cells[3].primary_engine.is_allocated

        for i in (1, 3):
            ray.kill(cells[i].primary_engine.actor_handle)

    async def test_already_allocated_slot_is_skipped(self, patched_sglang_engine, placement_group_factory):
        """A second start_engines() call must NOT replace an already-allocated
        actor — the existing handle is preserved verbatim."""
        pg = placement_group_factory(2)
        cells = build_cells(pg_tuple=pg, num_cells=2)

        # First call: allocates both cells.
        await start_cells(cells)
        first_handles = [e.actor_handle for e in flatten_cells(cells)]

        # Second call: should skip both.
        await start_cells([cell for cell in cells if not cell.is_allocated])
        for first, e in zip(first_handles, flatten_cells(cells), strict=True):
            assert e.actor_handle is first  # still the same actor

        for h in first_handles:
            ray.kill(h)

    async def test_a_cell_that_lost_one_node_restarts_whole(self, patched_sglang_engine, placement_group_factory):
        """The surviving node belongs to a process group that no longer exists,
        so restarting the dead node alone would leave the engine broken."""
        pg = placement_group_factory(16)
        (cell,) = build_cells(pg_tuple=pg, num_cells=1, num_gpus_per_engine=16)
        await start_cells([cell])
        assert len(cell.engines) == 2
        original_handles = [e.actor_handle for e in cell.engines]

        cell.engines[1].mark_stopped()

        try:
            assert await cell.start_engines(PortAllocator()) is True
            assert all(e.is_allocated for e in cell.engines)
            # Both node-ranks are fresh actors, not just the one that died.
            for original, engine in zip(original_handles, cell.engines, strict=True):
                assert engine.actor_handle is not original
        finally:
            for e in cell.engines:
                if e.is_allocated:
                    ray.kill(e.actor_handle)


# FIXME(@fzyzcjy): TestStopCellsRealKill is a timing-sensitive Ray actor
# termination race that flakes in CI (stage-a-cpu). Real fix tracked in
# https://github.com/radixark/miles/pull/1282 — re-enable once that lands.
@pytest.mark.skip(reason="FIXME(@fzyzcjy): flaky Ray actor termination race; real fix in #1282")
class TestStopCellsRealKill:
    """``ray.kill`` is the real thing here — we verify the actor is actually
    dead by issuing a follow-up ``.remote()`` and expecting RayActorError."""

    async def test_stop_marks_engines_stopped_and_actor_truly_dies(
        self, patched_sglang_engine, placement_group_factory
    ):
        pg = placement_group_factory(2)
        cells = build_cells(pg_tuple=pg, num_cells=2)
        await start_cells(cells)
        srv = RolloutServer(server_cells=cells, args=cells[0].args)

        actors = [e.actor_handle for e in flatten_cells(cells)]
        await srv.stop_cells([0, 1])

        for e in flatten_cells(cells):
            assert not e.is_allocated, "engine should be stopped"

        # Real-Ray claim: a follow-up call on a killed actor must surface as
        # RayActorError, not silently return.
        for actor in actors:
            with pytest.raises((ray.exceptions.RayActorError, ray.exceptions.RayTaskError)):
                ray.get(actor.get_calls.remote(), timeout=10.0)

    async def test_stop_handles_shutdown_failure_gracefully(self, patched_sglang_engine, placement_group_factory):
        """If ``shutdown`` raises on the actor, ``stop_cells`` must still
        mark the engine stopped (and ray.kill is still called).

        We use ``set_fault`` to make shutdown raise on its next invocation."""
        pg = placement_group_factory(2)
        cells = build_cells(pg_tuple=pg, num_cells=2)
        await start_cells(cells)
        srv = RolloutServer(server_cells=cells, args=cells[0].args)

        # Plant a one-shot shutdown failure on engine 1.
        ray.get(
            cells[1].primary_engine.actor_handle.set_fault.remote(
                "shutdown",
                RuntimeError("boom"),
            )
        )

        await srv.stop_cells([0, 1])
        for e in flatten_cells(cells):
            assert not e.is_allocated, "all engines must be stopped despite shutdown raise"


class TestStartEnginesRealAllocator:
    """Drive ``start_engines``'s inline port allocation (no stub) so that the
    actor → driver port round-trip via
    ``_get_current_node_ip_and_free_port.remote`` actually runs."""

    async def test_real_allocator_assigns_distinct_ports_via_remote_calls(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        pg = placement_group_factory(2)
        cells = build_cells(pg_tuple=pg, num_cells=2)

        await start_cells(cells)

        # init kwargs == the addr_and_ports map produced by the real allocator
        kwargs0, kwargs1 = ray.get(
            [
                cells[0].primary_engine.actor_handle.get_init_kwargs.remote(),
                cells[1].primary_engine.actor_handle.get_init_kwargs.remote(),
            ]
        )

        # Real-allocator claim 1: each engine got a fully-formed addr/port set
        for k in kwargs0, kwargs1:
            for key in ("host", "port", "nccl_port", "dist_init_addr"):
                assert key in k, f"missing {key} in init kwargs from real allocator"
            assert k["host"] == "127.0.0.1"

        # Real-allocator claim 2: ports are distinct between engines (the
        # node_port_cursor must advance across engines on the same node).
        ports_engine0 = {kwargs0["port"], kwargs0["nccl_port"]}
        ports_engine1 = {kwargs1["port"], kwargs1["nccl_port"]}
        assert ports_engine0.isdisjoint(
            ports_engine1
        ), f"port collision across engines: {ports_engine0} vs {ports_engine1}"

        # Real-allocator claim 3: the allocator actually called
        # _get_current_node_ip_and_free_port on each cell's engine; this
        # assertion catches a regression where the allocator silently fell
        # back to a stub or swallowed the .remote() calls.
        calls = ray.get(cells[0].primary_engine.actor_handle.get_calls.remote())
        method_names = [name for name, _, _ in calls]
        assert (
            "_get_current_node_ip_and_free_port" in method_names
        ), f"allocator never called the port-finder; saw {method_names}"

        for e in flatten_cells(cells):
            ray.kill(e.actor_handle)

    async def test_real_allocator_advances_cursor_across_sequential_cells(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """Two sequentially-started batches of cells on independent PGs both
        invoke the real allocator. ``start_engines`` mutates the passed-in
        PortAllocator in place; reusing it for B must shift B's ports past
        A's — that's the cursor's job."""
        pg_a = placement_group_factory(2)
        pg_b = placement_group_factory(2)
        a = build_cells(pg_tuple=pg_a, num_cells=2)
        b = build_cells(pg_tuple=pg_b, num_cells=2)

        allocator = PortAllocator()
        await start_cells(a, allocator)
        # `allocator` now carries the next-free-port state from batch A.

        await start_cells(b, allocator)

        kwargs_a = ray.get([e.actor_handle.get_init_kwargs.remote() for e in flatten_cells(a)])
        kwargs_b = ray.get([e.actor_handle.get_init_kwargs.remote() for e in flatten_cells(b)])
        ports_a = {p for kw in kwargs_a for p in (kw["port"], kw["nccl_port"])}
        ports_b = {p for kw in kwargs_b for p in (kw["port"], kw["nccl_port"])}

        assert ports_a.isdisjoint(ports_b), f"sequential cells overlapped on ports: a={ports_a} b={ports_b}"

        for cells in (a, b):
            for e in flatten_cells(cells):
                ray.kill(e.actor_handle)


class TestRejectedConfigurations:
    @pytest.mark.parametrize("overrides", [{"port": 40000}, {"host": "10.9.9.9"}, {"host": "10.9.9.9", "port": 40000}])
    async def test_host_or_port_override_is_rejected(self, patched_sglang_engine, placement_group_factory, overrides):
        """An override of host or port would make the rollout process address the wrong endpoint."""
        (cell,) = build_cells(pg_tuple=placement_group_factory(1), num_cells=1)
        cell.sglang_overrides = overrides

        with pytest.raises(AssertionError, match="must not override host/port"):
            await cell.start_engines(PortAllocator())
