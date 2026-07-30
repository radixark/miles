from __future__ import annotations

import asyncio

import pytest
import ray
from tests.fast.ray.rollout.conftest import make_args
from tests.fast.ray.rollout.real_ray.conftest import build_cells, start_cells

from miles.backends.sglang_utils.sglang_engine import build_server_url
from miles.ray.rollout.addr_allocator import PortAllocator
from miles.ray.rollout.rollout_server import RolloutServer


def _all_actor_handles(cells) -> list:
    return [handle for cell in cells for handle in cell.actor_handles]


class TestStartEnginesShortCircuits:
    """Branches that bail before hitting the PG / actor creation path."""

    async def test_debug_train_only_returns_immediately(self, placement_group_factory):
        """In debug_train_only the server schedules no actors at all."""
        pg = placement_group_factory(2)
        cells = build_cells(pg_tuple=pg, num_cells=2, debug_train_only=True)
        srv = RolloutServer(server_cells={f"cell-{i}": cell for i, cell in enumerate(cells)}, args=cells[0].args)
        await srv.start_all_cells(PortAllocator())
        assert srv.has_new_engines is False
        for cell in cells:
            assert not cell.is_allocated

    async def test_a_server_without_cells_starts_as_a_noop(self):
        """Placeholder groups produce no cells, so starting the server does nothing."""
        srv = RolloutServer(server_cells={}, args=make_args(num_gpus_per_node=8))
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

        for cell in cells:
            assert cell.is_allocated
            calls = ray.get(cell.primary_actor_handle.get_calls.remote())
            method_names = [name for name, _, _ in calls]
            assert "init" in method_names
            init_kwargs = ray.get(cell.primary_actor_handle.get_init_kwargs.remote())
            assert init_kwargs["host"] == "127.0.0.1"
            assert cell.addr_info.server_url == build_server_url(host=init_kwargs["host"], port=init_kwargs["port"])

        # Cleanup: kill the actors we created.
        for handle in _all_actor_handles(cells):
            ray.kill(handle)

    async def test_starting_a_subset_of_cells_leaves_the_rest_unallocated(
        self, patched_sglang_engine, placement_group_factory
    ):
        pg = placement_group_factory(4)
        cells = build_cells(pg_tuple=pg, num_cells=4)

        allocator = PortAllocator()
        await asyncio.gather(*[cells[i].start_engines(allocator) for i in (1, 3)])

        assert not cells[0].is_allocated
        assert cells[1].is_allocated
        assert not cells[2].is_allocated
        assert cells[3].is_allocated

        for i in (1, 3):
            ray.kill(cells[i].primary_actor_handle)

    async def test_an_already_running_cell_is_skipped(self, patched_sglang_engine, placement_group_factory):
        """A second start_engines() call must NOT replace a running cell's
        actors — the existing handles are preserved verbatim."""
        pg = placement_group_factory(2)
        cells = build_cells(pg_tuple=pg, num_cells=2)

        # First call: allocates both cells.
        await start_cells(cells)
        first_handles = _all_actor_handles(cells)

        # Second call: should skip both.
        await start_cells([cell for cell in cells if not cell.is_allocated])
        for first, handle in zip(first_handles, _all_actor_handles(cells), strict=True):
            assert handle is first  # still the same actor

        for handle in first_handles:
            ray.kill(handle)

    async def test_restarting_a_multi_node_cell_replaces_every_node_rank(
        self, patched_sglang_engine, placement_group_factory
    ):
        """A cell is one distributed engine: a restart must bring back every
        node-rank, since a survivor would belong to a process group that is gone."""
        pg = placement_group_factory(16)
        (cell,) = build_cells(pg_tuple=pg, num_cells=1, num_gpus_per_engine=16)
        await start_cells([cell])
        assert len(cell.actor_handles) == 2
        original_handles = list(cell.actor_handles)

        cell.stop()
        assert not cell.is_allocated

        try:
            assert await cell.start_engines(PortAllocator()) is True
            assert len(cell.actor_handles) == 2
            for original, handle in zip(original_handles, cell.actor_handles, strict=True):
                assert handle is not original
        finally:
            if cell.is_allocated:
                for handle in cell.actor_handles:
                    ray.kill(handle)


# FIXME(@fzyzcjy): TestStopCellsRealKill is a timing-sensitive Ray actor
# termination race that flakes in CI (stage-a-cpu). Real fix tracked in
# https://github.com/radixark/miles/pull/1282 — re-enable once that lands.
@pytest.mark.skip(reason="FIXME(@fzyzcjy): flaky Ray actor termination race; real fix in #1282")
class TestStopCellsRealKill:
    """``ray.kill`` is the real thing here — we verify the actor is actually
    dead by issuing a follow-up ``.remote()`` and expecting RayActorError."""

    async def test_stop_marks_cells_stopped_and_actors_truly_die(self, patched_sglang_engine, placement_group_factory):
        pg = placement_group_factory(2)
        cells = build_cells(pg_tuple=pg, num_cells=2)
        await start_cells(cells)
        srv = RolloutServer(server_cells={f"cell-{i}": cell for i, cell in enumerate(cells)}, args=cells[0].args)

        actors = _all_actor_handles(cells)
        await srv.stop_cells(["cell-0", "cell-1"])

        for cell in cells:
            assert not cell.is_allocated, "cell should be stopped"

        # Real-Ray claim: a follow-up call on a killed actor must surface as
        # RayActorError, not silently return.
        for actor in actors:
            with pytest.raises((ray.exceptions.RayActorError, ray.exceptions.RayTaskError)):
                ray.get(actor.get_calls.remote(), timeout=10.0)

    async def test_stop_handles_shutdown_failure_gracefully(self, patched_sglang_engine, placement_group_factory):
        """If ``shutdown`` raises on the actor, ``stop_cells`` must still
        mark the cell stopped (and ray.kill is still called).

        We use ``set_fault`` to make shutdown raise on its next invocation."""
        pg = placement_group_factory(2)
        cells = build_cells(pg_tuple=pg, num_cells=2)
        await start_cells(cells)
        srv = RolloutServer(server_cells={f"cell-{i}": cell for i, cell in enumerate(cells)}, args=cells[0].args)

        # Plant a one-shot shutdown failure on cell 1.
        ray.get(cells[1].primary_actor_handle.set_fault.remote("shutdown", RuntimeError("boom")))

        await srv.stop_cells(["cell-0", "cell-1"])
        for cell in cells:
            assert not cell.is_allocated, "all cells must be stopped despite shutdown raise"


class TestStartEnginesRealAllocator:
    """Drive ``start_engines``'s inline port allocation (no stub) so that the
    actor → driver port round-trip via
    ``_get_free_port_block.remote`` actually runs."""

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
                cells[0].primary_actor_handle.get_init_kwargs.remote(),
                cells[1].primary_actor_handle.get_init_kwargs.remote(),
            ]
        )

        # Real-allocator claim 1: each engine got a fully-formed addr/port set
        for k in kwargs0, kwargs1:
            for key in ("host", "port", "nccl_port", "dist_init_addr"):
                assert key in k, f"missing {key} in init kwargs from real allocator"
            assert k["host"] == "127.0.0.1"

        # Real-allocator claim 2: ports are distinct between engines (the
        # node cursor must advance across engines on the same node).
        ports_engine0 = {kwargs0["port"], kwargs0["nccl_port"]}
        ports_engine1 = {kwargs1["port"], kwargs1["nccl_port"]}
        assert ports_engine0.isdisjoint(
            ports_engine1
        ), f"port collision across engines: {ports_engine0} vs {ports_engine1}"

        # Real-allocator claim 3: the allocator actually called
        # _get_free_port_block on each cell's actor; this
        # assertion catches a regression where the allocator silently fell
        # back to a stub or swallowed the .remote() calls.
        calls = ray.get(cells[0].primary_actor_handle.get_calls.remote())
        method_names = [name for name, _, _ in calls]
        assert "_get_free_port_block" in method_names, f"allocator never called the port-finder; saw {method_names}"

        for handle in _all_actor_handles(cells):
            ray.kill(handle)

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

        kwargs_a = ray.get([handle.get_init_kwargs.remote() for handle in _all_actor_handles(a)])
        kwargs_b = ray.get([handle.get_init_kwargs.remote() for handle in _all_actor_handles(b)])
        ports_a = {p for kw in kwargs_a for p in (kw["port"], kw["nccl_port"])}
        ports_b = {p for kw in kwargs_b for p in (kw["port"], kw["nccl_port"])}

        assert ports_a.isdisjoint(ports_b), f"sequential cells overlapped on ports: a={ports_a} b={ports_b}"

        for handle in _all_actor_handles(a) + _all_actor_handles(b):
            ray.kill(handle)


class TestRejectedConfigurations:
    @pytest.mark.parametrize("overrides", [{"port": 40000}, {"host": "10.9.9.9"}, {"host": "10.9.9.9", "port": 40000}])
    async def test_host_or_port_override_is_rejected(self, patched_sglang_engine, placement_group_factory, overrides):
        """An override of host or port would make the rollout process address the wrong endpoint."""
        (cell,) = build_cells(pg_tuple=placement_group_factory(1), num_cells=1)
        cell.sglang_overrides = overrides

        with pytest.raises(AssertionError, match="must not override host/port"):
            await cell.start_engines(PortAllocator())
