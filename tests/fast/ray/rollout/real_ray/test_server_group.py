from __future__ import annotations

import pytest
import ray
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.addr_allocator import PortCursors
from miles.ray.rollout.server_engine import ServerEngine
from miles.ray.rollout.server_group import ServerGroup


def _build_group(
    *,
    pg_tuple: tuple,
    num_engines: int = 2,
    debug_train_only: bool = False,
    worker_type: str = "regular",
    update_weights: bool = True,
    needs_offload: bool = False,
    model_path: str | None = None,
) -> ServerGroup:
    args = make_args(num_gpus_per_node=8, debug_train_only=debug_train_only)
    engines = [ServerEngine() for _ in range(num_engines)] if worker_type != "placeholder" else []
    return ServerGroup(
        args=args,
        pg=pg_tuple,
        all_engines=engines,
        num_gpus_per_engine=1,
        has_new_engines=False,
        worker_type=worker_type,
        needs_offload=needs_offload,
        update_weights=update_weights,
        model_path=model_path,
    )


class TestStartEnginesShortCircuits:
    """Branches that bail before hitting the PG / actor creation path."""

    def test_debug_train_only_returns_immediately(self, placement_group_factory):
        # PG made but unused — start_engines should bail before scheduling.
        pg = placement_group_factory(2)
        group = _build_group(pg_tuple=pg, num_engines=2, debug_train_only=True)
        handles, indices = group.start_engines(PortCursors.empty())
        assert handles == [] and indices == []
        assert group.has_new_engines is False
        for e in group.all_engines:
            assert not e.is_allocated

    def test_placeholder_worker_short_circuits(self, placement_group_factory):
        # PG is unused in this short-circuit path; min size 1 keeps Ray happy.
        pg = placement_group_factory(1)
        group = _build_group(pg_tuple=pg, num_engines=0, worker_type="placeholder")
        handles, indices = group.start_engines(PortCursors.empty())
        assert handles == [] and indices == []
        assert group.has_new_engines is False


class TestStartEnginesRealActors:
    """Drives the actor-creation loop end-to-end. Verifies the actors are
    real Ray actors (via ``get_calls()`` round-trip) and that ``init`` was
    invoked with the addr/port kwargs from the allocator."""

    def test_creates_real_actors_and_init_runs(self, patched_sglang_engine, placement_group_factory):
        pg = placement_group_factory(2)
        group = _build_group(pg_tuple=pg, num_engines=2)

        handles, indices = group.start_engines(PortCursors.empty())
        assert sorted(indices) == [0, 1]
        assert group.has_new_engines is True
        # Wait for init.remote() to actually complete on each actor.
        ray.get(handles)

        for i, e in enumerate(group.all_engines):
            assert e.is_allocated
            calls = ray.get(e.actor_handle.get_calls.remote())
            method_names = [name for name, _, _ in calls]
            assert "init" in method_names
            init_kwargs = ray.get(e.actor_handle.get_init_kwargs.remote())
            assert init_kwargs["host"] == "127.0.0.1"
            assert init_kwargs["port"] == 30000 + i

        # Cleanup: kill the actors we created.
        for e in group.all_engines:
            ray.kill(e.actor_handle)

    def test_start_indices_filters_to_subset(self, patched_sglang_engine, placement_group_factory):
        pg = placement_group_factory(4)
        group = _build_group(pg_tuple=pg, num_engines=4)

        handles, indices = group.start_engines(PortCursors.empty(), start_indices=[1, 3])
        assert sorted(indices) == [1, 3]
        ray.get(handles)

        assert not group.all_engines[0].is_allocated
        assert group.all_engines[1].is_allocated
        assert not group.all_engines[2].is_allocated
        assert group.all_engines[3].is_allocated

        for i in (1, 3):
            ray.kill(group.all_engines[i].actor_handle)

    def test_already_allocated_slot_is_skipped(self, patched_sglang_engine, placement_group_factory):
        """A second start_engines() call must NOT replace an already-allocated
        actor — the existing handle is preserved verbatim."""
        pg = placement_group_factory(2)
        group = _build_group(pg_tuple=pg, num_engines=2)

        # First call: allocates both slots.
        handles, _ = group.start_engines(PortCursors.empty())
        ray.get(handles)
        first_handles = [e.actor_handle for e in group.all_engines]

        # Second call with no start_indices: should skip both.
        handles2, indices2 = group.start_engines(PortCursors.empty())
        assert handles2 == [] and indices2 == []
        for first, e in zip(first_handles, group.all_engines):
            assert e.actor_handle is first  # still the same actor

        for h in first_handles:
            ray.kill(h)


class TestStopEnginesRealKill:
    """``ray.kill`` is the real thing here — we verify the actor is actually
    dead by issuing a follow-up ``.remote()`` and expecting RayActorError."""

    def test_stop_marks_engines_stopped_and_actor_truly_dies(self, patched_sglang_engine, placement_group_factory):
        pg = placement_group_factory(2)
        group = _build_group(pg_tuple=pg, num_engines=2)
        handles, _ = group.start_engines(PortCursors.empty())
        ray.get(handles)

        actors = [e.actor_handle for e in group.all_engines]
        group.stop_engines(engine_indices=[0, 1])

        for e in group.all_engines:
            assert not e.is_allocated, "engine should be stopped"

        # Real-Ray claim: a follow-up call on a killed actor must surface as
        # RayActorError, not silently return.
        for actor in actors:
            with pytest.raises((ray.exceptions.RayActorError, ray.exceptions.RayTaskError)):
                ray.get(actor.health_generate.remote(timeout=1.0), timeout=10.0)

    def test_stop_handles_shutdown_failure_gracefully(self, patched_sglang_engine, placement_group_factory):
        """If ``shutdown`` raises on the actor, ``stop_engines`` must still
        mark the engine stopped (and ray.kill is still called).

        We use ``set_fault`` to make shutdown raise on its next invocation."""
        pg = placement_group_factory(2)
        group = _build_group(pg_tuple=pg, num_engines=2)
        handles, _ = group.start_engines(PortCursors.empty())
        ray.get(handles)

        # Plant a one-shot shutdown failure on engine 1.
        ray.get(
            group.all_engines[1].actor_handle.set_fault.remote(
                "shutdown",
                RuntimeError("boom"),
            )
        )

        group.stop_engines(engine_indices=[0, 1])
        for e in group.all_engines:
            assert not e.is_allocated, "all engines must be stopped despite shutdown raise"


class TestStartEnginesRealAllocator:
    """Drive ``start_engines`` with the real
    ``allocate_rollout_engine_addr_and_ports_normal`` (no stub) so that the
    actor → driver port round-trip via
    ``_get_current_node_ip_and_free_port.remote`` actually runs.

    The deterministic-port stub used by other tests bypasses this whole code
    path; without a real-allocator test, a regression in either side of that
    interface (mock_engine return shape vs. allocator's per-node cursor
    bookkeeping) would silently slip past the suite."""

    def test_real_allocator_assigns_distinct_ports_via_remote_calls(
        self,
        patched_sglang_engine_real_allocator,
        placement_group_factory,
    ):
        pg = placement_group_factory(2)
        group = _build_group(pg_tuple=pg, num_engines=2)

        handles, indices = group.start_engines(PortCursors.empty())
        assert sorted(indices) == [0, 1]
        ray.get(handles)

        # init kwargs == the addr_and_ports map produced by the real allocator
        kwargs0, kwargs1 = ray.get(
            [
                group.all_engines[0].actor_handle.get_init_kwargs.remote(),
                group.all_engines[1].actor_handle.get_init_kwargs.remote(),
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
        # _get_current_node_ip_and_free_port on the node-leader engine. The
        # allocator collapses per-node port lookups onto the first engine
        # for each node; with both engines on node 0 here, only engine 0
        # sees these calls — but engine 1's ports still come from those
        # results, so this assertion catches a regression where the allocator
        # silently fell back to a stub or swallowed the .remote() calls.
        leader_calls = ray.get(group.all_engines[0].actor_handle.get_calls.remote())
        leader_method_names = [name for name, _, _ in leader_calls]
        assert (
            "_get_current_node_ip_and_free_port" in leader_method_names
        ), f"allocator never called the port-finder; saw {leader_method_names}"

        for e in group.all_engines:
            ray.kill(e.actor_handle)

    def test_real_allocator_advances_cursor_across_sequential_groups(
        self,
        patched_sglang_engine_real_allocator,
        placement_group_factory,
    ):
        """Two sequentially-started groups on independent PGs both invoke the
        real allocator. ``start_engines`` mutates the passed-in PortCursors
        in place (via ``assign``); reusing it for B must shift B's ports past
        A's — that's the cursor's job."""
        pg_a = placement_group_factory(2)
        pg_b = placement_group_factory(2)
        a = _build_group(pg_tuple=pg_a, num_engines=2)
        b = _build_group(pg_tuple=pg_b, num_engines=2)

        cursors = PortCursors.empty()
        handles_a, _ = a.start_engines(cursors)
        ray.get(handles_a)
        # `cursors` now carries the next-free-port state from group A.

        handles_b, _ = b.start_engines(cursors)
        ray.get(handles_b)

        kwargs_a = ray.get([e.actor_handle.get_init_kwargs.remote() for e in a.all_engines])
        kwargs_b = ray.get([e.actor_handle.get_init_kwargs.remote() for e in b.all_engines])
        ports_a = {p for kw in kwargs_a for p in (kw["port"], kw["nccl_port"])}
        ports_b = {p for kw in kwargs_b for p in (kw["port"], kw["nccl_port"])}

        assert ports_a.isdisjoint(ports_b), f"sequential groups overlapped on ports: a={ports_a} b={ports_b}"

        for g in (a, b):
            for e in g.all_engines:
                ray.kill(e.actor_handle)
