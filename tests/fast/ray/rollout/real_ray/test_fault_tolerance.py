"""Fault tolerance — real ``ray.kill`` injects the fault, real ``recover()``
restores the engine.

A MagicMock-only ``actor_handle`` never enters the Ray scheduler, so it
can't simulate "actor dies for real". Here we drive ``ServerGroup.recover()``
after ``ray.kill``ing a real ``MockSGLangEngine`` actor, so the recovery
path runs against genuinely-dead handles."""

from __future__ import annotations

import asyncio

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
    needs_offload: bool = False,
    update_weights: bool = True,
    model_path: str | None = None,
) -> ServerGroup:
    args = make_args(num_gpus_per_node=8)
    engines = [ServerEngine() for _ in range(num_engines)]
    return ServerGroup(
        args=args,
        pg=pg_tuple,
        all_engines=engines,
        num_gpus_per_engine=1,
        has_new_engines=False,
        needs_offload=needs_offload,
        update_weights=update_weights,
        model_path=model_path,
    )


def _start(group: ServerGroup) -> None:
    handles, indices = group.start_engines(PortCursors.empty())
    ray.get(handles)
    group.mark_alive(indices)


def _kill_all(group: ServerGroup) -> None:
    for e in group.all_engines:
        if e.is_allocated:
            try:
                ray.kill(e.actor_handle)
            except Exception:
                pass


# ----------------------------- single-engine kill + recover -----------------------------


@pytest.mark.asyncio
class TestKillAndRecover:
    async def test_recover_creates_new_actor_after_real_kill(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """Kill engine 0 for real, recover, verify a fresh actor replaces it
        and the surviving engine is untouched."""
        pg = placement_group_factory(2)
        group = _build_group(pg_tuple=pg, num_engines=2)
        _start(group)

        original_handles = [e.actor_handle for e in group.all_engines]
        # Real fault: kill engine 0 + mark its slot stopped (production code's
        # health monitor would do this; here we simulate it directly).
        ray.kill(original_handles[0])
        group.all_engines[0].mark_stopped()

        try:
            await group.recover(port_cursors=PortCursors.empty(), filter_indices=[0])
            # New actor for slot 0
            assert group.all_engines[0].is_allocated
            assert group.all_engines[0].actor_handle is not original_handles[0]
            calls = ray.get(group.all_engines[0].actor_handle.get_calls.remote())
            assert "init" in [c[0] for c in calls]

            # Slot 1 untouched, still the same actor
            assert group.all_engines[1].actor_handle is original_handles[1]
        finally:
            _kill_all(group)

    async def test_recover_default_filter_picks_all_dead_slots(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """When ``filter_indices=None``, recover picks every slot whose
        ``is_allocated`` is False. We kill 0 and 2, leave 1 alive, expect
        only 0 and 2 to be re-created."""
        pg = placement_group_factory(3)
        group = _build_group(pg_tuple=pg, num_engines=3)
        _start(group)

        old = [e.actor_handle for e in group.all_engines]
        for i in (0, 2):
            ray.kill(old[i])
            group.all_engines[i].mark_stopped()

        try:
            await group.recover(port_cursors=PortCursors.empty())
            for i in (0, 2):
                assert group.all_engines[i].is_allocated
                assert group.all_engines[i].actor_handle is not old[i]
            assert group.all_engines[1].actor_handle is old[1]
        finally:
            _kill_all(group)

    async def test_recover_with_offload_calls_release_then_resume(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """``needs_offload=True`` + ``update_weights=True`` means recover()
        must release_memory_occupation, then resume with WEIGHTS tag.
        Verify by reading the new actor's call log."""
        pg = placement_group_factory(2)
        group = _build_group(pg_tuple=pg, num_engines=2, needs_offload=True, update_weights=True)
        _start(group)
        old = [e.actor_handle for e in group.all_engines]

        ray.kill(old[0])
        group.all_engines[0].mark_stopped()

        try:
            await group.recover(port_cursors=PortCursors.empty(), filter_indices=[0])
            calls = ray.get(group.all_engines[0].actor_handle.get_calls.remote())
            method_names = [c[0] for c in calls]
            # init → release → resume(tags=[WEIGHTS])
            assert "init" in method_names
            assert "release_memory_occupation" in method_names
            assert "resume_memory_occupation" in method_names

            # Ordering claim: release must precede resume — otherwise GPU
            # memory would be re-occupied before being released, defeating
            # the offload. Use the first occurrence of each.
            release_idx = method_names.index("release_memory_occupation")
            resume_idx = method_names.index("resume_memory_occupation")
            assert release_idx < resume_idx, f"release must precede resume; saw order {method_names}"

            # Find the resume call and confirm WEIGHTS tag
            resume_calls = [c for c in calls if c[0] == "resume_memory_occupation"]
            assert len(resume_calls) == 1
            from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

            assert resume_calls[0][2] == {"tags": [GPU_MEMORY_TYPE_WEIGHTS]}
        finally:
            _kill_all(group)


# ----------------------------- concurrent recover -----------------------------


@pytest.mark.asyncio
class TestConcurrentRecover:
    async def test_two_groups_recover_in_parallel_completes_without_deadlock(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """Two ServerGroups recovering simultaneously through real
        ``asyncio.gather`` must both complete — no deadlock, no exception
        leaking out of the gather chain.

        We do not claim "no port collision" here because the deterministic
        port stub from the conftest gives each group its own range (groups
        don't see each other's ranks), so disjoint-port is trivially true.
        The real-ray claim being verified is end-to-end gather completion
        across two groups."""
        pg_a = placement_group_factory(2)
        pg_b = placement_group_factory(2)
        a = _build_group(pg_tuple=pg_a, num_engines=2)
        b = _build_group(pg_tuple=pg_b, num_engines=2)
        _start(a)
        _start(b)

        # Kill one engine in each group
        for g in (a, b):
            old = g.all_engines[0].actor_handle
            ray.kill(old)
            g.all_engines[0].mark_stopped()

        try:
            # Real concurrent recover via asyncio.gather
            await asyncio.gather(
                a.recover(port_cursors=PortCursors.empty(), filter_indices=[0]),
                b.recover(port_cursors=PortCursors.empty(), filter_indices=[0]),
            )
            assert a.all_engines[0].is_allocated
            assert b.all_engines[0].is_allocated
        finally:
            _kill_all(a)
            _kill_all(b)


# ----------------------------- simulate_crash at ServerGroup level -----------------------------


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
        group = _build_group(pg_tuple=pg, num_engines=1)
        _start(group)
        actor = group.all_engines[0].actor_handle

        try:
            ray.get(actor.simulate_crash.remote())
            # Actor handle still reachable at Ray level — follow-up returns.
            ray.get(actor.health_generate.remote(timeout=1.0), timeout=10.0)
        finally:
            _kill_all(group)
