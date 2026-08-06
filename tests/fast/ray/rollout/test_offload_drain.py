"""Offload must quiesce engines before any memory release.

``RolloutServer.offload`` contract: every offloaded engine acknowledges
``pause_generation``, then ``flush_cache``, before any engine is told to
``release_memory_occupation``; groups that don't overlap megatron GPUs are
untouched. The fake handles record ``("call", ...)`` when a remote method
fires and ``("ack", ...)`` when its handle is awaited, so the assertions read
the real cross-engine ordering."""

from __future__ import annotations

import pytest
from tests.fast.ray.rollout.conftest import fake_actor_handle, make_args

from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_engine import ServerEngine
from miles.ray.rollout.server_group import ServerGroup

_ENGINE_METHODS = ("pause_generation", "flush_cache", "release_memory_occupation", "continue_generation")


def _remote_recorder(name: str, method: str, log: list):
    def remote(**kwargs):
        log.append(("call", name, method, kwargs))

        async def ack():
            log.append(("ack", name, method, kwargs))
            return f"{name}:{method}"

        return ack()

    return remote


def _logged_handle(name: str, log: list):
    handle = fake_actor_handle()
    for method in _ENGINE_METHODS:
        getattr(handle, method).remote.side_effect = _remote_recorder(name, method, log)
    return handle


def _group(names: list[str], log: list, *, needs_offload: bool = True, num_dead_engines: int = 0) -> ServerGroup:
    engines = []
    for name in names:
        engine = ServerEngine()
        engine.mark_allocated_uninitialized(_logged_handle(name, log))
        engines.append(engine)
    engines.extend(ServerEngine() for _ in range(num_dead_engines))
    return ServerGroup(
        args=make_args(),
        pg=None,
        all_engines=engines,
        num_gpus_per_engine=1,
        has_new_engines=False,
        needs_offload=needs_offload,
    )


def _indices(log: list, kind: str, method: str) -> list[int]:
    return [i for i, (k, _, m, _kw) in enumerate(log) if k == kind and m == method]


@pytest.mark.asyncio
class TestOffloadQuiescence:
    async def test_every_engine_acks_pause_and_flush_before_any_release(self):
        # A dead slot in group "a" must be skipped by every phase — touching
        # its actor_handle would raise inside ServerEngine.
        log: list = []
        srv = RolloutServer(
            server_groups=[
                _group(["a0", "a1"], log, num_dead_engines=1),
                _group(["b0"], log),
            ]
        )

        results = await srv.offload(tags=["kv_cache"])

        pause_calls = _indices(log, "call", "pause_generation")
        flush_calls = _indices(log, "call", "flush_cache")
        release_calls = _indices(log, "call", "release_memory_occupation")
        assert len(pause_calls) == len(flush_calls) == len(release_calls) == 3

        # Phase barriers: every pause acked before any flush fires; every
        # flush acked before any release fires.
        assert max(_indices(log, "ack", "pause_generation")) < min(flush_calls)
        assert max(_indices(log, "ack", "flush_cache")) < min(release_calls)

        # abort, not retract: retract would park requests that update_weights'
        # continue_generation resumes before the KV cache is restored, and is
        # unsupported in PD disaggregation.
        assert all(log[i][3] == {"mode": "abort"} for i in pause_calls)
        assert all(log[i][3] == {"tags": ["kv_cache"]} for i in release_calls)

        assert sorted(results) == [
            "a0:release_memory_occupation",
            "a1:release_memory_occupation",
            "b0:release_memory_occupation",
        ]

    async def test_non_offloaded_groups_are_untouched(self):
        log: list = []
        srv = RolloutServer(
            server_groups=[
                _group(["a0"], log),
                _group(["p0"], log, needs_offload=False),
            ]
        )

        await srv.offload(tags=["kv_cache"])
        await srv.continue_generation()

        assert [entry for entry in log if entry[1] == "p0"] == []
        a0_methods = [m for kind, name, m, _kw in log if kind == "call" and name == "a0"]
        assert a0_methods == ["pause_generation", "flush_cache", "release_memory_occupation", "continue_generation"]
