from __future__ import annotations

import asyncio
import threading
import time

import ray
from tests.fast.ray.rollout.conftest import make_args

import miles.ray.rollout.server_cell as server_cell_module
from miles.ray.rollout.addr_allocator import PortAllocator
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell
from miles.ray.rollout.server_engine import ServerEngine


@ray.remote(num_cpus=0)
class _HangingEngine:
    def shutdown(self):
        time.sleep(3600)


def _build_server(*, pg_tuple: tuple) -> RolloutServer:
    args = make_args(num_gpus_per_node=8)
    cell = ServerCell(engines=[ServerEngine()], args=args, pg=pg_tuple, num_gpus_per_engine=1)
    return RolloutServer(server_cells=[cell], args=args)


def _is_dead(actor_handle, *, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ray.get(actor_handle.__ray_ready__.remote(), timeout=1.0)
        except ray.exceptions.RayActorError:
            return True
        except ray.exceptions.GetTimeoutError:
            pass
        time.sleep(0.1)
    return False


class TestTeardownIsTerminal:
    async def test_a_failing_shutdown_still_kills_the_actor(self, patched_sglang_engine, placement_group_factory):
        """A graceful shutdown that raises must not leave the actor and its server process behind."""
        srv = _build_server(pg_tuple=placement_group_factory(1))
        await srv.server_cells[0].start_engines(PortAllocator())
        actor_handle = srv.server_cells[0].primary_engine.actor_handle
        ray.get(actor_handle.set_fault.remote("shutdown", RuntimeError("shutdown blew up")))

        await srv.stop_cells([0])

        assert _is_dead(actor_handle)
        assert not srv.server_cells[0].primary_engine.is_allocated

    def test_a_hanging_shutdown_does_not_block_teardown(self, monkeypatch, ray_local_mode):
        """A wedged engine must not stall teardown forever, since teardown is how a wedged engine is reclaimed."""
        monkeypatch.setattr(server_cell_module, "SHUTDOWN_TIMEOUT", 0.5)
        srv = _build_server(pg_tuple=(None, [], []))
        actor_handle = _HangingEngine.remote()
        srv.server_cells[0].primary_engine.mark_allocated_uninitialized(actor_handle)

        finished = threading.Event()

        def _teardown():
            asyncio.run(srv.stop_cells([0]))
            finished.set()

        thread = threading.Thread(target=_teardown, daemon=True)
        thread.start()
        thread.join(timeout=30)

        assert finished.is_set(), "stop_cells waited on a shutdown that never returns"
        assert _is_dead(actor_handle)
        assert not srv.server_cells[0].primary_engine.is_allocated
