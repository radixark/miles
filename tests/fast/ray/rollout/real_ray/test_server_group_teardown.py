from __future__ import annotations

import threading
import time

import ray
from tests.fast.ray.rollout.conftest import chunk_engines_into_cells, make_args

import miles.ray.rollout.server_cell as server_cell_module
from miles.ray.rollout.addr_allocator import PortCursors
from miles.ray.rollout.server_cell import flatten_cells
from miles.ray.rollout.server_engine import ServerEngine
from miles.ray.rollout.server_group import ServerGroup


@ray.remote(num_cpus=0)
class _HangingEngine:
    def shutdown(self):
        time.sleep(3600)


def _build_group(*, pg_tuple: tuple, num_engines: int = 1) -> ServerGroup:
    return ServerGroup(
        args=make_args(num_gpus_per_node=8),
        pg=pg_tuple,
        cells=chunk_engines_into_cells(
            [ServerEngine() for _ in range(num_engines)], num_gpus_per_engine=1, num_gpus_per_node=8
        ),
        num_gpus_per_engine=1,
        has_new_engines=False,
        worker_type="regular",
    )


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
    def test_a_failing_shutdown_still_kills_the_actor(self, patched_sglang_engine, placement_group_factory):
        """A graceful shutdown that raises must not leave the actor and its server process behind."""
        group = _build_group(pg_tuple=placement_group_factory(1))
        handles, _ = group.start_engines(PortCursors.empty())
        ray.get(handles)
        actor_handle = flatten_cells(group.cells)[0].actor_handle
        ray.get(actor_handle.set_fault.remote("shutdown", RuntimeError("shutdown blew up")))

        group.stop_engines(engine_indices=[0])

        assert _is_dead(actor_handle)
        assert not flatten_cells(group.cells)[0].is_allocated

    def test_a_hanging_shutdown_does_not_block_teardown(self, monkeypatch, ray_local_mode):
        """A wedged engine must not stall teardown forever, since teardown is how a wedged engine is reclaimed."""
        monkeypatch.setattr(server_cell_module, "SHUTDOWN_TIMEOUT", 0.5)
        group = _build_group(pg_tuple=(None, [], []))
        actor_handle = _HangingEngine.remote()
        flatten_cells(group.cells)[0].mark_allocated_uninitialized(actor_handle)

        finished = threading.Event()
        thread = threading.Thread(target=lambda: (group.stop_engines(engine_indices=[0]), finished.set()), daemon=True)
        thread.start()
        thread.join(timeout=30)

        assert finished.is_set(), "stop_engines waited on a shutdown that never returns"
        assert _is_dead(actor_handle)
        assert not flatten_cells(group.cells)[0].is_allocated
