from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

import pytest
import ray
from tests.fast.utils.workers.conformance import (
    CHECK_IDS,
    CHECKS,
    POOL_ID,
    READY_TIMEOUT_SECONDS,
    HandleCheck,
    compute_spec,
)

from miles.utils.workers.naming import compute_cell_id
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.types import WorkerCommBackend
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.ray import RayWorkerProvider

CELL_ID = compute_cell_id(pool_id=POOL_ID, cell_index=0)

CONFIRM_DEAD_TIMEOUT_SECONDS = 60.0


@dataclass
class _LaunchedPool:
    manager: ray.actor.ActorHandle
    infos: list[WorkerInfo]
    handles: list[BaseWorkerHandle]


@pytest.fixture
def rpc_pool(manager_factory) -> Iterator[_LaunchedPool]:
    manager = manager_factory([compute_spec(rpc_port=0)], {}, WorkerCommBackend.RPC)
    provider = RayWorkerProvider(worker_manager_handle=manager, pool_ids=[POOL_ID])
    (infos,) = provider.get_worker_infos(cell_ids=[CELL_ID])
    handles = provider.get_handles_of_worker_infos(infos)
    yield _LaunchedPool(manager=manager, infos=infos, handles=[handles[info.name] for info in infos])


@pytest.fixture
async def rpc_handle(rpc_pool: _LaunchedPool) -> AsyncIterator[BaseWorkerHandle]:
    handle = rpc_pool.handles[0]
    await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)
    yield handle


class TestARayLaunchedWorkerServedOverRpc:
    def test_the_launcher_answers_with_an_rpc_handle(self, rpc_pool: _LaunchedPool):
        """Under rpc comm the driver must not be handed an actor handle it would call over ray."""
        assert isinstance(rpc_pool.handles[0], RpcWorkerHandle)

    def test_the_worker_serves_on_the_port_the_launcher_allocated(self, rpc_pool: _LaunchedPool):
        """A dynamically allocated port is the only thing that lets two workers share one node."""
        assert rpc_pool.infos[0].self_addrs["rpc"].port > 0

    async def test_the_worker_is_reachable(self, rpc_handle: BaseWorkerHandle):
        """This is the end to end claim of the mode: ray started the worker, http drives it."""
        assert await rpc_handle.add(a=2, b=5) == 7

    async def test_the_worker_runs_inside_a_ray_actor(self, rpc_handle: BaseWorkerHandle):
        """RDT and the rest of the ray ecosystem need the worker in the actor, not in a child process of it."""
        assert await rpc_handle.report_ray_actor_id() is not None

    @pytest.mark.parametrize("check", CHECKS, ids=CHECK_IDS)
    async def test_the_handle_contract_holds(self, rpc_handle: BaseWorkerHandle, check: HandleCheck):
        """The same contract as the serve-subprocess column, now over a worker that ray launched."""
        await check(rpc_handle)


class TestWhenTheLauncherStopsTheCell:
    async def test_the_worker_is_confirmed_dead(self, rpc_pool: _LaunchedPool, rpc_handle: BaseWorkerHandle):
        """Fault tolerance kills a cell and then waits for this confirmation before healing it."""
        await rpc_pool.manager.stop_cells.remote([CELL_ID])

        await rpc_handle.wait_dead(timeout=CONFIRM_DEAD_TIMEOUT_SECONDS)

    async def test_the_probe_that_confirms_it_reads_a_refused_connection(
        self, rpc_pool: _LaunchedPool, rpc_handle: BaseWorkerHandle
    ):
        """Killing the actor takes the server down with it, which is what makes the probe conclusive."""
        await rpc_pool.manager.stop_cells.remote([CELL_ID])
        await rpc_handle.wait_dead(timeout=CONFIRM_DEAD_TIMEOUT_SECONDS)

        assert await rpc_handle.probe_is_dead() is True
