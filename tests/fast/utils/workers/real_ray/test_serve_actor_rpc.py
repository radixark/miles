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

from tests.fast.utils.workers.conftest import worker_manager_args
from tests.fast.utils.workers.real_ray.conftest import (
    kill_named_worker_manager,
    kill_quietly,
    wait_until_named_manager_is_gone,
)

from miles.utils.workers.naming import compute_cell_id
from miles.utils.workers.ray_worker_manager import RayWorkerManager
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


# every actor the manager launches is a fresh process that imports miles, so the checks that
# only read from the pool share one, and only the class that stops the cell rebuilds it
@pytest.fixture(autouse=True, scope="class")
def clean_named_worker_manager(ray_local_mode) -> Iterator[None]:
    _free_the_well_known_name()
    yield
    _free_the_well_known_name()


def _free_the_well_known_name() -> None:
    kill_named_worker_manager()
    wait_until_named_manager_is_gone()


def _launch_rpc_pool() -> _LaunchedPool:
    _free_the_well_known_name()
    manager = RayWorkerManager.launch(
        worker_manager_args(env_report_interval_seconds=0.0),
        [compute_spec(rpc_port=0)],
        {},
        comm_backend=WorkerCommBackend.RPC,
    )
    provider = RayWorkerProvider(worker_manager_handle=manager, pool_ids=[POOL_ID])
    (infos,) = provider.get_worker_infos(cell_ids=[CELL_ID])
    handles = provider.get_handles_of_worker_infos(infos)
    return _LaunchedPool(manager=manager, infos=infos, handles=[handles[info.name] for info in infos])


@pytest.fixture(scope="class")
def shared_rpc_pool(ray_local_mode) -> Iterator[_LaunchedPool]:
    pool = _launch_rpc_pool()
    yield pool
    kill_quietly(pool.manager)


@pytest.fixture
def rpc_pool(ray_local_mode) -> Iterator[_LaunchedPool]:
    pool = _launch_rpc_pool()
    yield pool
    kill_quietly(pool.manager)


@pytest.fixture
async def shared_rpc_handle(shared_rpc_pool: _LaunchedPool) -> AsyncIterator[BaseWorkerHandle]:
    handle = shared_rpc_pool.handles[0]
    await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)
    yield handle


@pytest.fixture
async def rpc_handle(rpc_pool: _LaunchedPool) -> AsyncIterator[BaseWorkerHandle]:
    handle = rpc_pool.handles[0]
    await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)
    yield handle


class TestARayLaunchedWorkerServedOverRpc:
    def test_the_launcher_answers_with_an_rpc_handle(self, shared_rpc_pool: _LaunchedPool):
        """Under rpc comm the driver must not be handed an actor handle it would call over ray."""
        assert isinstance(shared_rpc_pool.handles[0], RpcWorkerHandle)

    def test_the_worker_serves_on_the_port_the_launcher_allocated(self, shared_rpc_pool: _LaunchedPool):
        """A dynamically allocated port is the only thing that lets two workers share one node."""
        assert shared_rpc_pool.infos[0].self_addrs["rpc"].port > 0

    async def test_the_worker_is_reachable(self, shared_rpc_handle: BaseWorkerHandle):
        """This is the end to end claim of the mode: ray started the worker, http drives it."""
        assert await shared_rpc_handle.add(a=2, b=5) == 7

    async def test_the_worker_runs_inside_a_ray_actor(self, shared_rpc_handle: BaseWorkerHandle):
        """RDT and the rest of the ray ecosystem need the worker in the actor, not in a child process of it."""
        assert await shared_rpc_handle.report_ray_actor_id() is not None

    @pytest.mark.parametrize("check", CHECKS, ids=CHECK_IDS)
    async def test_the_handle_contract_holds(self, shared_rpc_handle: BaseWorkerHandle, check: HandleCheck):
        """The same contract as the serve-subprocess column, now over a worker that ray launched."""
        await check(shared_rpc_handle)


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
