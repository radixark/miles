from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pytest
import ray
from tests.fast.utils.workers.conformance import (
    POOL_ID,
    READY_TIMEOUT_SECONDS,
    SHARED_CHECK_IDS,
    SHARED_CHECKS,
    HandleCheck,
    compute_spec,
)

from miles.utils.workers.naming import compute_cell_id
from miles.utils.workers.ray_worker_handle import RayWorkerHandle
from miles.utils.workers.types import WorkerCommBackend
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_provider.ray import RayWorkerProvider

CELL_ID = compute_cell_id(pool_id=POOL_ID, cell_index=0)

CONFIRM_DEAD_TIMEOUT_SECONDS = 60.0


@pytest.fixture
def ray_comm_pool(manager_factory) -> Iterator[ray.actor.ActorHandle]:
    yield manager_factory([compute_spec(rpc_port=0)], {}, WorkerCommBackend.RAY)


@pytest.fixture
async def ray_comm_handle(ray_comm_pool: ray.actor.ActorHandle) -> AsyncIterator[BaseWorkerHandle]:
    provider = RayWorkerProvider(worker_manager_handle=ray_comm_pool, pool_ids=[POOL_ID])
    handle = provider.get_handle(f"{POOL_ID}-0-0")
    await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)
    yield handle


class TestARayLaunchedWorkerCalledOverRay:
    def test_the_launcher_answers_with_an_actor_handle(self, ray_comm_pool: ray.actor.ActorHandle):
        """Both wires stay supported until the default flips, so this column must keep running beside rpc."""
        provider = RayWorkerProvider(worker_manager_handle=ray_comm_pool, pool_ids=[POOL_ID])

        handle = provider.get_handle(f"{POOL_ID}-0-0")

        assert isinstance(handle, RayWorkerHandle)

    @pytest.mark.parametrize("check", SHARED_CHECKS, ids=SHARED_CHECK_IDS)
    async def test_the_handle_contract_holds(self, ray_comm_handle: BaseWorkerHandle, check: HandleCheck):
        """The contract a driver is written against must not depend on which wire carries the call."""
        await check(ray_comm_handle)


class TestWhenTheLauncherStopsTheCell:
    async def test_the_worker_is_confirmed_dead(
        self, ray_comm_pool: ray.actor.ActorHandle, ray_comm_handle: BaseWorkerHandle
    ):
        """Fault tolerance kills a cell and waits for this confirmation before healing it, on either wire."""
        await ray_comm_pool.stop_cells.remote([CELL_ID])

        await ray_comm_handle.wait_dead(timeout=CONFIRM_DEAD_TIMEOUT_SECONDS)

        assert await ray_comm_handle.probe_is_dead() is True
