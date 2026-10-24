import asyncio
from collections.abc import Callable
from pathlib import Path

import ray

from miles.utils.workers.worker_provider.ray import RayWorkerProvider
from miles.utils.workers.worker_spec import BaseWorkerSpec, SchedulingSpec, ServeWorkerSpec


async def test_heartbeat_responds_while_the_same_caller_is_waiting_for_training(
    manager_factory: Callable[[list[BaseWorkerSpec]], ray.actor.ActorHandle], tmp_path: Path
) -> None:
    """A blocked training call must not prevent a dedicated heartbeat from running."""
    spec = ServeWorkerSpec(
        name="trainer",
        port_infos=[],
        env_var=lambda _ctx: {},
        scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0),
        worker_class="tests.fast.utils.workers.real_ray.heartbeat_worker._HeartbeatWorker",
        ctor_kwargs=lambda _ctx: {},
        concurrency_groups={"heartbeat_status": 1, "default": 1},
    )
    manager = manager_factory([spec])
    handle = RayWorkerProvider(worker_manager_handle=manager, pool_ids=["trainer"]).get_handle("trainer-00000-00000")
    await handle.wait_ready(timeout=30)
    started_path = tmp_path / "training-started"
    training = asyncio.create_task(handle.train(started_path=str(started_path)))

    async def wait_started() -> None:
        while not started_path.exists():
            await asyncio.sleep(0.01)

    try:
        await asyncio.wait_for(wait_started(), timeout=10)
        assert not training.done()
        assert await asyncio.wait_for(handle.heartbeat(), timeout=3) == "alive"
        await asyncio.wait_for(training, timeout=5)
    finally:
        training.cancel()
        await asyncio.gather(training, return_exceptions=True)
