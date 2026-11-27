from __future__ import annotations

import asyncio
import os

import pytest
import ray
from tests.fast.ray.rollout.conftest import make_rollout_config
from tests.fast.utils.workers.conftest import worker_manager_args
from tests.fast.utils.workers.real_ray.conftest import (
    kill_named_worker_manager,
    kill_quietly,
    wait_until_named_manager_is_gone,
)

from miles.ray.rollout.reward_pool import RewardPoolClient
from miles.ray.specs.reward import RewardPoolSpec, reward_pool_id, reward_worker_name
from miles.utils.args.configs.reward_pool import RewardPoolConfig
from miles.utils.workers.backend_capability.ray import RayBackendCapability
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.types import WorkerCommBackend
from miles.utils.workers.worker_provider.ray import RayWorkerProvider

_CONFIG = RewardPoolConfig(
    name="length",
    scorer_path=f"{__name__}.LengthScorer",
    num_workers=1,
    uses_gpu=False,
    batch_size=2,
    num_gpus_per_node=1,
)


class LengthScorer:
    def __init__(self, *, device) -> None:
        self.pid = os.getpid()

    def score_batch(self, outputs, prompts):
        return [float(len(prompt) * output) for output, prompt in zip(outputs, prompts, strict=True)]


@pytest.fixture
def worker_manager(ray_local_mode):
    kill_named_worker_manager()
    wait_until_named_manager_is_gone()
    handle = RayWorkerManager.launch(
        worker_manager_args(env_report_interval_seconds=0.0),
        RewardPoolSpec.create(make_rollout_config().model_copy(update={"reward_pools": [_CONFIG]})),
        {},
        comm_backend=WorkerCommBackend.RAY,
    )
    yield handle
    kill_quietly(handle)
    kill_named_worker_manager()
    wait_until_named_manager_is_gone()


def _actor(worker_manager) -> ray.actor.ActorHandle:
    provider = RayWorkerProvider(worker_manager_handle=worker_manager, pool_ids=[reward_pool_id(_CONFIG.name)])
    return provider.get_handle(reward_worker_name(_CONFIG.name, cell_index=0))._actor_handle


async def test_a_killed_reward_worker_is_restarted_by_the_client_and_rescores_the_same_batch(worker_manager):
    client = RewardPoolClient(
        config=_CONFIG, capability=RayBackendCapability(worker_manager_handle=worker_manager), ready_timeout=120.0
    )
    outputs, prompts = [1, 2, 3], ["a", "bb", "ccc"]
    before = await client.score(outputs, prompts)
    killed = _actor(worker_manager)

    ray.kill(killed, no_restart=True)
    after = await client.score(outputs, prompts)

    assert before == after == [1.0, 4.0, 9.0]
    assert _actor(worker_manager) != killed


async def test_a_reward_worker_whose_cell_was_torn_down_before_its_first_call_is_restarted(worker_manager):
    """The worker dies before the client ever looked it up, and the manager's liveness scan drops its cell."""
    ray.kill(_actor(worker_manager), no_restart=True)
    operations = RayBackendCapability(worker_manager_handle=worker_manager).cell_operations()
    for _ in range(60):
        infos = await operations.cell_infos(pool_ids=[reward_pool_id(_CONFIG.name)], category=None)
        if not any(info.alive for info in infos.values()):
            break
        await asyncio.sleep(1)
    assert not any(info.alive for info in infos.values())
    client = RewardPoolClient(
        config=_CONFIG, capability=RayBackendCapability(worker_manager_handle=worker_manager), ready_timeout=120.0
    )

    assert await client.score([1, 2], ["a", "bb"]) == [1.0, 4.0]
