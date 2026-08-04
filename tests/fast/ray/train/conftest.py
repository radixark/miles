from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import ray
from tests.fast.ray.train.fake_worker_manager import FakeWorkerManager

import miles.ray.train.group as group_module
from miles.ray.train.cell import RayTrainCell
from miles.utils.ft_utils.health_checker import NoopHealthChecker
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.retry_utils import retry
from miles.utils.workers.worker_provider.ray import RayWorkerProvider

fake_worker_manager: FakeWorkerManager | None = None


@pytest.fixture(autouse=True)
def _patch_worker_backends():
    global fake_worker_manager
    fake_worker_manager = FakeWorkerManager()
    with (
        patch("miles.utils.workers.ray_worker_manager.RayWorkerManager.get_handle", lambda: fake_worker_manager),
        patch(
            "miles.utils.workers.worker_provider.ray.RayWorkerProvider.create",
            lambda *, pool_ids=None: RayWorkerProvider(worker_manager_handle=fake_worker_manager, pool_ids=pool_ids),
        ),
    ):
        yield
    fake_worker_manager.kill_all_actors()


@pytest.fixture(scope="module", autouse=True)
def ray_env(ray_local_mode):
    yield


@pytest.fixture(autouse=True)
def instant_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _no_sleep(_seconds: float) -> None:
        return None

    async def _retry_without_sleeping(fn: Callable[[int], Awaitable[Any]], **kwargs: Any) -> Any:
        return await retry(fn, **{**kwargs, "sleep_fn": _no_sleep})

    monkeypatch.setattr(group_module, "retry", _retry_without_sleeping)


def get_raw_actor_handles(cell: RayTrainCell) -> list[ray.actor.ActorHandle]:
    return [handle._actor_handle for handle in cell._get_worker_handles()]


def make_indep_dp_info(
    *,
    cell_index: int = 0,
    alive_cell_indices: list[int] | None = None,
    quorum_id: int = 1,
) -> IndepDPInfo:
    if alive_cell_indices is None:
        alive_cell_indices = [0]
    return IndepDPInfo(
        cell_index=cell_index,
        num_cells=3,
        alive_rank=alive_cell_indices.index(cell_index),
        alive_size=len(alive_cell_indices),
        quorum_id=quorum_id,
        alive_cell_indices=alive_cell_indices,
    )


def make_cell(
    cell_index: int = 0,
    *,
    actor_count: int = 2,
    rollout_executor: object | None = None,
) -> RayTrainCell:
    fake_worker_manager.actor_count_per_cell = actor_count
    return RayTrainCell(
        args=MagicMock(),
        role="actor",
        with_ref=False,
        cell_id=f"trainer-actor-{cell_index}",
        cell_index=cell_index,
        workers_hash="pseudo-hash-1",
        rollout_executor=rollout_executor,
        health_checker=NoopHealthChecker(),
    )


def make_alive_cell(cell_index: int, *, alive_cell_indices: list[int], quorum_id: int = 0) -> RayTrainCell:
    """Create a cell and transition it to Alive state."""
    cell = make_cell(cell_index)
    cell._mark_as_alive(
        indep_dp_info=make_indep_dp_info(
            cell_index=cell_index,
            alive_cell_indices=alive_cell_indices,
            quorum_id=quorum_id,
        )
    )
    return cell
