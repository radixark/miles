import os
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import ray
from tests.fast.ray.train.dummy_actor import DummyTrainActor

import miles.ray.train.group as group_module
from miles.ray.specs.train import MASTER_PORT_NAME
from miles.ray.train.cell import RayTrainCell
from miles.utils.ft_utils.health_checker import NoopHealthChecker
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.retry_utils import retry
from miles.utils.workers.ray_worker_manager import WorkerInfo
from miles.utils.workers.worker_spec import HostAndPort


class FakeWorkerProvider:
    def __init__(self, actor_count_per_cell: int = 2):
        self.actor_count_per_cell = actor_count_per_cell
        self._cell_indices_failing_init: set[int] = set()

    def fail_init_for_cell(self, cell_index: int) -> None:
        self._cell_indices_failing_init.add(cell_index)

    def get_worker_infos(self, *, pool: str, cell_index: int) -> list[WorkerInfo]:
        handles = [DummyTrainActor.remote() for _ in range(self.actor_count_per_cell)]
        if cell_index in self._cell_indices_failing_init:
            ray.get([handle.set_fail_methods.remote(["init"]) for handle in handles])
        return [
            WorkerInfo(
                name=f"{pool}-{cell_index}-{worker_index}",
                generation=1,
                self_addrs={MASTER_PORT_NAME: HostAndPort(host="10.0.0.1", port=20000)},
                gpu_ids=[worker_index],
                actor_handle=handle,
            )
            for worker_index, handle in enumerate(handles)
        ]


fake_worker_provider: FakeWorkerProvider | None = None


@pytest.fixture(autouse=True)
def _patch_worker_provider():
    global fake_worker_provider
    fake_worker_provider = FakeWorkerProvider()
    provider_factory = SimpleNamespace(create=lambda: fake_worker_provider)
    with patch("miles.ray.train.cell.RayWorkerProvider", provider_factory):
        yield


@pytest.fixture(scope="module", autouse=True)
def ray_env():
    if ray.is_initialized():
        # Reuse the cluster some outer fixture created (e.g. the session-scoped
        # one in tests/conftest.py) and never tear down what we did not create.
        yield
        return

    init_kwargs: dict = {"ignore_reinit_error": True}
    if "RAY_ADDRESS" not in os.environ:
        # address="local" forces a fresh cluster: with no address, ray.init
        # auto-connects to any leaked local cluster (via /tmp/ray), and
        # connecting with num_cpus/num_gpus set is a hard ValueError.
        init_kwargs["address"] = "local"
        init_kwargs["num_cpus"] = 4
        init_kwargs["num_gpus"] = 0
    ray.init(**init_kwargs)
    yield
    ray.shutdown()


@pytest.fixture(autouse=True)
def instant_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _no_sleep(_seconds: float) -> None:
        return None

    async def _retry_without_sleeping(fn: Callable[[int], Awaitable[Any]], **kwargs: Any) -> Any:
        return await retry(fn, **{**kwargs, "sleep_fn": _no_sleep})

    monkeypatch.setattr(group_module, "retry", _retry_without_sleeping)


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
    fake_worker_provider.actor_count_per_cell = actor_count
    return RayTrainCell(
        args=MagicMock(),
        role="actor",
        with_ref=False,
        cell_index=cell_index,
        pool="trainer-actor",
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
