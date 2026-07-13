import os
from unittest.mock import MagicMock

import pytest
import ray
from tests.fast.ray.train.dummy_actor import DummyTrainActor

from miles.ray.train.cell import RayTrainCell
from miles.utils.ft_utils.health_checker import NoopHealthChecker
from miles.utils.ft_utils.indep_dp import IndepDPInfo


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
    rollout_manager: object | None = None,
) -> RayTrainCell:
    def factory():
        return [DummyTrainActor.remote() for _ in range(actor_count)]

    return RayTrainCell(
        args=MagicMock(),
        role="actor",
        with_ref=False,
        cell_index=cell_index,
        actor_factory=factory,
        rollout_manager=rollout_manager,
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
