"""The backend trainer actors report which data-parallel group admission reads from."""

from __future__ import annotations

import importlib

import pytest

from miles.backends.training_utils import parallel as parallel_mod


@pytest.fixture
def fsdp_actor_module():
    return importlib.import_module("miles.backends.fsdp_utils.actor")


@pytest.fixture(autouse=True)
def restore_parallel_state():
    saved = parallel_mod._parallel_state
    try:
        yield
    finally:
        parallel_mod._parallel_state = saved


def set_effective_dp(rank: int, size: int) -> None:
    from miles.utils.ft_utils.process_group_utils import GroupInfo

    trivial = GroupInfo(rank=0, size=1, group=None)
    parallel_mod.set_parallel_state(
        parallel_mod.ParallelState(
            intra_dp=GroupInfo(rank=rank, size=size, group=None),
            intra_dp_cp=trivial,
            cp=trivial,
            tp=trivial,
            pp=trivial,
            ep=trivial,
            etp=trivial,
            indep_dp=trivial,
        )
    )


def test_megatron_actor_reports_effective_data_parallel(megatron_actor_module) -> None:
    actor = object.__new__(megatron_actor_module.MegatronTrainRayActor)
    set_effective_dp(rank=2, size=4)

    assert actor._admission_data_parallel() == (2, 4)


def test_fsdp_actor_reports_effective_data_parallel(fsdp_actor_module) -> None:
    actor = object.__new__(fsdp_actor_module.FSDPTrainRayActor)
    set_effective_dp(rank=1, size=2)

    assert actor._admission_data_parallel() == (1, 2)


def test_megatron_actor_reports_unknown_data_parallel_before_initialization(megatron_actor_module) -> None:
    actor = object.__new__(megatron_actor_module.MegatronTrainRayActor)
    parallel_mod._parallel_state = None

    assert actor._admission_data_parallel() is None


def test_fsdp_actor_reports_unknown_data_parallel_before_initialization(fsdp_actor_module) -> None:
    actor = object.__new__(fsdp_actor_module.FSDPTrainRayActor)
    parallel_mod._parallel_state = None

    assert actor._admission_data_parallel() is None
