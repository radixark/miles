import sys
from concurrent.futures import Future
from types import ModuleType

mooncake = ModuleType("mooncake")
mooncake_engine = ModuleType("mooncake.engine")
mooncake_engine.TransferEngine = object
sys.modules.setdefault("mooncake", mooncake)
sys.modules.setdefault("mooncake.engine", mooncake_engine)

from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p_transfer_utils import (
    P2PTransferManager,
    RemoteTransferPlan,
)


def _plan(*, source_count: int, source_rank: int) -> RemoteTransferPlan:
    plan = RemoteTransferPlan.__new__(RemoteTransferPlan)
    plan._pp_rank = 0
    plan._gathered_dp_size = source_count
    plan._gathered_dp_rank = source_rank
    plan._rollout_num_gpu_per_engine = 8
    plan._rollout_engine_count = 4
    plan._rollout_num_gpus = 32
    plan._engine_gpu_counts = [8, 8, 8, 8]
    return plan


def test_p2p_plan_uses_actual_pd_update_engine_topology() -> None:
    plan = _plan(source_count=8, source_rank=0)

    plan.set_target_topology([8, 8])
    all_tasks = [task for source_rank in range(8) for task in _plan_for_source(plan, source_rank)]

    assert len(all_tasks) == 16
    assert {task.engine_ind for task in all_tasks} == {0, 1}
    assert {task.engine_rank for task in all_tasks} == set(range(8))


def test_p2p_plan_supports_heterogeneous_engine_sizes() -> None:
    plan = _plan(source_count=4, source_rank=0)

    plan.set_target_topology([2, 4])
    all_tasks = [task for source_rank in range(4) for task in _plan_for_source(plan, source_rank)]

    assert sorted((task.engine_ind, task.engine_rank) for task in all_tasks) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
    ]


def test_p2p_plan_rejects_empty_target_topology() -> None:
    plan = _plan(source_count=8, source_rank=0)

    try:
        plan.set_target_topology([])
    except ValueError as error:
        assert "positive engine GPU counts" in str(error)
    else:
        raise AssertionError("empty target topology was accepted")


def test_p2p_transfer_failure_is_not_silenced() -> None:
    manager = P2PTransferManager()
    failed = Future()
    failed.set_exception(RuntimeError("transfer failed"))
    manager.transfer_futures.append(failed)

    try:
        manager.wait_transfers()
    except RuntimeError as error:
        assert str(error) == "transfer failed"
    else:
        raise AssertionError("transfer failure was silenced")

    assert manager.transfer_futures == []


def _plan_for_source(plan: RemoteTransferPlan, source_rank: int):
    plan._gathered_dp_rank = source_rank
    return plan.plan_p2p()
