from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch


def _plan(
    p2p_transfer_utils: ModuleType,
    *,
    gathered_dp_rank: int,
    gathered_dp_size: int,
    pp_rank: int = 0,
):
    plan = object.__new__(p2p_transfer_utils.RemoteTransferPlan)
    plan._pp_rank = pp_rank
    plan._pp_size = 1
    plan._gathered_dp_rank = gathered_dp_rank
    plan._gathered_dp_size = gathered_dp_size
    plan._rollout_pp_size = 1
    return plan


def _targets_of(
    p2p_transfer_utils: ModuleType,
    *,
    gathered_dp_rank: int,
    gathered_dp_size: int,
    engine_gpu_counts: list[int],
    pp_rank: int = 0,
) -> list[tuple[int, int]]:
    tasks = _plan(
        p2p_transfer_utils,
        gathered_dp_rank=gathered_dp_rank,
        gathered_dp_size=gathered_dp_size,
        pp_rank=pp_rank,
    ).plan_p2p(engine_gpu_counts)
    return sorted((task.rollout_engine_ind, task.rollout_engine_rank) for task in tasks)


def _install_trainer_parallelism(
    p2p_transfer_utils: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    *,
    pp_rank: int,
    pp_size: int,
    gathered_dp_rank: int,
    gathered_dp_size: int,
) -> None:
    monkeypatch.setattr(
        p2p_transfer_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(pp=SimpleNamespace(rank=pp_rank, size=pp_size)),
    )
    monkeypatch.setattr(
        p2p_transfer_utils,
        "get_data_replica_rank_and_size",
        lambda *args, **kwargs: (gathered_dp_rank, gathered_dp_size),
    )


class TestRemoteTransferPlanParallelism:
    """The trainer-side parallelism the plan is built from."""

    def test_the_plan_reads_its_shard_and_replica_position_from_the_trainer_state(
        self, p2p_transfer_utils: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A rank plans for its own pp shard and its own position among the weight replicas."""
        _install_trainer_parallelism(
            p2p_transfer_utils, monkeypatch, pp_rank=1, pp_size=2, gathered_dp_rank=3, gathered_dp_size=4
        )

        plan = p2p_transfer_utils.RemoteTransferPlan(SimpleNamespace(sglang_pp_size=1))

        assert (plan._pp_rank, plan._pp_size) == (1, 2)
        assert (plan._gathered_dp_rank, plan._gathered_dp_size) == (3, 4)

    def test_rollout_pipeline_parallelism_is_refused(
        self, p2p_transfer_utils: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pipelined rollout engine would need a per-stage plan that does not exist yet."""
        _install_trainer_parallelism(
            p2p_transfer_utils, monkeypatch, pp_rank=0, pp_size=1, gathered_dp_rank=0, gathered_dp_size=1
        )

        with pytest.raises(NotImplementedError):
            p2p_transfer_utils.RemoteTransferPlan(SimpleNamespace(sglang_pp_size=2))


class TestPlanP2P:
    """Assignment of rollout engine ranks to the trainer ranks that write to them."""

    def test_the_documented_four_source_two_engine_example_is_planned_as_described(
        self, p2p_transfer_utils: ModuleType
    ) -> None:
        """Round-robin first, then the remainder follows the source already serving that engine rank."""
        by_source = {
            gathered_dp_rank: _targets_of(
                p2p_transfer_utils,
                gathered_dp_rank=gathered_dp_rank,
                gathered_dp_size=4,
                engine_gpu_counts=[3, 3],
            )
            for gathered_dp_rank in range(4)
        }

        assert by_source == {
            0: [(0, 0)],
            1: [(0, 1), (1, 1)],
            2: [(0, 2), (1, 2)],
            3: [(1, 0)],
        }

    def test_every_rollout_engine_rank_is_written_by_exactly_one_source(self, p2p_transfer_utils: ModuleType) -> None:
        """A rank written twice would race and a rank written by nobody would keep stale weights."""
        assigned = [
            target
            for gathered_dp_rank in range(3)
            for target in _targets_of(
                p2p_transfer_utils,
                gathered_dp_rank=gathered_dp_rank,
                gathered_dp_size=3,
                engine_gpu_counts=[2, 4, 1],
            )
        ]

        assert sorted(assigned) == [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0)]

    def test_the_first_round_uses_every_source_before_any_source_gets_a_second_target(
        self, p2p_transfer_utils: ModuleType
    ) -> None:
        """With as many targets as sources the plan is a plain one-to-one round robin."""
        by_source = {
            gathered_dp_rank: _targets_of(
                p2p_transfer_utils,
                gathered_dp_rank=gathered_dp_rank,
                gathered_dp_size=4,
                engine_gpu_counts=[2, 2],
            )
            for gathered_dp_rank in range(4)
        }

        assert by_source == {0: [(0, 0)], 1: [(0, 1)], 2: [(1, 0)], 3: [(1, 1)]}

    def test_the_remainder_prefers_the_least_loaded_source_already_serving_that_rank(
        self, p2p_transfer_utils: ModuleType
    ) -> None:
        """Reusing the source of the same rollout engine rank lets one CPU replica serve both writes."""
        by_source = {
            gathered_dp_rank: _targets_of(
                p2p_transfer_utils,
                gathered_dp_rank=gathered_dp_rank,
                gathered_dp_size=2,
                engine_gpu_counts=[1, 1, 1],
            )
            for gathered_dp_rank in range(2)
        }

        assert by_source == {0: [(0, 0), (2, 0)], 1: [(1, 0)]}

    def test_more_engines_than_sources_gives_each_source_one_rollout_engine_rank(
        self, p2p_transfer_utils: ModuleType
    ) -> None:
        """Every extra engine reuses the source that already holds the replica for that rank."""
        by_source = {
            gathered_dp_rank: _targets_of(
                p2p_transfer_utils,
                gathered_dp_rank=gathered_dp_rank,
                gathered_dp_size=2,
                engine_gpu_counts=[2, 2, 2],
            )
            for gathered_dp_rank in range(2)
        }

        assert by_source == {
            0: [(0, 0), (1, 0), (2, 0)],
            1: [(0, 1), (1, 1), (2, 1)],
        }

    def test_a_source_rank_beyond_the_target_count_is_given_nothing(self, p2p_transfer_utils: ModuleType) -> None:
        """More trainer replicas than rollout engine ranks leaves the trailing ranks without targets."""
        assert _targets_of(p2p_transfer_utils, gathered_dp_rank=0, gathered_dp_size=4, engine_gpu_counts=[1]) == [
            (0, 0)
        ]
        for gathered_dp_rank in (1, 2, 3):
            assert (
                _targets_of(
                    p2p_transfer_utils,
                    gathered_dp_rank=gathered_dp_rank,
                    gathered_dp_size=4,
                    engine_gpu_counts=[1],
                )
                == []
            )

    def test_engines_are_addressed_by_their_position_in_the_handed_over_list(
        self, p2p_transfer_utils: ModuleType
    ) -> None:
        """The caller has already dropped the dead cells, so the plan may only name the engines it was given."""
        targets = _targets_of(p2p_transfer_utils, gathered_dp_rank=0, gathered_dp_size=1, engine_gpu_counts=[1, 1])

        assert [rollout_engine_ind for rollout_engine_ind, _rank in targets] == [0, 1]

    def test_no_engine_at_all_plans_no_transfer(self, p2p_transfer_utils: ModuleType) -> None:
        """A trainer handed an empty fleet is not a sender."""
        assert _targets_of(p2p_transfer_utils, gathered_dp_rank=0, gathered_dp_size=2, engine_gpu_counts=[]) == []

    def test_every_task_carries_the_source_pp_shard(self, p2p_transfer_utils: ModuleType) -> None:
        """The target has to know which pipeline shard of the weights it is receiving."""
        tasks = _plan(p2p_transfer_utils, gathered_dp_rank=0, gathered_dp_size=1, pp_rank=2).plan_p2p([2])

        assert {task.source_shard for task in tasks} == {2}


class TestRegisterCPUMemory:
    def test_each_buffer_is_registered_at_its_own_address_and_byte_size(
        self, p2p_transfer_utils: ModuleType, fake_transfer_engine: Any
    ) -> None:
        """A wrong address or length lets the RDMA write read outside the buffer it claims to send."""
        backing = torch.zeros(8)
        params = {"fp32_view": backing[2:5], "bf16": torch.zeros(5, dtype=torch.bfloat16)}

        registry = p2p_transfer_utils.register_cpu_memory(params, fake_transfer_engine)

        assert fake_transfer_engine.registered == [
            (backing.data_ptr() + 2 * 4, 3 * 4),
            (params["bf16"].data_ptr(), 5 * 2),
        ]
        assert registry == {
            "fp32_view": (backing.data_ptr() + 2 * 4, 3, 4),
            "bf16": (params["bf16"].data_ptr(), 5, 2),
        }

    def test_a_nonzero_registration_code_raises_and_names_the_weight(
        self, p2p_transfer_utils: ModuleType, fake_transfer_engine: Any
    ) -> None:
        """An unregistered buffer cannot be read by the transfer engine, so every later write would fail."""
        fake_transfer_engine.register_return_code = 5

        with pytest.raises(RuntimeError, match="register CPU memory failed for weight w, error: 5"):
            p2p_transfer_utils.register_cpu_memory({"w": torch.zeros(2)}, fake_transfer_engine)
