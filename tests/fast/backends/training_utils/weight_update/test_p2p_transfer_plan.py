from argparse import Namespace
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _make_plan(utils, *, gathered_dp_rank: int, gathered_dp_size: int, pp_rank: int = 0, sglang_pp_size: int = 1):
    parallel_state = SimpleNamespace(pp=SimpleNamespace(rank=pp_rank, size=1))
    with (
        patch.object(utils, "get_parallel_state", return_value=parallel_state),
        patch.object(utils, "get_data_replica_rank_and_size", return_value=(gathered_dp_rank, gathered_dp_size)),
    ):
        return utils.RemoteTransferPlan(Namespace(sglang_pp_size=sglang_pp_size))


def _plan_of_every_source_rank(utils, *, gathered_dp_size: int, engine_gpu_counts: list[int]) -> list[list[tuple]]:
    return [
        [
            (task.engine_ind, task.engine_rank)
            for task in _make_plan(utils, gathered_dp_rank=source_rank, gathered_dp_size=gathered_dp_size).plan_p2p(
                engine_gpu_counts
            )
        ]
        for source_rank in range(gathered_dp_size)
    ]


class TestPlanP2P:
    """The plan maps the engines it was handed, not a topology derived from the launch flags."""

    def test_every_engine_rank_is_served_by_exactly_one_source_rank(self, p2p_transfer_utils) -> None:
        """A rank served twice is written twice over the same buffers, and one served zero times keeps old weights."""
        plans = _plan_of_every_source_rank(p2p_transfer_utils, gathered_dp_size=4, engine_gpu_counts=[2, 2, 2])

        assert Counter(target for plan in plans for target in plan) == Counter(
            {(engine_ind, engine_rank): 1 for engine_ind in range(3) for engine_rank in range(2)}
        )

    def test_more_targets_than_source_ranks_still_covers_every_target(self, p2p_transfer_utils) -> None:
        """The trainer cell is smaller than the fleet it feeds, so the leftover targets must be shared out."""
        plans = _plan_of_every_source_rank(p2p_transfer_utils, gathered_dp_size=4, engine_gpu_counts=[3, 3])

        assert sorted(target for plan in plans for target in plan) == [
            (engine_ind, engine_rank) for engine_ind in range(2) for engine_rank in range(3)
        ]

    def test_engines_of_different_sizes_are_enumerated_by_their_own_rank_count(self, p2p_transfer_utils) -> None:
        """Assuming one uniform GPU count per engine would invent ranks the smaller engine does not have."""
        plans = _plan_of_every_source_rank(p2p_transfer_utils, gathered_dp_size=2, engine_gpu_counts=[1, 3])

        assert sorted(target for plan in plans for target in plan) == [(0, 0), (1, 0), (1, 1), (1, 2)]

    def test_fewer_targets_than_source_ranks_leaves_the_rest_without_work(self, p2p_transfer_utils) -> None:
        """Every trainer rank still joins the bucket gathers, but only the assigned ones may send."""
        plans = _plan_of_every_source_rank(p2p_transfer_utils, gathered_dp_size=4, engine_gpu_counts=[1])

        assert plans == [[(0, 0)], [], [], []]

    def test_an_empty_engine_list_gives_every_source_rank_nothing_to_send(self, p2p_transfer_utils) -> None:
        """All the inference cells can be gone, and planning must answer that rather than fail."""
        plans = _plan_of_every_source_rank(p2p_transfer_utils, gathered_dp_size=3, engine_gpu_counts=[])

        assert plans == [[], [], []]

    def test_the_engines_are_addressed_by_their_position_in_the_supplied_list(self, p2p_transfer_utils) -> None:
        """The caller filters dead cells out, so a surviving engine's global GPU offset is not its index here."""
        plans = _plan_of_every_source_rank(p2p_transfer_utils, gathered_dp_size=2, engine_gpu_counts=[2, 2])

        assert {engine_ind for plan in plans for engine_ind, _rank in plan} == {0, 1}

    def test_a_replanned_fleet_follows_the_new_engine_list(self, p2p_transfer_utils) -> None:
        """After a reconnect the same plan object must forget the target count it saw before."""
        plan = _make_plan(p2p_transfer_utils, gathered_dp_rank=1, gathered_dp_size=2)

        before = [(task.engine_ind, task.engine_rank) for task in plan.plan_p2p([2, 2])]
        after = [(task.engine_ind, task.engine_rank) for task in plan.plan_p2p([2])]

        assert before == [(0, 1), (1, 1)]
        assert after == [(0, 1)]
        assert [(task.engine_ind, task.engine_rank) for task in plan.plan_p2p([1])] == []

    def test_the_source_shard_of_every_task_is_the_pipeline_stage(self, p2p_transfer_utils) -> None:
        """Each pipeline stage owns a different slice of the model, so the target must know which one wrote."""
        plan = _make_plan(p2p_transfer_utils, gathered_dp_rank=0, gathered_dp_size=1, pp_rank=2)

        assert [task.source_shard for task in plan.plan_p2p([2])] == [2, 2]

    @pytest.mark.parametrize("engine_gpu_count", [0, -1, True])
    def test_an_engine_without_a_positive_gpu_count_is_rejected(
        self, p2p_transfer_utils, engine_gpu_count: int
    ) -> None:
        """An engine planned no targets for keeps serving the old weights while the run believes it was updated."""
        plan = _make_plan(p2p_transfer_utils, gathered_dp_rank=0, gathered_dp_size=1)

        with pytest.raises(AssertionError, match="no rank to send weights to"):
            plan.plan_p2p([2, engine_gpu_count])

    def test_rollout_pipeline_parallelism_is_still_refused(self, p2p_transfer_utils) -> None:
        """A pipelined rollout engine splits the weights along a dimension this plan does not model."""
        with pytest.raises(NotImplementedError, match="pipeline parallelism"):
            _make_plan(p2p_transfer_utils, gathered_dp_rank=0, gathered_dp_size=1, sglang_pp_size=2)
