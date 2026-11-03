from miles.backends.training_utils.weight_update.protocols.colocated_p2p import assign_colocated_rollout_engine_ranks
from miles.backends.training_utils.weight_update.protocols.utils.rollout_engine_rank_assignment import (
    RolloutEngineRankAssignment,
)


def test_each_trainer_rank_serves_the_engine_rank_on_its_own_gpu():
    """A pull copies out of this GPU's memory, so a trainer rank paired with an engine rank on another GPU cannot
    serve it; a placeholder slot between engines serves nothing."""
    engine_gpu_counts, engine_gpu_offsets = [2, 4], [0, 3]

    assignments = [
        assign_colocated_rollout_engine_ranks(
            trainer_rank=trainer_rank, engine_gpu_counts=engine_gpu_counts, engine_gpu_offsets=engine_gpu_offsets
        )
        for trainer_rank in range(7)
    ]

    assert assignments == [
        [RolloutEngineRankAssignment(rollout_engine_rank=0, rollout_engine_indices=(0,))],
        [RolloutEngineRankAssignment(rollout_engine_rank=1, rollout_engine_indices=(0,))],
        [],
        *[[RolloutEngineRankAssignment(rollout_engine_rank=rank, rollout_engine_indices=(1,))] for rank in range(4)],
    ]
