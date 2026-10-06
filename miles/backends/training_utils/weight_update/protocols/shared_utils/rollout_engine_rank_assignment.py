from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.utils import get_data_replica_rank_and_size


@dataclass(frozen=True)
class RolloutEngineRankAssignment:
    """The weights of one rollout engine rank that a trainer rank sends, and the rollout engines it sends
    them to.

    These rollout engines have the same layout, so the trainer rank prepares the weights once for all of them.
    """

    rollout_engine_rank: int
    rollout_engine_indices: tuple[int, ...]


def assign_rollout_engine_ranks(
    parallel_state: ParallelState,
    placement: WeightUpdatePlacement,
    engine_gpu_counts: Sequence[int],
) -> list[RolloutEngineRankAssignment]:
    """Returns the rollout engine ranks this trainer rank sends weights to in p2p weight updates.

    The p2p protocol calls it in `connect`. An empty list means this trainer rank sends nothing.
    """
    replica_rank, replica_size = get_data_replica_rank_and_size(parallel_state, placement)
    return assign_rollout_engine_ranks_for_replica(
        replica_rank=replica_rank, replica_size=replica_size, engine_gpu_counts=engine_gpu_counts
    )


def assign_rollout_engine_ranks_for_replica(
    replica_rank: int, replica_size: int, engine_gpu_counts: Sequence[int]
) -> list[RolloutEngineRankAssignment]:
    """Same as `assign_rollout_engine_ranks`, but for a given replica index and count.

    Replicas are the trainer ranks that hold the same weights. Every (rollout engine, rollout engine rank)
    gets exactly one sender. First each replica gets one, so all of them send in parallel; the rest go to a
    replica already sending the same rollout engine rank, which prepares those weights once for several
    rollout engines.
    """
    targets = [
        (rollout_engine_ind, rollout_engine_rank)
        for rollout_engine_ind, gpu_count in enumerate(engine_gpu_counts)
        for rollout_engine_rank in range(gpu_count)
    ]
    rollout_engine_indices_by_replica = [defaultdict(list) for _ in range(replica_size)]
    for replica, (rollout_engine_ind, rollout_engine_rank) in zip(range(replica_size), targets, strict=False):
        rollout_engine_indices_by_replica[replica][rollout_engine_rank].append(rollout_engine_ind)

    next_round_robin_replica = 0
    for rollout_engine_ind, rollout_engine_rank in targets[replica_size:]:
        loads = [
            len(rollout_engine_indices[rollout_engine_rank])
            for rollout_engine_indices in rollout_engine_indices_by_replica
        ]
        if max(loads) > 0:
            _, replica = min((load, replica) for replica, load in enumerate(loads) if load > 0)
        else:
            replica = next_round_robin_replica % replica_size
            next_round_robin_replica += 1
        rollout_engine_indices_by_replica[replica][rollout_engine_rank].append(rollout_engine_ind)

    return [
        RolloutEngineRankAssignment(
            rollout_engine_rank=rollout_engine_rank, rollout_engine_indices=tuple(rollout_engine_indices)
        )
        for rollout_engine_rank, rollout_engine_indices in rollout_engine_indices_by_replica[replica_rank].items()
    ]
