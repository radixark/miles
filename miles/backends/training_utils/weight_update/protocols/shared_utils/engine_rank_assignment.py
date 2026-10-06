from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.utils import get_data_replica_rank_and_size


@dataclass(frozen=True)
class EngineRankAssignment:
    engine_rank: int
    # one load of this engine rank's shard serves all of them
    engine_indices: tuple[int, ...]


def assign_engine_ranks(
    parallel_state: ParallelState,
    placement: WeightUpdatePlacement,
    engine_gpu_counts: Sequence[int],
) -> list[EngineRankAssignment]:
    """Engine ranks this trainer rank writes, over the engines handed over; empty when it is not a
    sender. Collective."""
    replica_rank, replica_size = get_data_replica_rank_and_size(parallel_state, placement)
    return assign_engine_ranks_for_replica(
        replica_rank=replica_rank, replica_size=replica_size, engine_gpu_counts=engine_gpu_counts
    )


def assign_engine_ranks_for_replica(
    replica_rank: int, replica_size: int, engine_gpu_counts: Sequence[int]
) -> list[EngineRankAssignment]:
    """Every (engine, engine rank) goes to exactly one of the `replica_size` data replicas.

    The first `replica_size` targets go round robin, one per replica; each remaining target goes to the
    least-loaded replica already writing that engine rank, so a replica loads one engine rank's shard for
    several engines, or round robin when no replica writes that engine rank yet.
    """
    targets = [
        (engine_index, engine_rank)
        for engine_index, gpu_count in enumerate(engine_gpu_counts)
        for engine_rank in range(gpu_count)
    ]
    engine_indices_by_replica = [defaultdict(list) for _ in range(replica_size)]
    for replica, (engine_index, engine_rank) in zip(range(replica_size), targets, strict=False):
        engine_indices_by_replica[replica][engine_rank].append(engine_index)

    next_round_robin_replica = 0
    for engine_index, engine_rank in targets[replica_size:]:
        loads = [len(engine_indices[engine_rank]) for engine_indices in engine_indices_by_replica]
        if max(loads) > 0:
            _, replica = min((load, replica) for replica, load in enumerate(loads) if load > 0)
        else:
            replica = next_round_robin_replica % replica_size
            next_round_robin_replica += 1
        engine_indices_by_replica[replica][engine_rank].append(engine_index)

    return [
        EngineRankAssignment(engine_rank=engine_rank, engine_indices=tuple(engine_indices))
        for engine_rank, engine_indices in engine_indices_by_replica[replica_rank].items()
    ]
