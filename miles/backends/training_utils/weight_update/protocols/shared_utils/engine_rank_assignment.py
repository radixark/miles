from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.utils import get_data_replica_rank_and_size


@dataclass(frozen=True)
class EngineRankAssignment:
    """The weights of one engine rank that a trainer rank sends, and the engines it sends them to.

    These engines have the same layout, so the trainer rank prepares the weights once for all of them.
    """

    engine_rank: int
    engine_indices: tuple[int, ...]


def assign_engine_ranks(
    parallel_state: ParallelState,
    placement: WeightUpdatePlacement,
    engine_gpu_counts: Sequence[int],
) -> list[EngineRankAssignment]:
    """Returns the engine ranks this trainer rank sends weights to in p2p weight updates.

    The p2p protocol calls it in `connect`. An empty list means this trainer rank sends nothing.
    """
    replica_rank, replica_size = get_data_replica_rank_and_size(parallel_state, placement)
    return assign_engine_ranks_for_replica(
        replica_rank=replica_rank, replica_size=replica_size, engine_gpu_counts=engine_gpu_counts
    )


def assign_engine_ranks_for_replica(
    replica_rank: int, replica_size: int, engine_gpu_counts: Sequence[int]
) -> list[EngineRankAssignment]:
    """Same as `assign_engine_ranks`, but for a given replica index and count.

    Replicas are the trainer ranks that hold the same weights. Every (engine, engine rank) gets exactly
    one sender. First each replica gets one, so all of them send in parallel; the rest go to a replica
    already sending the same engine rank, which prepares those weights once for several engines.
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
