import logging
from argparse import Namespace

import torch.distributed as dist

from miles.utils.distributed_utils import get_gloo_group
from miles.utils.ft_utils.process_group_utils import GroupInfo

from ...training_utils.parallel import ParallelState

logger = logging.getLogger(__name__)


def create_torchtitan_parallel_state(args: Namespace) -> ParallelState:
    """ParallelState for the torchtitan backend: device mesh via torchtitan ParallelDims; FSDP2 stays torch-native."""
    from torchtitan.distributed import ParallelDims

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    cp_size = args.context_parallel_size
    dp_rank = rank // cp_size
    cp_rank = rank % cp_size

    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=world_size // cp_size,
        cp=cp_size,
        tp=1,
        pp=1,
        ep=1,
        world_size=world_size,
    )
    parallel_dims.build_mesh()
    fsdp_mesh = parallel_dims.get_mesh("fsdp")  # dp_shard(+cp) shard dimension for fully_shard

    logger.info(f"[Rank {rank}] torchtitan ParallelDims mesh: world={world_size} cp={cp_size} fsdp={fsdp_mesh.size()}")

    if cp_size > 1:
        from ring_flash_attn import substitute_hf_flash_attn

        substitute_hf_flash_attn(parallel_dims.get_mesh("cp").get_group(), heads_k_stride=1)

    parallel_state = ParallelState(
        intra_dp=GroupInfo(rank=dp_rank, size=world_size // cp_size, group=fsdp_mesh.get_group()),
        intra_dp_cp=GroupInfo(rank=rank, size=world_size, group=dist.group.WORLD, gloo_group=get_gloo_group()),
        cp=GroupInfo(
            rank=cp_rank,
            size=cp_size,
            group=(parallel_dims.get_mesh("cp").get_group() if cp_size > 1 else dist.new_group([rank])),
        ),
        tp=GroupInfo(rank=0, size=1, group=dist.new_group([rank])),
        pp=GroupInfo(rank=0, size=1, group=None),
        ep=GroupInfo(rank=0, size=1, group=None),
        etp=GroupInfo(rank=0, size=1, group=None),
        indep_dp=GroupInfo(rank=0, size=1, group=None),
    )
    parallel_state.dp_mesh = fsdp_mesh
    return parallel_state
