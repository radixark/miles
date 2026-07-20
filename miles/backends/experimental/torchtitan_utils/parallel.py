"""torchtitan ParallelDims -> miles ParallelState, mirroring fsdp_utils/parallel.py's
population pattern but with a titan-native mesh underneath.

The tp GroupInfo must be real (size=1 in v1, but wired for M8's tp>1 milestone):
``training_utils.data.get_batch`` pads batches to a multiple of ``parallel_state.tp.size``,
so a fake trivial GroupInfo here would silently corrupt padding once tp>1 lands.
"""

import logging
from argparse import Namespace

import torch.distributed as dist

from miles.utils.distributed_utils import get_gloo_group
from miles.utils.ft_utils.process_group_utils import GroupInfo

from ...training_utils.parallel import ParallelState

logger = logging.getLogger(__name__)


def create_torchtitan_parallel_state(args: Namespace):
    from torchtitan.distributed import ParallelDims

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    parallel_dims = ParallelDims(
        dp_replicate=args.tt_dp_replicate,
        dp_shard=-1,
        cp=1,
        tp=args.tt_tensor_parallel_size,
        pp=1,
        ep=args.tt_expert_parallel_size,
        world_size=world_size,
    )
    parallel_dims.build_mesh()
    dp_mesh = parallel_dims.get_mesh("dp")

    tp_size = args.tt_tensor_parallel_size
    tp_group = parallel_dims.get_mesh("tp").get_group() if tp_size > 1 else dist.new_group([rank])
    tp_rank = dist.get_rank(tp_group) if tp_size > 1 else 0

    logger.info(
        f"[Rank {rank}] torchtitan ParallelDims: world_size={world_size} "
        f"dp_shard={parallel_dims.dp_shard} dp_replicate={parallel_dims.dp_replicate} "
        f"tp={tp_size} ep={args.tt_expert_parallel_size}"
    )

    parallel_state = ParallelState(
        intra_dp=GroupInfo(
            rank=dist.get_rank(dp_mesh.get_group()),
            size=dp_mesh.size(),
            group=dp_mesh.get_group(),
        ),
        intra_dp_cp=GroupInfo(
            rank=rank,
            size=world_size,
            group=dist.group.WORLD,
            gloo_group=get_gloo_group(),
        ),
        cp=GroupInfo(rank=0, size=1, group=dist.new_group([rank])),
        tp=GroupInfo(rank=tp_rank, size=tp_size, group=tp_group),
        pp=GroupInfo(rank=0, size=1, group=None),
        ep=GroupInfo(rank=0, size=1, group=None),
        etp=GroupInfo(rank=0, size=1, group=None),
        indep_dp=GroupInfo(rank=0, size=1, group=None),
    )
    parallel_state.dp_mesh = dp_mesh
    parallel_state.parallel_dims = parallel_dims

    return parallel_state
