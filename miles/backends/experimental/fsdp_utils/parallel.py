import logging
from argparse import Namespace

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from miles.utils.distributed_utils import get_gloo_group

from miles.utils.ft_utils.process_group_utils import GroupInfo

from ...training_utils.parallel import ParallelState

logger = logging.getLogger(__name__)


def build_fsdp_meshes(
    device_type: str,
    world_size: int,
    context_parallel_size: int,
    dp_replicate_size: int,
) -> dict[str, DeviceMesh]:
    """Build the data/context-parallel views and the FSDP2 shard mesh."""
    data_parallel_size = world_size // context_parallel_size

    dp_cp_mesh = init_device_mesh(
        device_type,
        mesh_shape=(data_parallel_size, context_parallel_size),
        mesh_dim_names=("dp", "cp"),
    )
    dp_mesh = dp_cp_mesh["dp"]
    fsdp_mesh = dp_mesh
    if dp_replicate_size > 1:
        fsdp_mesh = dp_mesh._unflatten(
            0,
            (dp_replicate_size, data_parallel_size // dp_replicate_size),
            ("dp_replicate", "dp_shard"),
        )

    return {
        "dp_cp": dp_cp_mesh,
        "dp": dp_mesh,
        "cp": dp_cp_mesh["cp"],
        "fsdp": fsdp_mesh,
    }


def create_fsdp_parallel_state(args: Namespace) -> ParallelState:
    """Create a ParallelState instance for FSDP configuration."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    cp_size = args.context_parallel_size
    dp_rank = rank // cp_size
    cp_rank = rank % cp_size

    meshes = build_fsdp_meshes(
        device_type="cuda",
        world_size=world_size,
        context_parallel_size=cp_size,
        dp_replicate_size=args.dp_replicate_size,
    )
    dp_mesh = meshes["dp"]
    cp_mesh = meshes["cp"]
    fsdp_mesh = meshes["fsdp"]

    logger.info(
        f"[Rank {rank}] FSDP mesh shape={fsdp_mesh.shape}, "
        f"dp_replicate_size={args.dp_replicate_size}, "
        f"dp_shard_size={(world_size // cp_size) // args.dp_replicate_size}, "
        f"dp_rank={dp_rank}, cp_rank={cp_rank}"
    )

    # Setup Ring Flash Attention with CP group from mesh (only when cp_size > 1)
    if cp_size > 1:
        # TODO: Pin ring_flash_attn for torch 2.11+ compatibility; keep this local import to unblock non-FSDP+CP paths.
        from ring_flash_attn import substitute_hf_flash_attn

        substitute_hf_flash_attn(cp_mesh.get_group(), heads_k_stride=1)
        logger.info(f"[Rank {rank}] CP initialized via device mesh")
    else:
        logger.info(f"[Rank {rank}] Pure DP mode (cp_size=1)")

    parallel_state = ParallelState(
        intra_dp=GroupInfo(
            rank=dp_rank,
            size=world_size // cp_size,
            group=dp_mesh.get_group(),
        ),
        intra_dp_cp=GroupInfo(
            rank=rank,
            size=world_size,
            group=dist.group.WORLD,
            gloo_group=get_gloo_group(),
        ),
        cp=GroupInfo(
            rank=cp_rank,
            size=cp_size,
            group=cp_mesh.get_group(),
        ),
        tp=GroupInfo(
            rank=0,
            size=1,
            group=dist.new_group([rank]),
        ),
        pp=GroupInfo(rank=0, size=1, group=None),
        ep=GroupInfo(rank=0, size=1, group=None),
        etp=GroupInfo(rank=0, size=1, group=None),
        indep_dp=GroupInfo(
            rank=0,
            size=1,
            group=None,
        ),
        meshes=meshes,
    )

    return parallel_state
