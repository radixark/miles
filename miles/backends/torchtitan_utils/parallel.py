"""Translate torchtitan's ParallelDims into miles' shared ParallelState.

torchtitan owns all mesh construction; this is a mapping over the named meshes
it exposes. Context parallelism is deliberately absent: the trainer shards the
sequence and gathers the logits back before the loss, so the shared helpers are
told cp=1 and reduce metrics over the sample-parallel ("batch") mesh.
"""

import logging

import torch.distributed as dist

from miles.backends.training_utils.parallel import ParallelState
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.ft_utils.process_group_utils import GroupInfo

logger = logging.getLogger(__name__)


def parallel_dims_from_config(parallelism_config):
    from torchtitan.distributed import ParallelDims

    return ParallelDims.from_config(parallelism_config, dist.get_world_size())


def _gloo_subgroup(my_ranks: list[int]):
    """A gloo subgroup over ``my_ranks``; every rank joins every distinct group's creation."""
    if len(my_ranks) == dist.get_world_size():
        return get_gloo_group()

    all_lists: list = [None] * dist.get_world_size()
    dist.all_gather_object(all_lists, my_ranks)
    my_group = None
    for ranks in sorted({tuple(lst) for lst in all_lists}):
        group = dist.new_group(list(ranks), backend="gloo")
        if dist.get_rank() in ranks:
            my_group = group
    return my_group


def _group_info(parallel_dims, mesh_name: str, *, with_gloo: bool = False) -> GroupInfo:
    """The GroupInfo for one of titan's meshes; an absent (degree-1) mesh is this rank alone."""
    mesh = parallel_dims.get_optional_mesh(mesh_name)
    if mesh is None:
        ranks = [dist.get_rank()]
        group = dist.new_group(ranks)
    else:
        group = mesh.get_group()
        ranks = dist.get_process_group_ranks(group)
    return GroupInfo(
        rank=dist.get_rank(group=group),
        size=dist.get_world_size(group=group),
        group=group,
        gloo_group=_gloo_subgroup(ranks) if with_gloo else None,
    )


def create_titan_parallel_state(parallel_dims, *, is_pp_last_stage: bool = True) -> ParallelState:
    """Map titan's meshes onto ParallelState.

    The sample-parallel group carries a gloo group even at degree 1, because the
    shared log gathering reduces over it unconditionally.
    """
    alone = GroupInfo(rank=0, size=1, group=dist.new_group([dist.get_rank()]))
    dp = _group_info(parallel_dims, "batch", with_gloo=True)
    state = ParallelState(
        intra_dp=dp,
        intra_dp_cp=dp,
        cp=alone,
        tp=_group_info(parallel_dims, "tp"),
        pp=_group_info(parallel_dims, "pp"),
        ep=_group_info(parallel_dims, "ep"),
        etp=alone,
        indep_dp=alone,
        meshes={"fsdp": parallel_dims.get_mesh("fsdp")} if parallel_dims.get_optional_mesh("fsdp") else {},
        is_pp_last_stage=is_pp_last_stage,
        vpp_size=1,
    )
    logger.info(
        f"[Rank {dist.get_rank()}] titan ParallelState: dp={state.intra_dp.size} tp={state.tp.size} "
        f"pp={state.pp.size} ep={state.ep.size} pp_last={is_pp_last_stage}"
    )
    return state
