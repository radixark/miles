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

_MESH_TO_FIELD = {
    "batch": "intra_dp",
    "tp": "tp",
    "pp": "pp",
    "ep": "ep",
}


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


def create_titan_parallel_state(parallel_dims, *, is_pp_last_stage: bool = True) -> ParallelState:
    """Map titan's meshes onto ParallelState; a degree-1 axis becomes a single-rank group.

    The sample-parallel group carries a gloo group even at degree 1, because the
    shared log gathering reduces over it unconditionally.
    """
    rank = dist.get_rank()
    self_group = dist.new_group([rank])
    trivial = GroupInfo(rank=0, size=1, group=self_group)

    fields: dict[str, GroupInfo] = {"intra_dp_cp": trivial, "cp": trivial}
    for mesh_name, field in _MESH_TO_FIELD.items():
        mesh = parallel_dims.get_optional_mesh(mesh_name)
        if mesh is None:
            fields[field] = (
                GroupInfo(rank=0, size=1, group=self_group, gloo_group=_gloo_subgroup([rank]))
                if field == "intra_dp"
                else trivial
            )
            continue
        group = mesh.get_group()
        member_ranks = dist.get_process_group_ranks(group)
        fields[field] = GroupInfo(
            rank=dist.get_rank(group=group),
            size=dist.get_world_size(group=group),
            group=group,
            gloo_group=_gloo_subgroup(member_ranks) if field == "intra_dp" else None,
        )
    fields["intra_dp_cp"] = fields["intra_dp"]

    meshes = {name: parallel_dims.get_mesh(name) for name in ("fsdp",) if parallel_dims.get_optional_mesh(name)}

    state = ParallelState(
        intra_dp=fields["intra_dp"],
        intra_dp_cp=fields["intra_dp_cp"],
        cp=fields["cp"],
        tp=fields["tp"],
        pp=fields["pp"],
        ep=fields["ep"],
        etp=trivial,
        indep_dp=trivial,
        meshes=meshes,
        is_pp_last_stage=is_pp_last_stage,
        vpp_size=1,
    )
    logger.info(
        f"[Rank {rank}] titan ParallelState: dp={state.intra_dp.size} dp_cp={state.intra_dp_cp.size} "
        f"cp={state.cp.size} tp={state.tp.size} pp={state.pp.size} ep={state.ep.size} "
        f"pp_last={is_pp_last_stage}"
    )
    return state
