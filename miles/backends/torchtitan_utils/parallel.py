"""Translate torchtitan's ParallelDims into miles' shared ParallelState.

ParallelState is what every shared helper in ``training_utils`` reads (data
iteration, loss normalization, logging), so a backend's only topology
responsibility is producing one. torchtitan owns all mesh construction --
``ParallelDims.from_config`` (with ``data_parallel_shard_degree=-1`` inferring
the FSDP degree) builds the dims, and this module is a pure mapping over the
named meshes it exposes.
"""

import logging

import torch.distributed as dist

from miles.backends.training_utils.parallel import ParallelState
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.ft_utils.process_group_utils import GroupInfo

logger = logging.getLogger(__name__)

# titan mesh name -> miles ParallelState field. titan's "batch" mesh is
# dp_replicate x dp_shard, the sample-parallel view.
#
# Context parallelism is deliberately absent: it stays internal to the trainer,
# which shards the sequence itself and gathers the logits back before the loss
# sees them. So the shared helpers must believe cp is 1 -- they would otherwise
# slice the rollout a second time (in get_batch) and reduce metrics over ranks
# that hold identical values. That is also why intra_dp_cp maps to "batch"
# rather than titan's "loss" mesh, which folds cp in.
_MESH_TO_FIELD = {
    "batch": "intra_dp",
    "tp": "tp",
    "pp": "pp",
    "ep": "ep",
}


def parallel_dims_from_config(parallelism_config):
    """torchtitan's own dims construction, exactly as Trainer.init_distributed does it."""
    from torchtitan.distributed import ParallelDims

    return ParallelDims.from_config(parallelism_config, dist.get_world_size())


def _gloo_subgroup(my_ranks: list[int]):
    """A gloo subgroup over exactly ``my_ranks``.

    Object-based reductions over the DP-CP group need gloo, and the shared
    helpers use it even when the group is this rank alone (model parallelism
    can shrink DP-CP to 1). ``new_group`` is a collective every rank must join
    for every subgroup, so the groups are enumerated globally: each rank
    contributes its own member list and all ranks create all distinct groups.
    """
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
    """Map titan's meshes onto ParallelState.

    Axes titan leaves at degree 1 have no mesh; they become trivial single-rank
    groups, which is what the shared helpers expect for a disabled dimension.
    ``is_pp_last_stage`` comes from the trainer's stage placement (interleaved
    schedules can place the last stage on any rank), and gates whose ranks
    report loss metrics and log probs.
    """
    rank = dist.get_rank()
    self_group = dist.new_group([rank])
    trivial = GroupInfo(rank=0, size=1, group=self_group)

    fields: dict[str, GroupInfo] = {"intra_dp_cp": trivial, "cp": trivial}
    for mesh_name, field in _MESH_TO_FIELD.items():
        mesh = parallel_dims.get_optional_mesh(mesh_name)
        if mesh is None:
            # titan builds no mesh for a degree-1 axis. The sample-parallel one
            # still needs a gloo group even when it is this rank alone -- the
            # shared log gathering reduces over it unconditionally, and a None
            # group fails with "Group None is not registered". Model parallelism
            # alone (tp x pp filling the world) puts us here.
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
            # Shared helpers reduce metrics over the DP-CP group, and some of
            # those reductions are object-based, which needs a gloo group --
            # a degree-1 DP-CP included (log gathering runs regardless).
            gloo_group=_gloo_subgroup(member_ranks) if field == "intra_dp" else None,
        )
    # The metric-reduction axis is the sample-parallel one (see _MESH_TO_FIELD).
    fields["intra_dp_cp"] = fields["intra_dp"]

    meshes = {name: parallel_dims.get_mesh(name) for name in ("fsdp",) if parallel_dims.get_optional_mesh(name)}

    state = ParallelState(
        intra_dp=fields["intra_dp"],
        intra_dp_cp=fields["intra_dp_cp"],
        cp=fields["cp"],
        tp=fields["tp"],
        pp=fields["pp"],
        ep=fields["ep"],
        # titan has no separate expert-tensor axis; its EP region uses "efsdp".
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
