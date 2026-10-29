import logging

import torch.distributed as dist
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.parallelism_context import DistributedTopology

from miles.backends.training_utils.parallel import ParallelState
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.ft_utils.process_group_utils import GroupInfo

logger = logging.getLogger(__name__)


def parallelism_context_from_config(parallelism_config: ParallelismConfig, *, output_dir: str) -> ParallelismContext:
    return ParallelismContext.from_config(
        parallelism_config, DistributedTopology(dist.get_world_size()), dump_folder=output_dir
    )


def _gloo_subgroup(my_ranks: list[int]):
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


def _group_info(parallelism: ParallelismContext, mesh_name: str, *, with_gloo: bool = False) -> GroupInfo:
    mesh = parallelism.get_optional_mesh(mesh_name)
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


def create_titan_parallel_state(parallelism: ParallelismContext, *, is_pp_last_stage: bool = True) -> ParallelState:
    alone = GroupInfo(rank=0, size=1, group=dist.new_group([dist.get_rank()]))
    dp = _group_info(parallelism, "dp", with_gloo=True)
    state = ParallelState(
        intra_dp=dp,
        intra_dp_cp=dp,
        cp=alone,
        tp=_group_info(parallelism, "tp"),
        pp=_group_info(parallelism, "pp"),
        ep=_group_info(parallelism, "ep"),
        etp=alone,
        indep_dp=alone,
        meshes={},
        is_pp_last_stage=is_pp_last_stage,
        vpp_size=1,
    )
    logger.info(
        f"[Rank {dist.get_rank()}] titan ParallelState: dp={state.intra_dp.size} tp={state.tp.size} "
        f"pp={state.pp.size} ep={state.ep.size} pp_last={is_pp_last_stage}"
    )
    return state
