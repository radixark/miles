import logging
from datetime import timedelta

import torch.distributed as dist

from miles.utils.indep_dp import IndepDPInfo
from miles.utils.process_group_utils import GroupInfo
from miles.utils.structured_log import log_structured

from ..training_utils.parallel import ParallelState

logger = logging.getLogger(__name__)


def create_indep_dp_group(
    store_addr: str | None,
    indep_dp_info: IndepDPInfo,
    megatron_rank: int,
    megatron_world_size: int,
) -> GroupInfo:
    if indep_dp_info.alive_size <= 1:
        return GroupInfo(rank=0, size=1, group=None, debug_info={"quorum": indep_dp_info.quorum_id})

    try:
        from torchft.process_group import ProcessGroupGloo, ProcessGroupNCCL
    except ImportError as e:
        raise ImportError("torchft is required for indep_dp. Install with: pip install torchft") from e

    _TIMEOUT = timedelta(seconds=120)

    def _create(pg_cls: type, backend_name: str) -> dist.ProcessGroup:
        pg = pg_cls(timeout=_TIMEOUT)
        pg.configure(
            store_addr=f"{store_addr}/indep_dp/{backend_name}/{indep_dp_info.quorum_id}/{megatron_rank}",
            replica_id=str(indep_dp_info.cell_index),
            rank=indep_dp_info.alive_rank,
            world_size=indep_dp_info.alive_size,
            quorum_id=indep_dp_info.quorum_id,
            group_rank=megatron_rank,
            group_world_size=megatron_world_size,
        )
        return pg

    nccl_pg = _create(ProcessGroupNCCL, "nccl")
    gloo_pg = _create(ProcessGroupGloo, "gloo")
    log_structured(
        logger.info,
        op="create_pg",
        cell=indep_dp_info.cell_index,
        cell_rank=indep_dp_info.alive_rank,
        members=indep_dp_info.alive_size,
        quorum=indep_dp_info.quorum_id,
        megatron_rank=megatron_rank,
        megatron_world_size=megatron_world_size,
    )
    return GroupInfo(
        rank=indep_dp_info.alive_rank,
        size=indep_dp_info.alive_size,
        group=nccl_pg,
        gloo_group=gloo_pg,
        debug_info={"quorum": indep_dp_info.quorum_id},
    )


def reconfigure_indep_dp_group(
    parallel_state: ParallelState,
    store_addr: str | None,
    indep_dp_info: IndepDPInfo,
    megatron_rank: int,
    megatron_world_size: int,
) -> None:
    """Shut down old indep_dp PGs and create new ones with a fresh quorum_id."""
    old = parallel_state.indep_dp
    log_structured(
        logger.info,
        op="reconfig",
        phase="start",
        cell=indep_dp_info.cell_index,
        quorum_to=indep_dp_info.quorum_id,
        alive_rank=indep_dp_info.alive_rank,
        members=indep_dp_info.alive_size,
    )
    for g in [old.group, old.gloo_group]:
        if g is not None:
            g.shutdown()

    parallel_state.indep_dp = create_indep_dp_group(
        store_addr=store_addr,
        indep_dp_info=indep_dp_info,
        megatron_rank=megatron_rank,
        megatron_world_size=megatron_world_size,
    )
    log_structured(
        logger.info, op="reconfig", phase="end", cell=indep_dp_info.cell_index, quorum=indep_dp_info.quorum_id
    )
