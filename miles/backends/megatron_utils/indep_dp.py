import logging
from collections.abc import Sequence
from datetime import timedelta
from typing import TYPE_CHECKING

import torch.distributed as dist
from megatron.core import mpu

from miles.utils.indep_dp import IndepDPInfo
from miles.utils.process_group_utils import GeneralPGUtil, GroupInfo
from miles.utils.structured_log import log_structured

from ..training_utils.log_utils import aggregate_train_losses
from ..training_utils.parallel import ParallelState

if TYPE_CHECKING:
    from megatron.core.distributed import DistributedDataParallel as DDP

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


def allreduce_grads_and_losses_across_replicas(
    args, model: Sequence["DDP"], parallel_state: ParallelState, losses_reduced: list
) -> dict[str, float]:
    assert not args.calculate_per_token_loss, "calculate_per_token_loss is not supported with indep_dp yet"
    assert parallel_state.intra_dp.size == 1, (
        f"indep_dp requires intra_dp.size == 1, got {parallel_state.intra_dp.size}. "
        "Simultaneous intra and indep DP is not supported."
    )

    pg = parallel_state.indep_dp.group
    util = GeneralPGUtil.create(pg)
    log_structured(
        logger.info,
        op="cross_cell",
        phase="start",
        kind="grad_allreduce",
        cell_rank=parallel_state.indep_dp.rank,
        members=parallel_state.indep_dp.size,
        **parallel_state.indep_dp.debug_info,
    )

    loss_reduced: dict[str, float] = {}
    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        loss_reduced = aggregate_train_losses(losses_reduced)
    for model_chunk in model:
        # mimic: DistributedDataParallel.start_grad_sync
        for bucket_group in model_chunk.bucket_groups + model_chunk.expert_parallel_bucket_groups:
            for bucket in bucket_group.buckets:
                util.all_reduce(bucket.grad_data, pg, op=dist.ReduceOp.SUM)

    log_structured(
        logger.info,
        op="cross_cell",
        phase="end",
        kind="grad_allreduce",
        cell_rank=parallel_state.indep_dp.rank,
        members=parallel_state.indep_dp.size,
        **parallel_state.indep_dp.debug_info,
    )
    return loss_reduced
