import logging
from argparse import Namespace
from collections.abc import Sequence

import torch
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_model_config
from megatron.training.global_vars import get_args

from ..training_utils.parallel import GroupInfo, ParallelState, get_parallel_state

logger = logging.getLogger(__name__)


def create_megatron_parallel_state() -> ParallelState:
    vpp_size, microbatch_group_size_per_vp_stage = _compute_vpp_fields()

    def _create_intra_dp(with_context_parallel: bool):
        return GroupInfo(
            rank=mpu.get_data_parallel_rank(with_context_parallel=with_context_parallel),
            size=mpu.get_data_parallel_world_size(with_context_parallel=with_context_parallel),
            group=mpu.get_data_parallel_group(with_context_parallel=with_context_parallel),
            gloo_group=mpu.get_data_parallel_group_gloo(with_context_parallel=with_context_parallel),
        )

    return ParallelState(
        intra_dp=_create_intra_dp(with_context_parallel=False),
        intra_dp_cp=_create_intra_dp(with_context_parallel=True),
        cp=GroupInfo(
            rank=mpu.get_context_parallel_rank(),
            size=mpu.get_context_parallel_world_size(),
            group=mpu.get_context_parallel_group(),
        ),
        tp=GroupInfo(
            rank=mpu.get_tensor_model_parallel_rank(),
            size=mpu.get_tensor_model_parallel_world_size(),
            group=mpu.get_tensor_model_parallel_group(),
        ),
        is_pp_last_stage=mpu.is_pipeline_last_stage(),
        vpp_size=vpp_size,
        microbatch_group_size_per_vp_stage=microbatch_group_size_per_vp_stage,
    )


def _compute_vpp_fields() -> tuple[int, int | None]:
    vpp_size_value = mpu.get_virtual_pipeline_model_parallel_world_size()
    if vpp_size_value is None or vpp_size_value <= 1:
        return 1, None

    return vpp_size_value, get_args().pipeline_model_parallel_size


def verify_megatron_parallel_state(
    model: torch.nn.Module | Sequence[torch.nn.Module],
) -> None:
    """Verify that ParallelState fields match what the model config produces."""
    parallel_state = get_parallel_state()
    vpp_size_value = mpu.get_virtual_pipeline_model_parallel_world_size()
    if vpp_size_value is not None and vpp_size_value > 1:
        model_to_check = model[0] if isinstance(model, Sequence) else model
        config = get_model_config(model_to_check)
        expected = config.microbatch_group_size_per_vp_stage
        actual = parallel_state.microbatch_group_size_per_vp_stage
        assert (
            actual == expected
        ), f"microbatch_group_size_per_vp_stage mismatch: ParallelState has {actual}, model config has {expected}"


def get_packed_seq_params(batch: dict[str, torch.Tensor], args: Namespace) -> PackedSeqParams:
    if args.qkv_format == "thd":
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=batch["cu_seqlens"],
            cu_seqlens_kv=batch["cu_seqlens"],
            max_seqlen_q=batch["max_seqlen"],
            max_seqlen_kv=batch["max_seqlen"],
            qkv_format="thd",
        )
        batch["packed_seq_params"] = packed_seq_params
        return packed_seq_params
    else:
        return None
