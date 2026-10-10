import logging
from argparse import Namespace
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_model_config
from megatron.training.global_vars import get_args

from miles.utils.ft_utils.process_group_utils import GroupInfo

from ..training_utils.parallel import ParallelState, get_parallel_state

logger = logging.getLogger(__name__)


def create_megatron_parallel_state(
    indep_dp: GroupInfo,
) -> ParallelState:
    vpp_size, microbatch_group_size_per_vp_stage = _compute_vpp_fields()
    args = get_args()
    tp_dp_cp_group = mpu.get_tensor_and_data_parallel_group(with_context_parallel=True)

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
        cp_comm_type=getattr(args, "cp_comm_type", None),
        tp=GroupInfo(
            rank=mpu.get_tensor_model_parallel_rank(),
            size=mpu.get_tensor_model_parallel_world_size(),
            group=mpu.get_tensor_model_parallel_group(),
        ),
        pp=GroupInfo(
            rank=mpu.get_pipeline_model_parallel_rank(),
            size=mpu.get_pipeline_model_parallel_world_size(),
            group=mpu.get_pipeline_model_parallel_group(),
        ),
        ep=GroupInfo(
            rank=mpu.get_expert_model_parallel_rank(),
            size=mpu.get_expert_model_parallel_world_size(),
            group=mpu.get_expert_model_parallel_group(),
        ),
        etp=GroupInfo(
            rank=mpu.get_expert_tensor_parallel_rank(),
            size=mpu.get_expert_tensor_parallel_world_size(),
            group=mpu.get_expert_tensor_parallel_group(),
        ),
        edp=GroupInfo(
            rank=mpu.get_expert_data_parallel_rank(),
            size=mpu.get_expert_data_parallel_world_size(),
            group=mpu.get_expert_data_parallel_group(),
        ),
        tp_dp_cp=GroupInfo(
            rank=torch.distributed.get_rank(tp_dp_cp_group),
            size=torch.distributed.get_world_size(tp_dp_cp_group),
            group=tp_dp_cp_group,
        ),
        indep_dp=indep_dp,
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


@dataclass
class PackedSeqParamsWithHostCuSeqlens(PackedSeqParams):
    """``PackedSeqParams`` plus the host copies of ``cu_seqlens_q`` that ``get_batch`` already built.

    ``cu_seqlens_host`` is the tuple of ints from ``get_batch``; ``cu_seqlens_cpu`` is the same
    boundaries as one CPU ``torch.int64`` tensor -- the host half of the KDA layers' boundary contract
    (device int32 ``cu_seqlens_q`` + ``cu_seqlens_cpu``; see ``kda_chunk_train.kda_backend``). It is
    built exactly once per micro-batch, here, and every KDA layer's forward, recompute forward and
    backward of that micro-batch reads this one object, so no layer copies the boundaries off the
    device and fla's identity-keyed ``tensor_cache`` hits for all of them.

    Under context parallelism the first KDA layer builds this rank's fla CP context once per
    micro-batch and keeps it, with the int64 host copy of the rank-local boundaries, in
    ``fla_cp_context`` / ``fla_cp_cu_seqlens_cpu`` for the later layers and the recompute forward.
    """

    cu_seqlens_host: tuple[int, ...] = field(kw_only=True)
    cu_seqlens_cpu: torch.Tensor | None = field(default=None, kw_only=True)
    fla_cp_context: object = field(default=None, kw_only=True, repr=False, compare=False)
    fla_cp_cu_seqlens_cpu: torch.Tensor | None = field(default=None, kw_only=True, repr=False, compare=False)

    def __post_init__(self):
        super().__post_init__()
        if self.cu_seqlens_cpu is None:
            self.cu_seqlens_cpu = torch.tensor(self.cu_seqlens_host, dtype=torch.int64)


def get_packed_seq_params(batch: dict[str, torch.Tensor], args: Namespace) -> PackedSeqParams:
    """One ``PackedSeqParamsWithHostCuSeqlens`` per micro-batch (``thd``), carrying the device
    ``cu_seqlens`` and its host copies; ``None`` for ``bshd``."""
    if args.qkv_format == "thd":
        packed_seq_params = PackedSeqParamsWithHostCuSeqlens(
            cu_seqlens_q=batch["cu_seqlens"],
            cu_seqlens_kv=batch["cu_seqlens"],
            max_seqlen_q=batch["max_seqlen"],
            max_seqlen_kv=batch["max_seqlen"],
            qkv_format="thd",
            cu_seqlens_host=batch["cu_seqlens_host"],
        )
        batch["packed_seq_params"] = packed_seq_params
        return packed_seq_params
    else:
        return None
