import logging
from collections.abc import Sequence
from types import SimpleNamespace

import torch
import torch.nn as nn
from megatron.core import tensor_parallel
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.optimizer import ChainedOptimizer
from megatron.core.transformer.utils import sharded_state_dict_default
from torch import Tensor

from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.audit_utils.event_logger.logger import get_event_logger
from miles.utils.audit_utils.event_logger.models import WitnessSnapshotParamEvent
from miles.utils.audit_utils.witness.allocator import WitnessInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def install_witness(
    model: nn.Module,
    *,
    buffer_size: int,
    sequence_parallel: bool = False,
) -> None:
    model.local_head_witness = _DataWitness(buffer_size=buffer_size, sequence_parallel=sequence_parallel)
    model.local_tail_witness = _DataWitness(buffer_size=buffer_size, sequence_parallel=sequence_parallel)


def witness_dump_and_clear_stale(
    *,
    model: Sequence[nn.Module],
    witness_info: WitnessInfo,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Log nonzero witness param rows, then clear stale ring buffer entries."""
    pp_rank = get_parallel_state().pp.rank

    for chunk_index, chunk in enumerate(model):
        inner = _unwrap_to_witness_owner(chunk)
        for attr in _WITNESS_ATTRS:
            assert hasattr(inner, attr), f"chunk {chunk_index} missing {attr}"
            witness: _DataWitness = getattr(inner, attr)
            _record_and_log_witness_param(
                witness=witness,
                instance_id=f"pp{pp_rank}_chunk{chunk_index}." + attr.replace("_witness", ""),
                stale_ids=witness_info.stale_ids,
            )

    _clear_witness_stale_rows(model=model, stale_ids=witness_info.stale_ids, optimizer=optimizer)


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------


class _DataWitness(nn.Module):
    def __init__(
        self,
        buffer_size: int,
        *,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.buffer_size = buffer_size
        self._sequence_parallel = sequence_parallel
        self.witness = nn.Embedding(num_embeddings=buffer_size, embedding_dim=1)
        self.witness.weight._is_witness_param = True
        nn.init.zeros_(self.witness.weight)

    def forward(self, witness_ids: Tensor, hidden_states: Tensor) -> Tensor:
        w = self.witness(witness_ids)  # [b, s, 1]
        out = w - w.detach()  # forward: bitwise 0, backward: d/dw = I

        out = out.transpose(0, 1).contiguous()  # [s, b, 1]
        if self._sequence_parallel:
            out = tensor_parallel.scatter_to_sequence_parallel_region(out)

        return _abs_broadcast_add(hidden_states, out)

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata: object = None) -> dict:
        pp_rank = get_parallel_state().pp.rank
        # Embed PP rank in the checkpoint key so each pipeline stage has a unique
        # key (e.g. local_head_witness_pp0.witness.weight vs _pp1.witness.weight).
        # Without this, PP>1 causes a sharding validation error because multiple
        # stages register the same key with identical replica_id.
        prefix_with_pp = f"{prefix.rstrip('.')}_pp{pp_rank}."

        # Delegate to Megatron's sharded_state_dict_default (utils.py).
        # Use SimpleNamespace so it takes the `else` branch (no sharded_state_dict attr)
        # instead of recursing back into this method.
        return sharded_state_dict_default(
            module=SimpleNamespace(state_dict=self.state_dict),
            prefix=prefix_with_pp,
            sharded_offsets=sharded_offsets,
            metadata=metadata,
            tp_group=get_parallel_state().tp.group,
        )


def _abs_broadcast_add(hidden_states: Tensor, addend: Tensor) -> Tensor:
    return _AbsBroadcastAdd.apply(hidden_states, addend)


class _AbsBroadcastAdd(torch.autograd.Function):
    """Broadcast-add a low-dim addend to a high-dim tensor, using abs-reduced gradient for the addend.

    Forward: ``hidden_states + addend`` (standard broadcast).
    Backward for ``hidden_states``: pass-through.
    Backward for ``addend``: ``grad.abs().sum(dim=-1, keepdim=True)`` instead of ``grad.sum(dim=-1, keepdim=True)``.

    This avoids gradient cancellation when the upstream gradient has mixed signs
    across the last dimension.  The witness embedding only needs to detect
    *whether* gradient flowed (nonzero), not the exact magnitude, so using
    ``abs`` is acceptable.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, hidden_states: Tensor, addend: Tensor) -> Tensor:
        assert addend.shape[-1] == 1, f"addend last dim must be 1, got {addend.shape}"
        assert hidden_states.shape[:-1] == addend.shape[:-1], (
            f"hidden_states and addend must match on all dims except last, "
            f"got {hidden_states.shape} vs {addend.shape}"
        )
        return hidden_states + addend

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad: Tensor) -> tuple[Tensor, Tensor]:
        grad_addend = grad.abs().sum(dim=-1, keepdim=True)
        return grad, grad_addend


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


_WITNESS_ATTRS = ("local_head_witness", "local_tail_witness")


def _has_any_witness(module: nn.Module) -> bool:
    return any(hasattr(module, attr) for attr in _WITNESS_ATTRS)


def _unwrap_to_witness_owner(chunk: nn.Module) -> nn.Module:
    """Navigate through wrapping layers (DDP → Float16Module → GPTModel) to find the module with witness attrs."""
    inner = chunk.module
    while not _has_any_witness(inner) and hasattr(inner, "module"):
        inner = inner.module
    return inner


def _clear_witness_stale_rows(
    *,
    model: Sequence[nn.Module],
    stale_ids: list[int],
    optimizer: torch.optim.Optimizer,
) -> None:
    if not stale_ids:
        return

    witnesses = list(_get_all_witnesses_in_model(model))
    for witness in witnesses:
        idx = torch.tensor(stale_ids, dtype=torch.long, device=witness.witness.weight.device)
        _zero_witness_rows(witness=witness, idx=idx, optimizer=optimizer)


def _get_all_witnesses_in_model(model_chunks: Sequence[nn.Module]) -> list[_DataWitness]:
    witnesses: list[_DataWitness] = []
    for chunk in model_chunks:
        inner = _unwrap_to_witness_owner(chunk)
        for attr in _WITNESS_ATTRS:
            assert hasattr(inner, attr), f"model chunk missing {attr}"
            witnesses.append(getattr(inner, attr))
    return witnesses


def _zero_witness_rows(*, witness: _DataWitness, idx: Tensor, optimizer: torch.optim.Optimizer) -> None:
    model_weight = witness.witness.weight
    model_weight.data[idx] = 0.0

    for inner_optimizer in _iter_inner_optimizers(optimizer):
        # miles forces use_distributed_optimizer, so anything else is unreachable.
        assert isinstance(
            inner_optimizer, DistributedOptimizer
        ), f"unsupported optimizer: {type(inner_optimizer).__name__}"
        _zero_rows_in_distributed_optimizer(optimizer=inner_optimizer, model_param=model_weight, idx=idx)


def _iter_inner_optimizers(optimizer: torch.optim.Optimizer) -> list[torch.optim.Optimizer]:
    if isinstance(optimizer, ChainedOptimizer):
        return list(optimizer.chained_optimizers)
    return [optimizer]


def _zero_rows_in_distributed_optimizer(*, optimizer: DistributedOptimizer, model_param: Tensor, idx: Tensor) -> None:
    assert not optimizer.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    assert not optimizer.config.optimizer_cpu_offload, "HybridDeviceOptimizer state layout is not supported"
    assert optimizer.config.optimizer == "adam", f"unsupported optimizer kernel: {optimizer.config.optimizer}"
    if model_param not in optimizer.model_param_gbuf_map:
        # This dist-opt instance (e.g. the expert one) or this rank owns no shard of the param.
        return

    # The fp32 main weights are flat shards of the flattened model param;
    # embedding_dim == 1 makes flattened offsets equal witness row ids.
    assert model_param.shape[-1] == 1, f"witness weight last dim must be 1, got {model_param.shape}"
    param_range = optimizer._get_model_param_range_map(model_param)["param"]
    local_idx = idx[(idx >= param_range.start) & (idx < param_range.end)] - param_range.start
    if local_idx.numel() == 0:
        return

    group_index, group_order = optimizer.model_param_group_index_map[model_param]
    main_param = optimizer.optimizer.param_groups[group_index]["params"][group_order]
    assert main_param.numel() == param_range.size
    main_param.data[local_idx] = 0.0

    state = optimizer.optimizer.state
    if main_param not in state:
        # An optimizer that never stepped has no per-param state yet; a populated state
        # missing the witness entry means we are clearing the wrong key — fail loudly.
        assert len(state) == 0, f"witness main shard missing from optimizer state with {len(state)} entries"
        return

    param_state = state[main_param]
    for key in ("exp_avg", "exp_avg_sq"):
        assert key in param_state, f"expected Adam state key {key!r}, got {sorted(param_state)}"
        param_state[key][local_idx] = 0.0


def _record_and_log_witness_param(
    *,
    witness: _DataWitness,
    instance_id: str,
    stale_ids: list[int],
) -> None:
    model_weight = witness.witness.weight
    main_param = getattr(model_weight, "main_param", None)
    check_weight = main_param.data if main_param is not None else model_weight.data
    nonzero_witness_ids: list[int] = check_weight.squeeze(-1).nonzero(as_tuple=True)[0].tolist()

    get_event_logger().log(
        WitnessSnapshotParamEvent,
        dict(
            instance_id=instance_id,
            nonzero_witness_ids=nonzero_witness_ids,
            stale_ids=stale_ids,
        ),
        print_log=False,
    )
