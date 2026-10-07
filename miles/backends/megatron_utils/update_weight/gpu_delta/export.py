"""PP-local ordinary ownership and targeted TP export for GPU delta."""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch
import torch.distributed as dist

from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import (
    HfWeightIteratorDirect,
    _check_and_fix_partition,
    _gather_megatron_expert_batch,
    _gather_with_stride,
    _pack_param_infos_by_size,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.types import ParamInfo


@dataclass(frozen=True)
class _OrdinaryBatch:
    param_infos: tuple[ParamInfo, ...]
    partitions: tuple[tuple[int, int] | None, ...]
    owner: int


def _unit_key(name: str) -> tuple[int, int, int]:
    # Put embedding/head last so the tail balances fewer layers with higher-precision tensors.
    if match := re.search(r"\.(decoder|mtp)\.layers\.(\d+)\.", name):
        # Names already contain global PP/VPP layer indices. Keep MTP's own
        # namespace separate; never infer missing layers or a tied LM head.
        return (0, match[1] == "mtp", int(match[2]))
    if ".embedding." in name:
        return (1, 0, 0)
    if ".output_layer." in name:
        return (2, 0, 0)
    return (3, 0, 0)


def _ordinary_owners(param_infos: Sequence[ParamInfo], ranks: Sequence[int]) -> dict[str, int]:
    units = {}
    for info in param_infos:
        units.setdefault(_unit_key(info.name), []).append(info.name)
    count, remainder = divmod(len(units), len(ranks))
    owners = [rank for index, rank in enumerate(ranks) for _ in range(count + (index < remainder))]
    return {name: owner for key, owner in zip(sorted(units), owners, strict=True) for name in sorted(units[key])}


def _ordinary_batches(args, param_infos, owners, tp_ranks):
    by_name = {info.name: info for info in param_infos}
    owned = {}
    for name, owner in owners.items():
        owned.setdefault(owner, []).append(by_name[name])
    batches = []
    for owner, infos in owned.items():
        if owner not in tp_ranks:
            continue
        for batch in _pack_param_infos_by_size(args, infos):
            partitions = []
            for info in batch:
                attrs = info.attrs
                sharded = (
                    len(tp_ranks) > 1
                    and attrs["tensor_model_parallel"]
                    and attrs["parallel_mode"] != "duplicated"
                    and "expert_bias" not in info.name
                )
                partitions.append(
                    _check_and_fix_partition(args, info.name, attrs["partition_stride"], attrs["partition_dim"])
                    if sharded
                    else None
                )
            batches.append(_OrdinaryBatch(tuple(batch), tuple(partitions), owner))
    return tuple(batches)


def _gather_batch(batch, weights, rank, tp_group, tp_size, device):
    """Only the owner's TP replica loads shards; only the owner allocates outputs."""
    pending, handles = [], []
    try:
        for info, partition in zip(batch.param_infos, batch.partitions, strict=True):
            if partition is None and rank != batch.owner:
                continue
            tensor = weights[info.name].detach().to(device=device, non_blocking=True)
            if partition is None:
                pending.append((info.name, tensor, None, None))
                continue
            outputs = [torch.empty_like(tensor) for _ in range(tp_size)] if rank == batch.owner else None
            handles.append(dist.gather(tensor, gather_list=outputs, dst=batch.owner, group=tp_group, async_op=True))
            # Retain every input until all of this batch's asynchronous reads finish.
            pending.append((info.name, tensor, outputs, partition))
    finally:
        error = None
        for handle in handles:
            try:
                handle.wait()
            except Exception as caught:
                error = error or caught
        if error is not None:
            raise error
    return pending


class HfWeightIteratorGpuDelta(HfWeightIteratorDirect):
    """Consume complete tensors on their fixed owners, independent of yield placement."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        parallel = get_parallel_state()
        assert not self.placement.gather_pp, "GPU delta publishes PP-local owners"
        tp_ranks = tuple(dist.get_process_group_ranks(parallel.tp.group))
        ranks = tuple(dist.get_process_group_ranks(parallel.tp_dp_cp.group))
        infos = [info for batch in self._non_expert_batches for info in batch]
        self.ordinary_owners = MappingProxyType(_ordinary_owners(infos, ranks))
        self._non_expert_batches = _ordinary_batches(self.args, infos, self.ordinary_owners, tp_ranks)
        self._rank = dist.get_rank()
        self._tp_group, self._tp_size = parallel.tp.group, parallel.tp.size
        self._device = torch.cuda.current_device()
        self.local_consumer: Callable[[list[tuple[str, torch.Tensor]]], None] | None = None
        self.local_error_consumer: Callable[[Exception], None] | None = None

    def _hf_atomic_update_groups(self):
        # The complete publication is applied together. Incremental load-call groups
        # must not require tensors already consumed by owner callbacks.
        return []

    def _convert_to_hf_param_units(self, named_params):
        for named_param in named_params:
            try:
                yield from super()._convert_to_hf_param_units([named_param])
            except Exception as error:
                self.local_error_consumer(error)

    def _iter_non_expert_batch(self, batch, weights, materialize):
        pending = _gather_batch(
            batch, weights, rank=self._rank, tp_group=self._tp_group, tp_size=self._tp_size, device=self._device
        )
        if self._rank == batch.owner:
            try:
                for name, tensor, outputs, partition in pending:
                    if partition is not None:
                        stride, dim = partition
                        tensor = _gather_with_stride(outputs, dim, stride)
                    for unit in self._convert_to_hf_param_units([(name, tensor)]):
                        self.local_consumer(unit)
            except Exception as error:
                # Gather handles have drained before owner-only reconstruction
                # or conversion can fail. Peers must still visit later batches.
                self.local_error_consumer(error)
        return ()

    def _iter_expert_batch(self, batch, weights, materialize):
        if not self._convert_experts_before_gather:
            # ETP>1 retains gather-before-convert; stream ordering replaces the
            # generic export's device barrier so compression can keep running.
            named_params = _gather_megatron_expert_batch(
                self.args, batch.param_infos, weights, gather_pp=False, synchronize=False
            )
            if materialize:
                yield from self._convert_to_hf_param_units(named_params)
            return
        for info in batch.param_infos:
            if info.src_rank == self._rank:
                param = weights[info.name].detach().to(device=self._device, non_blocking=True)
                for unit in self._convert_to_hf_param_units([(info.name, param)]):
                    self.local_consumer(unit)
