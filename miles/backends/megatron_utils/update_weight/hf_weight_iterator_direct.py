import dataclasses
from argparse import Namespace
from collections.abc import Sequence

import torch
import torch.distributed as dist
from tqdm import tqdm

from miles.backends.megatron_utils.update_weight.hf_weight_iterator import MegatronHfWeightIteratorBase
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.types import ParamInfo

from ..megatron_to_hf import convert_to_hf
from ..sglang import monkey_patch_torch_reductions
from .common import all_gather_params_async, is_routed_expert_param, named_params_and_buffers


class HfWeightIteratorDirect(MegatronHfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.megatron_local_param_info_buckets = _get_megatron_local_param_info_buckets(self.args, self.model)

    def _iter_hf_param_units(self, weights, *, materialize):
        assert materialize, "non-materializing iteration lands with the distributed-path migration"
        rank = dist.get_rank()

        for megatron_local_param_infos in tqdm(
            self.megatron_local_param_info_buckets, disable=rank != 0, desc="Update weights"
        ):
            megatron_full_params = _get_megatron_full_params(self.args, megatron_local_param_infos, weights)
            yield from self._convert_to_hf_param_units(megatron_local_param_infos, megatron_full_params)
            del megatron_full_params

    def _export_pp_local_lora(self, adapter):
        assert adapter is None, "multi-LoRA export requires --megatron-to-hf-mode bridge"
        from miles_plugins.models.inkling.lora import export_inkling_lora_hf_named

        return export_inkling_lora_hf_named(self.model)

    def _convert_to_hf_param_units(self, param_infos: Sequence[ParamInfo], params: Sequence[torch.Tensor]):
        for info, param in zip(param_infos, params, strict=True):
            yield list(convert_to_hf(self.args, self.model_name, info.name, param, self.quantization_config))


def _get_megatron_full_params(
    args: Namespace,
    megatron_local_param_infos: Sequence[ParamInfo],
    megatron_local_weights,
) -> Sequence[torch.Tensor]:
    monkey_patch_torch_reductions()
    pp_size = get_parallel_state().pp.size
    ep_size = get_parallel_state().ep.size
    rank = dist.get_rank()
    # init params:
    params = []
    for info in megatron_local_param_infos:
        if dist.get_rank() == info.src_rank:
            params.append(
                torch.nn.Parameter(
                    megatron_local_weights[info.name].to(device=torch.cuda.current_device(), non_blocking=True),
                    requires_grad=False,
                )
            )
        else:
            params.append(torch.empty(info.shape, dtype=info.dtype, device=torch.cuda.current_device()))
    torch.cuda.synchronize()

    # broadcast params across pp ranks
    if pp_size > 1:
        handles = []
        for info, param in zip(megatron_local_param_infos, params, strict=False):
            if info.src_rank in dist.get_process_group_ranks(get_parallel_state().pp.group):
                handles.append(
                    torch.distributed.broadcast(
                        param, src=info.src_rank, group=get_parallel_state().pp.group, async_op=True
                    )
                )
        for handle in handles:
            handle.wait()

    # broadcast params across ep ranks
    if ep_size > 1:
        handles = []
        for info, param in zip(megatron_local_param_infos, params, strict=False):
            if is_routed_expert_param(info.name):
                src_rank = (
                    info.src_rank
                    if info.src_rank in dist.get_process_group_ranks(get_parallel_state().ep.group)
                    else rank
                )
                handles.append(
                    torch.distributed.broadcast(
                        param, src=src_rank, group=get_parallel_state().ep.group, async_op=True
                    )
                )
        for handle in handles:
            handle.wait()

    # Set tp attrs for all params
    for info, param in zip(megatron_local_param_infos, params, strict=False):
        for key, value in info.attrs.items():
            setattr(param, key, value)

    # Batch async all_gather for all parameters
    gathered_params = all_gather_params_async(args, list(zip(megatron_local_param_infos, params, strict=False)))

    return gathered_params


def _get_megatron_local_param_info_buckets(args: Namespace, model: Sequence[torch.nn.Module]) -> list[list[ParamInfo]]:
    """Partition params into gather batches ≤ update_weight_buffer_size."""
    param_infos = _get_megatron_local_param_infos(args, model)
    return _pack_param_infos_by_size(args, param_infos)


def _get_param_full_size(info: ParamInfo) -> int:
    if is_routed_expert_param(info.name):
        tp_size = get_parallel_state().etp.size
    else:
        tp_size = get_parallel_state().tp.size
    return info.size * tp_size


def _pack_param_infos_by_size(args: Namespace, param_infos: list[ParamInfo]) -> list[list[ParamInfo]]:
    """Greedy size packing into gather batches ≤ update_weight_buffer_size."""
    batches: list[list[ParamInfo]] = [[]]
    buffer_size = 0
    for info in param_infos:
        size = _get_param_full_size(info)
        if buffer_size + size > args.update_weight_buffer_size and batches[-1]:
            batches.append([])
            buffer_size = 0
        batches[-1].append(info)
        buffer_size += size
    return [batch for batch in batches if batch]


def _get_megatron_local_param_infos(args: Namespace, model: Sequence[torch.nn.Module]) -> list[ParamInfo]:
    """Collect param metadata, exchanged across PP and EP; identical on every rank."""
    pp_size = get_parallel_state().pp.size
    ep_size = get_parallel_state().ep.size

    from ..lora_utils import _is_adapter_param_name

    param_infos = {}
    rank = dist.get_rank()
    for name, param in named_params_and_buffers(args, model):
        if _is_adapter_param_name(name):
            continue
        param_infos[name] = ParamInfo(
            name=name,
            dtype=param.dtype,
            shape=param.shape,
            attrs={
                "tensor_model_parallel": getattr(param, "tensor_model_parallel", False),
                "partition_dim": getattr(param, "partition_dim", -1),
                "partition_stride": getattr(param, "partition_stride", 1),
                "parallel_mode": getattr(param, "parallel_mode", None),
            },
            size=param.numel() * param.element_size(),
            src_rank=rank,
        )

    if pp_size > 1:
        param_infos_list = [None] * pp_size
        dist.all_gather_object(
            obj=(rank, param_infos), object_list=param_infos_list, group=get_parallel_state().pp.group
        )
        for src_rank, infos in param_infos_list:
            if src_rank == rank:
                continue
            for name, info in infos.items():
                if name in param_infos:
                    # Duplicates across PP only exist for MTP virtual-PP layers.
                    assert args.mtp_num_layers is not None
                    if param_infos[name].src_rank > src_rank:
                        param_infos[name] = info
                else:
                    param_infos[name] = info

    if ep_size > 1:
        param_infos_list = [None] * ep_size
        dist.all_gather_object(
            obj=(rank, param_infos), object_list=param_infos_list, group=get_parallel_state().ep.group
        )
        for src_rank, infos in param_infos_list:
            for name, info in infos.items():
                if name not in param_infos:
                    # src_rank must be the rank within the expert model parallel group
                    info = dataclasses.replace(info, src_rank=src_rank)
                    param_infos[name] = info

    param_infos = sorted(param_infos.values(), key=lambda info: info.name)
    _check_param_infos_consistent(param_infos)
    return param_infos


def _check_param_infos_consistent(param_infos: list[ParamInfo]) -> None:
    """Every rank must hold identical param metadata."""
    all_param_info_list = [None] * dist.get_world_size()
    dist.all_gather_object(obj=param_infos, object_list=all_param_info_list, group=get_gloo_group())
    for i, param_info in enumerate(param_infos):
        for infos in all_param_info_list:
            assert infos[i].name == param_info.name, f"Parameter name mismatch: {infos[i].name} != {param_info.name}"
            assert (
                infos[i].shape == param_info.shape
            ), f"Parameter shape mismatch: {infos[i].shape} != {param_info.shape}"
            assert (
                infos[i].dtype == param_info.dtype
            ), f"Parameter dtype mismatch: {infos[i].dtype} != {param_info.dtype}"
