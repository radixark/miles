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
        non_expert_infos, expert_infos = _get_megatron_local_param_infos(self.args, self.model)
        ep_size = get_parallel_state().ep.size
        self._non_expert_batches = _pack_param_infos_by_size(self.args, non_expert_infos)
        # An expert batch materializes ep_size x its metadata size after the EP all_gather.
        self._expert_batches = _pack_param_infos_by_size(self.args, expert_infos, size_multiplier=ep_size)

    def _iter_hf_param_units(self, weights, *, materialize):
        assert materialize, "non-materializing iteration lands with the distributed-path migration"
        rank = dist.get_rank()

        pbar = tqdm(
            total=len(self._non_expert_batches) + len(self._expert_batches),
            disable=rank != 0,
            desc="Update weights",
        )
        for param_infos in self._non_expert_batches:
            named_params = _materialize_non_expert_batch(self.args, param_infos, weights)
            yield from self._convert_to_hf_param_units(named_params)
            del named_params
            pbar.update(1)
        for param_infos in self._expert_batches:
            named_params = _materialize_expert_batch(self.args, param_infos, weights)
            yield from self._convert_to_hf_param_units(named_params)
            del named_params
            pbar.update(1)
        pbar.close()

    def _export_pp_local_lora(self, adapter):
        assert adapter is None, "multi-LoRA export requires --megatron-to-hf-mode bridge"
        from miles_plugins.models.inkling.lora import export_inkling_lora_hf_named

        return export_inkling_lora_hf_named(self.model)

    def _convert_to_hf_param_units(self, named_params: Sequence[tuple[str, torch.Tensor]]):
        for name, param in named_params:
            yield list(convert_to_hf(self.args, self.model_name, name, param, self.quantization_config))


def _load_or_allocate_params(param_infos: Sequence[ParamInfo], megatron_local_weights) -> list[torch.Tensor]:
    """Owners load from the weight source; other ranks allocate receive buffers."""
    params = []
    for info in param_infos:
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
    return params


def _broadcast_across_pp(param_infos: Sequence[ParamInfo], params: Sequence[torch.Tensor]) -> None:
    pp = get_parallel_state().pp
    if pp.size == 1:
        return
    pp_ranks = dist.get_process_group_ranks(pp.group)
    handles = []
    for info, param in zip(param_infos, params, strict=True):
        if info.src_rank in pp_ranks:
            handles.append(dist.broadcast(param, src=info.src_rank, group=pp.group, async_op=True))
    for handle in handles:
        handle.wait()


def _set_tp_attrs(param_infos: Sequence[ParamInfo], params: Sequence[torch.Tensor]) -> None:
    for info, param in zip(param_infos, params, strict=True):
        for key, value in info.attrs.items():
            setattr(param, key, value)


def _materialize_non_expert_batch(
    args: Namespace,
    param_infos: Sequence[ParamInfo],
    megatron_local_weights,
) -> list[tuple[str, torch.Tensor]]:
    """Load -> PP broadcast -> TP all_gather."""
    monkey_patch_torch_reductions()
    params = _load_or_allocate_params(param_infos, megatron_local_weights)
    _broadcast_across_pp(param_infos, params)
    _set_tp_attrs(param_infos, params)
    gathered = all_gather_params_async(args, list(zip(param_infos, params, strict=True)))
    return [(info.name, param) for info, param in zip(param_infos, gathered, strict=True)]


def _materialize_expert_batch(
    args: Namespace,
    param_infos: Sequence[ParamInfo],
    megatron_local_weights,
) -> list[tuple[str, torch.Tensor]]:
    """Load -> PP broadcast -> ETP all_gather -> EP all_gather.

    Expert metadata is EP-local; the full expert set is materialized by a
    symmetric EP all_gather with a name exchange.
    """
    monkey_patch_torch_reductions()
    params = _load_or_allocate_params(param_infos, megatron_local_weights)
    _broadcast_across_pp(param_infos, params)
    _set_tp_attrs(param_infos, params)
    etp_gathered = all_gather_params_async(args, list(zip(param_infos, params, strict=True)))

    ep = get_parallel_state().ep
    if ep.size == 1:
        return [(info.name, param) for info, param in zip(param_infos, etp_gathered, strict=True)]

    names = [info.name for info in param_infos]
    all_names: list = [None] * ep.size
    dist.all_gather_object(all_names, names, group=ep.group)
    for ep_names in all_names:
        assert len(ep_names) == len(
            names
        ), f"EP-asymmetric expert batch: {len(names)} params locally vs {len(ep_names)} on a peer rank"

    all_gathered: list[list[tuple[str, torch.Tensor]]] = [[] for _ in range(ep.size)]
    handles = []
    for i, param in enumerate(etp_gathered):
        buffers = [torch.empty_like(param, device=torch.cuda.current_device()) for _ in range(ep.size)]
        handles.append(dist.all_gather(buffers, param, group=ep.group, async_op=True))
        for ep_rank, ep_names in enumerate(all_names):
            all_gathered[ep_rank].append((ep_names[i], buffers[ep_rank]))
    for handle in handles:
        handle.wait()

    return [named for per_rank in all_gathered for named in per_rank]


def _pack_param_infos_by_size(
    args: Namespace, param_infos: list[ParamInfo], *, size_multiplier: int = 1
) -> list[list[ParamInfo]]:
    """Greedy size packing into gather batches ≤ update_weight_buffer_size."""
    batches: list[list[ParamInfo]] = [[]]
    buffer_size = 0
    for info in param_infos:
        size = _get_param_full_size(info) * size_multiplier
        if buffer_size + size > args.update_weight_buffer_size and batches[-1]:
            batches.append([])
            buffer_size = 0
        batches[-1].append(info)
        buffer_size += size
    return [batch for batch in batches if batch]


def _get_param_full_size(info: ParamInfo) -> int:
    if is_routed_expert_param(info.name):
        tp_size = get_parallel_state().etp.size
    else:
        tp_size = get_parallel_state().tp.size
    return info.size * tp_size


def _get_megatron_local_param_infos(
    args: Namespace, model: Sequence[torch.nn.Module]
) -> tuple[list[ParamInfo], list[ParamInfo]]:
    """Collect param metadata, exchanged across PP.

    Returns (non_expert_infos, expert_infos); expert infos stay EP-local.
    """
    pp_size = get_parallel_state().pp.size

    from ..lora_utils import _is_adapter_param_name

    param_infos: dict[str, ParamInfo] = {}
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

    infos = sorted(param_infos.values(), key=lambda info: info.name)
    non_expert_infos = [info for info in infos if not is_routed_expert_param(info.name)]
    expert_infos = [info for info in infos if is_routed_expert_param(info.name)]

    _check_param_infos_consistent(non_expert_infos)

    return non_expert_infos, expert_infos


def _check_param_infos_consistent(param_infos: list[ParamInfo]) -> None:
    """Every rank must hold identical non-expert metadata once PP is gathered."""
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
