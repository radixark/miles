import itertools
import re
from argparse import Namespace
from collections.abc import Sequence

import torch
import torch.distributed as dist
from tqdm import tqdm

from miles.backends.megatron_utils.megatron_to_hf import convert_to_hf
from miles.backends.megatron_utils.named_weights import named_params_and_buffers
from miles.backends.megatron_utils.sglang import monkey_patch_torch_reductions
from miles.backends.megatron_utils.update_weight.expert_quantization import gather_expert_units
from miles.backends.megatron_utils.update_weight.hf_weight_iterator import (
    MegatronHfWeightIteratorBase,
    _iter_mm_tower_units,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.types import ParamInfo


class HfWeightIteratorDirect(MegatronHfWeightIteratorBase):
    # TP/EP are always gathered; PP follows the requirement. Routed experts require ETP1.
    forced_placement = WeightUpdatePlacement(gather_pp=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        parallel = get_parallel_state()
        if self.args.num_experts and parallel.etp.size != 1:
            raise ValueError(
                "Direct MoE weight updates require --expert-tensor-parallel-size 1 "
                "so each EP rank owns complete routed experts."
            )
        non_expert_infos, expert_infos = _get_megatron_local_param_infos(
            self.args, self.model, gather_pp=self.placement.gather_pp
        )
        ep_size = parallel.ep.size
        self._non_expert_batches = _pack_param_infos_by_size(self.args, non_expert_infos)
        self._expert_batches = []
        if expert_infos:
            edp = parallel.edp
            assert edp is not None, "Expert data parallel state is required for MoE weight updates"
            self._expert_dp = edp
            owner_infos = _partition_expert_infos(
                expert_infos, num_local_experts=self.args.num_experts // ep_size, edp_size=edp.size
            )
            # Pack each owner's share first so a round can quantize on every EDP
            # replica, even when the complete local expert set spans many batches.
            owner_batches = [
                _pack_param_infos_by_size(self.args, infos, size_multiplier=ep_size * edp.size)
                for infos in owner_infos
            ]
            self._expert_batches = [
                batches[edp.rank] for batches in itertools.zip_longest(*owner_batches, fillvalue=())
            ]

    def _iter_hf_param_units(self, weights, *, materialize):
        rank = dist.get_rank()

        pbar = tqdm(
            total=len(self._non_expert_batches) + len(self._expert_batches),
            disable=rank != 0,
            desc="Update weights",
        )
        for param_infos in self._non_expert_batches:
            named_params = _materialize_non_expert_batch(
                self.args, param_infos, weights, gather_pp=self.placement.gather_pp
            )
            if materialize:
                yield from self._convert_to_hf_param_units(named_params)
            del named_params
            pbar.update(1)
        for param_infos in self._expert_batches:
            units = self._materialize_expert_batch(param_infos, weights)
            if materialize:
                yield from units
            del units
            pbar.update(1)
        pbar.close()
        yield from _iter_mm_tower_units(self.args, materialize=materialize)

    def _materialize_expert_batch(self, param_infos, weights):
        """Convert once per expert across EP/EDP, then gather HF weights and scales."""
        parallel = get_parallel_state()
        device = torch.device("cuda", torch.cuda.current_device())
        # Sender placement is independent of ownership: non-senders also
        # quantize their assigned experts, once across all expert-DP replicas.
        local_params = (
            (info.name, weights[info.name].detach().to(device=device, non_blocking=True))
            for info in param_infos
            if info.src_rank == dist.get_rank()
        )
        units = list(self._convert_to_hf_param_units(local_params))
        if self.placement.gather_pp:
            units = gather_expert_units(units, group=parallel.pp.group, device=device)
        units = gather_expert_units(units, group=parallel.ep.group, device=device)
        return gather_expert_units(units, group=self._expert_dp.group, device=device)

    def _export_pp_local_lora(self, adapter):
        assert adapter is None, "multi-LoRA export requires --megatron-to-hf-mode bridge"
        # TODO: will rewrite in native lora refactor
        if "kimi_k3" in self.model_name.lower():
            from miles_plugins.models.kimi_k3.lora import export_kimi_k3_lora_hf_chunks

            return [named_tensor for chunk in export_kimi_k3_lora_hf_chunks(self.model) for named_tensor in chunk]
        if "inkling" in (self.args.custom_model_provider_path or ""):
            from miles_plugins.models.inkling.lora import export_inkling_lora_hf_named

            return export_inkling_lora_hf_named(self.model)
        raise NotImplementedError(f"Raw LoRA export is not implemented for model {self.model_name!r}")

    def _convert_to_hf_param_units(self, named_params: Sequence[tuple[str, torch.Tensor]]):
        for name, param in named_params:
            yield list(
                convert_to_hf(
                    self.args, self.model_name, name, param, self.quantization_config, self.packed_weight_basenames
                )
            )


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
    *,
    gather_pp: bool,
) -> list[tuple[str, torch.Tensor]]:
    """Load -> PP broadcast (when gather_pp) -> TP all_gather."""
    monkey_patch_torch_reductions()
    params = _load_or_allocate_params(param_infos, megatron_local_weights)
    if gather_pp:
        _broadcast_across_pp(param_infos, params)
    _set_tp_attrs(param_infos, params)
    gathered = all_gather_params_async(args, list(zip(param_infos, params, strict=True)))
    return [(info.name, param) for info, param in zip(param_infos, gathered, strict=True)]


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


def _partition_expert_infos(
    param_infos: Sequence[ParamInfo], *, num_local_experts: int, edp_size: int
) -> list[list[ParamInfo]]:
    """Assign contiguous local-expert ranges to EDP replicas, keeping FC1/FC2 together."""
    count, remainder = divmod(num_local_experts, edp_size)
    owners = [owner for owner in range(edp_size) for _ in range(count + (owner < remainder))]
    partitions: list[list[ParamInfo]] = [[] for _ in range(edp_size)]
    for info in param_infos:
        # The direct converter uses TE grouped weights with global expert IDs.
        match = re.search(r"\.weight(\d+)$", info.name)
        if match is None:
            raise ValueError(f"Cannot identify the routed expert in {info.name!r}")
        local_expert = int(match.group(1)) % num_local_experts
        partitions[owners[local_expert]].append(info)
    return partitions


def _get_param_full_size(info: ParamInfo) -> int:
    return info.size if is_routed_expert_param(info.name) else info.size * get_parallel_state().tp.size


def _get_megatron_local_param_infos(
    args: Namespace, model: Sequence[torch.nn.Module], *, gather_pp: bool
) -> tuple[list[ParamInfo], list[ParamInfo]]:
    """Collect param metadata, exchanged across PP when gather_pp.

    Returns (non_expert_infos, expert_infos); expert infos stay EP-local.
    """
    pp_size = get_parallel_state().pp.size

    from ..lora.utils import _is_adapter_param_name

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

    if gather_pp and pp_size > 1:
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

    if gather_pp:
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


def _gather_with_stride(
    param_partitions: list[torch.Tensor], partition_dim: int, partition_stride: int
) -> torch.Tensor:
    """Gather partitions respecting partition_stride (strided/interleaved TP sharding)."""
    if partition_stride == 1:
        return torch.cat(param_partitions, dim=partition_dim)
    # Interleaved (strided) partitioning, e.g. linear_fc1.weight under GLU/SwiGLU
    chunks_per_rank = [p.chunk(partition_stride, dim=partition_dim) for p in param_partitions]
    interleaved = [chunks_per_rank[r][s] for s in range(partition_stride) for r in range(len(param_partitions))]
    return torch.cat(interleaved, dim=partition_dim)


def is_routed_expert_param(name: str) -> bool:
    """Whether a Megatron param name belongs to the routed (expert-parallel) experts.

    Routed experts live under ".experts.", but shared experts may nest an inner
    ModuleList (e.g. Inkling's "mlp.shared_experts.experts.N.") whose params are
    regular-TP sharded and EP-replicated, so they must not match.
    """
    return ".experts." in name and ".shared_experts." not in name


def _check_and_fix_partition(args: Namespace, name: str, partition_stride: int, partition_dim: int) -> tuple[int, int]:
    """Validate partition_stride values for known parameter patterns.

    After Megatron-LM PR #2708, linear_fc1 correctly reports partition_stride=2
    (GLU/SwiGLU interleaved [gate, up]), so assert partition_stride==2 is removed.
    But TEGroupedLinear still does not set partition_stride/partition_dim correctly for grouped moe gemm
    """
    if "linear_fc1.weight" in name and args.swiglu:
        partition_stride = 2
        if partition_dim < 0:
            partition_dim = 0
    elif "linear_fc2.weight" in name:
        assert partition_stride == 1, f"Expected partition_stride=1 for {name}, got {partition_stride}"
        if partition_dim <= 0:
            partition_dim = 1
    else:
        assert partition_stride == 1, f"Expected partition_stride=1 for {name}, got {partition_stride}"
    return partition_stride, partition_dim


def all_gather_params_async(
    args: Namespace,
    param_infos_and_params: list[tuple[ParamInfo, torch.Tensor]],
) -> list[torch.Tensor]:
    """
    Gather nonexpert TP shards, including shared experts. Launch each all-gather,
    wait for all handles, then concatenate partitions with any GLU rechunking.
    """
    # Phase 1: Start all async all_gather operations
    gather_tasks = []
    handles = []

    for info, param in param_infos_and_params:
        # Prepare async all_gather
        if "expert_bias" in info.name:
            gather_tasks.append((info, param, None, None, None, None))
            handles.append(None)
        elif getattr(param, "parallel_mode", None) == "duplicated" or not param.tensor_model_parallel:
            gather_tasks.append((info, param.data, None, None, None, None))
            handles.append(None)
        else:
            # Start async all_gather
            tp_size = get_parallel_state().tp.size
            tp_group = get_parallel_state().tp.group

            if tp_size <= 1:
                gather_tasks.append((info, param.data, None, None, None, None))
                handles.append(None)
                continue

            param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
            handle = dist.all_gather(param_partitions, param.data, group=tp_group, async_op=True)
            gather_tasks.append((info, None, handle, param_partitions, param.partition_dim, param.partition_stride))
            handles.append(handle)

    # Phase 2: Wait for ALL async operations to complete at once
    # This ensures maximum parallelism by not blocking on individual operations
    for handle in handles:
        if handle is not None:
            handle.wait()

    # Phase 3: Process all results after all communications are done
    gathered_params = []
    for info, direct_param, handle, param_partitions, partition_dim, partition_stride in gather_tasks:
        if handle is None:
            # No all_gather needed
            param = direct_param
        else:
            partition_stride, partition_dim = _check_and_fix_partition(
                args, info.name, partition_stride, partition_dim
            )
            param = _gather_with_stride(param_partitions, partition_dim, partition_stride)

        gathered_params.append(param)

    return gathered_params
