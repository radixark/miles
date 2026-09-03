import inspect
import logging
import re
from argparse import Namespace
from collections.abc import Iterator, Mapping, Sequence

import ray
import torch
import torch.distributed as dist
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from ray.actor import ActorHandle

from miles.backends.megatron_utils.misc_utils import strip_param_name_prefix
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.types import ParamInfo

logger = logging.getLogger(__name__)


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


def _is_unmarked_grouped_expert_weight(name: str, param: torch.nn.Parameter) -> bool:
    """TEGroupedLinear never marks its per-expert weight0..weightN, so Megatron fills in
    the defaults (tensor_model_parallel=False, partition_dim=-1) and the tensor claims to
    be unsharded. It is expert-TP sharded whenever etp > 1, so the gather must still run.
    """
    return (
        is_routed_expert_param(name)
        and ("linear_fc1.weight" in name or "linear_fc2.weight" in name)
        and not param.tensor_model_parallel
        and get_parallel_state().etp.size > 1
    )


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
    Parallel TP all-gather for multiple params. Loop 1: for each TP param, allocate buffers +
    dist.all_gather(async_op=True) on expert-TP/regular-TP group (skip expert_bias/non-TP/duplicated).
    Loop 2: wait all NCCL handles (enables overlap). Loop 3: concat partitions + apply GLU rechunk/MoE dim fix.
    """
    # Phase 1: Start all async all_gather operations
    gather_tasks = []
    handles = []

    for info, param in param_infos_and_params:
        # Prepare async all_gather
        if "expert_bias" in info.name:
            gather_tasks.append((info, param, None, None, None, None))
            handles.append(None)
        elif getattr(param, "parallel_mode", None) == "duplicated" or (
            not param.tensor_model_parallel and not _is_unmarked_grouped_expert_weight(info.name, param)
        ):
            gather_tasks.append((info, param.data, None, None, None, None))
            handles.append(None)
        else:
            # Start async all_gather
            if is_routed_expert_param(info.name):
                tp_size = get_parallel_state().etp.size
                tp_group = get_parallel_state().etp.group
            else:
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


def named_params_and_buffers(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    convert_to_global_name: bool = True,
    translate_gpu_to_cpu: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    if convert_to_global_name:
        ans = _named_params_and_buffers_global(args, model)
    else:
        ans = _named_params_and_buffers_vanilla(model)

    if translate_gpu_to_cpu:
        ans = ((name, _maybe_get_cpu_backup(tensor)) for name, tensor in ans)

    return ans


def _maybe_get_cpu_backup(x: torch.Tensor):
    from torch_memory_saver import torch_memory_saver

    if (cpu_tensor := torch_memory_saver.get_cpu_backup(x, zero_copy=True)) is not None:
        return cpu_tensor

    return x


def _named_params_and_buffers_vanilla(model: Sequence[torch.nn.Module]) -> Iterator[tuple[str, torch.Tensor]]:
    for vp_stage, model_module in enumerate(model):

        def _compute_fqn(name, vp_stage=vp_stage):
            return f"vp_stages.{vp_stage}.{strip_param_name_prefix(name)}"

        for name, param in model_module.named_parameters():
            if getattr(param, "_is_witness_param", False):
                continue
            yield _compute_fqn(name), param

        for name, buffer in model_module.named_buffers():
            # TODO shall we handle (almost) all buffers like Megatron Bridge
            if "expert_bias" not in name:
                continue
            yield _compute_fqn(name), buffer


def _named_params_and_buffers_global(
    args: Namespace, model: Sequence[torch.nn.Module]
) -> Iterator[tuple[str, torch.Tensor]]:
    """
    Yield (global_name, param/buffer) with consistent names across PP/EP. Adjusts indices for
    virtual PP + EP offsets. Handles decoder.layers, mtp.layers (Multi-Token Prediction), expert_bias.
    """
    ep_size = get_parallel_state().ep.size
    ep_rank = get_parallel_state().ep.rank
    if args.num_experts:
        expert_offset = ep_rank * args.num_experts // ep_size

    sig = inspect.signature(get_transformer_layer_offset)
    need_vp_stage = "vp_stage" in sig.parameters

    for vp_stage, model_module in enumerate(model):
        if need_vp_stage:
            layer_offset = get_transformer_layer_offset(model_module.config, vp_stage)
        else:
            layer_offset = get_transformer_layer_offset(model_module.config)
        for name, param in model_module.named_parameters():
            if getattr(param, "_is_witness_param", False):
                continue
            # for model without ddp wrap
            if not name.startswith("module.module."):
                name = "module." + name

            decoder_layers_pattern = r"module\.module\.(?:language_model\.)?decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                # MTP (Multi-Token Prediction) layers for speculative decoding
                mtp_layers_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
                match = re.match(mtp_layers_pattern, name)
                if not match:
                    yield name, param
                    continue

                # MTP layer indices start from 0
                layer_idx, rest = match.groups()
                expert_pattern = r"(transformer_layer|mtp_model_layer)\.mlp\.experts\.(.+)\.weight(\d+)"
                match = re.match(expert_pattern, rest)
                if not match:
                    yield name, param
                    continue

                inner, rest, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield (
                    f"module.module.mtp.layers.{layer_idx}.{inner}.mlp.experts.{rest}.weight{expert_idx}",
                    param,
                )
                continue

            layer_idx, rest = match.groups()
            layer_idx = int(layer_idx) + layer_offset

            # this is hardcoded for te grouped matmul
            expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
            match = re.match(expert_pattern, rest)
            if match:
                rest, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield f"module.module.decoder.layers.{layer_idx}.mlp.experts.{rest}.weight{expert_idx}", param
            else:
                yield f"module.module.decoder.layers.{layer_idx}.{rest}", param

        # treat expert bias as normal parameters
        for name, buffer in model_module.named_buffers():
            # TODO shall we handle (almost) all buffers like Megatron Bridge
            if "expert_bias" not in name:
                continue
            # for model without ddp wrap
            if not name.startswith("module.module."):
                name = "module." + name

            decoder_layers_pattern = r"module\.module\.(?:language_model\.)?decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                yield name, buffer
            else:
                layer_idx, rest = match.groups()
                layer_idx = int(layer_idx) + layer_offset
                yield f"module.module.decoder.layers.{layer_idx}.{rest}", buffer


def begin_weight_update(rollout_engines: Sequence[ActorHandle], selector: str = "all"):
    """Open a weight-update session on the selected rollout engines (restore packed weights)."""
    ray.get([engine.begin_weight_update.remote(selector=selector) for engine in rollout_engines])


def weight_update_selector(args) -> str:
    """Exclude the draft only when the trainer provably has no MTP block to send it."""
    if (
        getattr(args, "sglang_speculative_algorithm", None)
        and not getattr(args, "mtp_num_layers", None)
        and getattr(args, "megatron_to_hf_mode", "raw") != "bridge"
    ):
        return "target"
    return "all"


def end_weight_update(rollout_engines: Sequence[ActorHandle]):
    """Close the weight-update session (post-load + quant post-process on the full model)."""
    ray.get([engine.end_weight_update.remote() for engine in rollout_engines])


def _check_weight_sync_results(results: list, *, is_lora: bool) -> None:
    """Validate return values from rollout engine weight-sync RPCs.

    Raises RuntimeError if any engine reports failure, preventing silent
    failures when SGLang versions are incompatible.
    """
    sync_type = "LoRA" if is_lora else "Base model"
    for result in results:
        if isinstance(result, Mapping):
            success = result.get("success")
            error_msg = result.get("error_message") or result.get("error") or "unknown error"
        elif hasattr(result, "success"):
            success = result.success
            error_msg = getattr(result, "error_message", "unknown error")
        else:
            continue

        if success is False:
            raise RuntimeError(
                f"{sync_type} weight sync failed on rollout engine: {error_msg}. "
                f"Check SGLang version compatibility."
            )
