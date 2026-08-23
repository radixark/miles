"""Wiring for --rematerialize-param-from-master-weight. Rebuilds the low-precision
weights from the optimizer's master weights instead of a pinned CPU copy."""

import logging
from argparse import Namespace
from collections.abc import Callable, Iterator, Sequence

import torch

from miles.backends.megatron_utils.misc_utils import strip_param_name_prefix
from miles.utils.tensor_backper import MainCastContext

logger = logging.getLogger(__name__)


def _named_restore_extras(model: Sequence[torch.nn.Module]) -> Iterator[tuple[str, torch.Tensor]]:
    """Tensors with no master weight to rebuild from, so they keep a pinned backup."""
    for vp_stage, model_module in enumerate(model):
        for name, buffer in model_module.named_buffers():
            if "expert_bias" in name:
                yield f"vp_stages.{vp_stage}.{strip_param_name_prefix(name)}", buffer
        for name, param in model_module.named_parameters():
            # fp32 params and frozen params (e.g. --moe-router-freeze-gate) are
            # excluded from the DDP param buffers, so they have no master weight.
            if param.dtype == torch.float32 or not param.requires_grad:
                yield f"vp_stages.{vp_stage}.{strip_param_name_prefix(name)}", param


def build_main_cast_context(args: Namespace, *, model: Sequence[torch.nn.Module], optimizer) -> MainCastContext:
    extras = list(_named_restore_extras(model))
    extras_bytes = sum(t.numel() * t.element_size() for _, t in extras)
    logger.info(
        f"rematerialize-param-from-master-weight: {len(extras)} extra tensors "
        f"({extras_bytes / 2**20:.1f} MiB) kept in pinned backup: "
        f"{[name for name, _ in extras[:20]]}"
    )
    return MainCastContext(
        cast_main_to_params=_build_cast_main_to_params_fn(
            optimizer, precision_aware=args.use_precision_aware_optimizer
        ),
        model_chunks=model,
        extras_getter=lambda: _named_restore_extras(model),
        rematerializable_ids=_assert_rematerialize_coverage(model, extras),
        check=args.check_rematerialize_param_from_master_weight,
    )


def _build_cast_main_to_params_fn(optimizer, *, precision_aware: bool) -> Callable[[], None]:
    dist_opts = optimizer.chained_optimizers
    if not precision_aware:

        def cast_mcore():
            for dist_opt in dist_opts:
                dist_opt._copy_main_params_to_model_params()

        return cast_mcore

    # Arg validation narrows precision-aware to --optimizer-cpu-offload, where
    # _copy_main_params_to_model_params is a no-op and HDO holds the masters instead.
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

    inners = [dist_opt.optimizer for dist_opt in dist_opts]
    for inner in inners:
        assert isinstance(inner, HybridDeviceOptimizer), type(inner)

    def cast_hdo():
        for inner in inners:
            _replay_hybrid_device_copy_back(inner)

    return cast_hdo


@torch.no_grad()
def _replay_hybrid_device_copy_back(hdo) -> None:
    """Replay HDO's two step post-hooks, which wrote this shard at step end."""
    for shard_view, cpu_master in hdo.gpu_params_map_cpu_copy.items():
        shard_view.data.copy_(cpu_master.data)
    # The CPU pass above already covered the params present in both maps.
    for shard_view, fp32_master in hdo.param_to_fp32_param.items():
        if shard_view not in hdo.gpu_params_map_cpu_copy:
            shard_view.data.copy_(fp32_master.data)


def _assert_rematerialize_coverage(model: Sequence[torch.nn.Module], extras: list[tuple[str, torch.Tensor]]) -> set:
    """Anything outside the DDP buffers and the extras backup would come back as garbage.

    DDP buffer membership is the right criterion: optimizer structures only cover this
    rank's shard under DP>1.
    """
    restorable = {id(t) for _, t in extras}
    for model_module in model:
        for buffer in model_module.buffers + model_module.expert_parallel_buffers:
            restorable.update(id(p) for p in buffer.params)
    uncovered = []
    for model_module in model:
        for name, param in model_module.named_parameters():
            if id(param) not in restorable:
                uncovered.append(name)
    assert not uncovered, (
        f"--rematerialize-param-from-master-weight cannot restore {len(uncovered)} params "
        f"(not in the DDP param buffers nor in the extras backup): {uncovered[:10]}"
    )
    return restorable
