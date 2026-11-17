"""LoRA utilities for Megatron backend using Megatron-Bridge PEFT integration."""

import logging
import re
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from miles.backends.megatron_utils.lora.native.modules.linear import NativeLoRAAdapter
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.lora.utils import (  # noqa: F401  (re-exported)
    build_lora_config,
    is_lora_enabled,
    lora_rollout_enabled,
)

if TYPE_CHECKING:
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

logger = logging.getLogger(__name__)

# a Megatron dist-checkpoint's optimizer keys; a ChainedOptimizer may prefix each child's keys with chained_<i>.
OPTIMIZER_KEY = re.compile(r"(chained_\d+\.)?optimizer\.")

_marked_lora_grad_params_cache: dict[int, list] = {}


def reduce_marked_lora_grads(model: Sequence[torch.nn.Module]) -> None:
    """Sum partial grads of replicated LoRA params over their tagged group ("tp"|"ep"), before the DP reduce-scatter."""
    from megatron.core import parallel_state as ps

    key = id(model[0]) if model else 0
    marked = _marked_lora_grad_params_cache.get(key)
    if marked is None:
        marked = []
        for chunk in model:
            for param in chunk.parameters():
                group_name = getattr(
                    param, "_lora_grad_sum_group", None
                )  # config-access-exempt: _lora_grad_sum_group is optional backend-attached tensor metadata
                if group_name is not None and param.requires_grad:
                    marked.append((param, group_name))
        _marked_lora_grad_params_cache[key] = marked
    if not marked:
        return
    groups = {
        "tp": (ps.get_tensor_model_parallel_group(), ps.get_tensor_model_parallel_world_size()),
        "ep": (ps.get_expert_model_parallel_group(), ps.get_expert_model_parallel_world_size()),
    }
    for group_name in ("tp", "ep"):
        group, size = groups[group_name]
        if size <= 1:
            continue
        grads = []
        for param, g_name in marked:
            if g_name != group_name:
                continue
            grad = getattr(
                param, "main_grad", None
            )  # config-access-exempt: main_grad is optional backend-attached tensor metadata
            if grad is None:
                grad = param.grad
            if grad is not None:
                grads.append(grad)
        # set iteration order follows address-derived hashes and need not agree across ranks
        for dt in sorted({g.dtype for g in grads}, key=str):
            gs = [g for g in grads if g.dtype == dt]
            if len(gs) == 1:
                dist.all_reduce(gs[0], op=dist.ReduceOp.SUM, group=group)
                continue
            flat = torch._utils._flatten_dense_tensors(gs)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
            for g, red in zip(gs, torch._utils._unflatten_dense_tensors(flat, gs), strict=False):
                g.copy_(red)


def is_lora_model(model: Sequence[torch.nn.Module]) -> bool:
    """Whether the model was built with LoRA; set at build so every rank agrees, including a pipeline
    stage the target selection left without adapters."""
    return model[0].lora_attached


def adapter_named_parameters(model: Sequence[torch.nn.Module]) -> list[dict[str, torch.nn.Parameter]]:
    """Every adapter parameter of each model chunk, found by the module that owns it: a native adapter,
    or what a Bridge ``AdapterWrapper`` holds besides the base module it wraps.

    Virtual pipeline chunks reuse local parameter names, so the parameters stay keyed per chunk.
    """
    try:
        from megatron.bridge.peft.adapter_wrapper import AdapterWrapper
    except ImportError:
        AdapterWrapper = None

    adapter_param_ids: set[int] = set()
    for model_chunk in model:
        for module in model_chunk.modules():
            if isinstance(module, NativeLoRAAdapter):
                adapter_param_ids.update(id(param) for param in module.parameters())
            elif AdapterWrapper is not None and isinstance(module, AdapterWrapper):
                base_param_ids = {id(param) for param in module.to_wrap.parameters()}
                adapter_param_ids.update(id(param) for param in module.parameters() if id(param) not in base_param_ids)
    return [
        {name: param for name, param in model_chunk.named_parameters() if id(param) in adapter_param_ids}
        for model_chunk in model
    ]


def _adapter_shard_params(model: Sequence[torch.nn.Module]) -> dict[str, torch.nn.Parameter]:
    """A rank's adapter parameters under their native-shard names."""
    params_by_chunk = adapter_named_parameters(model)
    if len(params_by_chunk) == 1:
        return params_by_chunk[0]
    # virtual pipeline chunks are prefixed model{i}, as Megatron's own checkpoints key them
    return {
        f"model{chunk_index}.{name}": param
        for chunk_index, params in enumerate(params_by_chunk)
        for name, param in params.items()
    }


_param_grad_buffer_patched = False


def patch_param_grad_buffer_for_colocate_mode_lora() -> None:
    """Patch _ParamAndGradBuffer to use disable_param_buffers_cpu_backup=True.

    In colocate mode with offload_train, torch_memory_saver.pause(tag="default")
    offloads default-region GPU memory.  During LoRA training, base weights are
    frozen (requires_grad=False) so DDP only creates buffers for adapter params.

    This patch ensures those buffers are allocated in the "param_buffer" region
    (enable_cpu_backup=False), making them invisible to pause(tag="default") —
    eliminating the need for resume()/pause() around update_weights.

    The patch is idempotent and only takes effect once.
    """
    global _param_grad_buffer_patched
    if _param_grad_buffer_patched:
        return
    _param_grad_buffer_patched = True

    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer

    _original_init = _ParamAndGradBuffer.__init__

    def _patched_init(self, *args, **kwargs):
        # Megatron reads these flags from ddp_config (its first ctor argument).
        ddp_config = kwargs.get("ddp_config", args[0] if args else None)
        ddp_config.disable_param_buffers_cpu_backup = True
        ddp_config.disable_grad_buffers_cpu_backup = True
        _original_init(self, *args, **kwargs)

    _ParamAndGradBuffer.__init__ = _patched_init
    logger.info("Patched _ParamAndGradBuffer.__init__ for LoRA colocate mode (disable cpu backup)")


# ---------------------------------------------------------------------------
# Model setup helpers (used by model.py)
# ---------------------------------------------------------------------------


def create_lora_instance(args: Namespace, *, target_modules):
    """Create a LoRA adapter with resolved Megatron target modules."""
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA
    from megatron.bridge.peft.lora import LoRA

    lora_cls = CanonicalLoRA if args.lora_type == "canonical_lora" else LoRA
    lora = lora_cls(
        target_modules=target_modules,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        lora_A_init_method=args.lora_A_init_method,
        lora_B_init_method=args.lora_B_init_method,
        # SGLang serves one adapter per expert, or shared-outer (PR #21466)
        share_expert_adapters=False,
        experts_shared_outer_loras=args.experts_shared_outer_loras,
    )

    logger.info(
        f"Created {lora_cls.__name__}: rank={args.lora_rank}, alpha={args.lora_alpha}, "
        f"dropout={args.lora_dropout}, target_modules={target_modules}"
    )
    return lora


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------


def _native_adapter_shard(adapter_dir: Path) -> Path | None:
    """This rank's native adapter shard, or None when the directory holds none at all."""
    global_rank = dist.get_rank() if dist.is_initialized() else 0
    shard = adapter_dir / f"adapter_megatron_rank{global_rank}.pt"
    if shard.exists():
        return shard
    assert not any(adapter_dir.glob("adapter_megatron_rank*.pt")), (
        f"{adapter_dir} holds per-rank adapter shards but none for global rank {global_rank}; "
        "it was saved under a different parallel layout."
    )
    legacy = adapter_dir / f"adapter_megatron_tp{get_parallel_state().tp.rank}_pp{get_parallel_state().pp.rank}.pt"
    if legacy.exists():
        logger.warning(f"Using legacy tp/pp-named adapter shard {legacy}; only valid when EP<=TP")
        return legacy
    return None


def _read_native_adapter_shard(
    model: Sequence[torch.nn.Module], shard: Path
) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
    """Each adapter parameter with its tensor from ``shard``, checked against the model; writes nothing."""
    state_dict = torch.load(shard, map_location="cpu", weights_only=True)
    adapter_params = _adapter_shard_params(model)
    missing = adapter_params.keys() - state_dict.keys()
    unexpected = state_dict.keys() - adapter_params.keys()
    if missing or unexpected:
        raise RuntimeError(
            f"Adapter checkpoint {shard} does not match the model's adapter parameters: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    mismatched = [
        f"{name}: saved {tuple(state_dict[name].shape)} != parameter {tuple(param.shape)}"
        for name, param in adapter_params.items()
        if state_dict[name].shape != param.shape
    ]
    # copy_ would broadcast a [1, ...] tensor into an [E, ...] parameter
    if mismatched:
        raise RuntimeError(f"Adapter checkpoint {shard} does not match the model's adapter shapes: {mismatched}")
    return [(param, state_dict[name]) for name, param in adapter_params.items()]


def _load_training_state(adapter_dir: Path) -> dict | None:
    rank = dist.get_rank() if dist.is_initialized() else 0
    state_path = adapter_dir / f"training_state_rank{rank}.pt"
    if not state_path.exists():
        return None
    # Optimizer state dicts hold non-tensor objects (step counts, param group metadata).
    return torch.load(state_path, map_location="cpu", weights_only=False)


def _distributed_optimizers(optimizer) -> list:
    # Megatron is imported on use: this module is also imported where Megatron is not
    from megatron.core.optimizer import DistributedOptimizer

    subs = optimizer.chained_optimizers
    # another optimizer's state would be silently left out of the checkpoint
    assert all(
        isinstance(sub, DistributedOptimizer) for sub in subs
    ), f"the adapter checkpoint needs --use-distributed-optimizer; got {[type(sub).__name__ for sub in subs]}"
    return subs


def _zero_init_empty_hdo_states(subs) -> None:
    """Gives every DistributedOptimizer sub whose HDO has no state yet the state of a never-stepped HDO.

    Megatron's load_state_dict would otherwise run dummy_step on it: a real Adam step on random gradients.
    """
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

    for sub in subs:
        if isinstance(sub.optimizer, HybridDeviceOptimizer) and not sub.optimizer.state:
            _init_zero_hdo_state(sub.optimizer)


def _init_zero_hdo_state(hdo: "HybridDeviceOptimizer") -> None:
    """The state a never-stepped HDO is equivalent to: its fp32 master, zero moments, step 0.

    Megatron's own dummy_step would take a real Adam step on random gradients.
    """
    for group in hdo.param_groups:
        for param in group["params"]:
            master = hdo.param_to_inner_param[param]
            hdo.state[param] = {
                "step": torch.tensor(0.0),
                "exp_avg": torch.zeros_like(master),
                "exp_avg_sq": torch.zeros_like(master),
                "master_param": master,
            }
