"""LoRA utilities for Megatron backend using Megatron-Bridge PEFT integration."""

import json
import logging
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from miles.backends.training_utils.checkpoint_io import write_checkpoint_dir
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.weight_update.snapshot_publisher import SnapshotPublisher
from miles.utils.lora.utils import (  # noqa: F401  (re-exported)
    build_lora_config,
    is_lora_enabled,
    lora_rollout_enabled,
)

logger = logging.getLogger(__name__)

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
                group_name = getattr(param, "_lora_grad_sum_group", None)
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
            grad = getattr(param, "main_grad", None)
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
    """Check if model has LoRA layers applied."""
    for model_chunk in model:
        if hasattr(model_chunk.module, "peft_config"):
            return True
        for name, _ in model_chunk.named_parameters():
            if "lora_" in name or "adapter" in name:
                return True
    return False


def _is_adapter_param_name(name: str) -> bool:
    """Check if a parameter name belongs to a LoRA adapter (Megatron internal naming)."""
    return "lora_" in name or (".adapter." in name and ("linear_in" in name or "linear_out" in name))


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

    lora_type_name = getattr(args, "lora_type", "lora").lower()

    if lora_type_name == "canonical_lora":
        lora_cls = CanonicalLoRA
    else:
        lora_cls = LoRA

    lora_kwargs = dict(
        target_modules=target_modules,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        lora_A_init_method=getattr(args, "lora_A_init_method", "xavier"),
        lora_B_init_method=getattr(args, "lora_B_init_method", "zero"),
    )
    if "share_expert_adapters" in getattr(lora_cls, "__dataclass_fields__", {}):
        lora_kwargs["share_expert_adapters"] = False
    # shared-outer grouped-expert LoRA (SGLang PR #21466); per-expert is the default
    if getattr(args, "experts_shared_outer_loras", False):
        assert lora_cls is LoRA, "--experts-shared-outer-loras requires the standard LoRA adapter type"
        lora_kwargs["experts_shared_outer_loras"] = True

    lora = lora_cls(**lora_kwargs)

    logger.info(
        f"Created {lora_cls.__name__}: rank={args.lora_rank}, alpha={args.lora_alpha}, "
        f"dropout={args.lora_dropout}, target_modules={target_modules}"
    )
    return lora


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------


def save_lora_checkpoint(
    model: Sequence[torch.nn.Module],
    args: Namespace,
    save_dir: str,
    *,
    publisher: SnapshotPublisher | None,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
    iteration: int | None = None,
) -> str:
    """Collectively save native adapter shards, training state, and optional HF adapter weights."""
    global_rank = dist.get_rank() if dist.is_initialized() else 0

    def write_shards(checkpoint_dir: Path):
        adapter_state = {
            name: param.detach().cpu()
            for model_chunk in model
            for name, param in model_chunk.named_parameters()
            if _is_adapter_param_name(name)
        }
        training_state = None
        if optimizer is not None:
            save_optimizer = not getattr(args, "no_save_optim", False)
            training_state = {
                "iteration": iteration,
                "optimizer": optimizer.state_dict() if save_optimizer else None,
                "opt_param_scheduler": opt_param_scheduler.state_dict() if opt_param_scheduler else None,
            }

        if args.megatron_to_hf_mode == "raw":
            if global_rank == 0:
                config = build_lora_config(args, target_modules=args.lora_adapter_targets)
                config.update(
                    experts_shared_outer_loras=bool(args.experts_shared_outer_loras),
                    format="megatron_rank_sharded",
                )
                (checkpoint_dir / "adapter_config.json").write_text(json.dumps(config, indent=2))
        else:
            try:
                publisher.write_adapter(None, checkpoint_dir)
            except Exception:
                logger.warning("HF adapter export failed; saving native checkpoint only", exc_info=True)
        torch.save(adapter_state, checkpoint_dir / f"adapter_megatron_rank{global_rank}.pt")
        if training_state is not None:
            torch.save(training_state, checkpoint_dir / f"training_state_rank{global_rank}.pt")

    write_checkpoint_dir(save_dir, write_shards)
    return str(save_dir)


def load_lora_adapter(
    model: Sequence[torch.nn.Module],
    adapter_path: str,
    *,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
    load_optimizer: bool = True,
) -> tuple[bool, int | None, bool]:
    """Restore native adapter shards and optional optimizer/scheduler state.

    HF adapters cannot be loaded into Bridge models through this path.
    """
    adapter_dir = Path(adapter_path).resolve()
    if not adapter_dir.exists():
        logger.warning(f"LoRA adapter path does not exist: {adapter_dir}")
        return False, None, False

    tp_rank = get_parallel_state().tp.rank
    pp_rank = get_parallel_state().pp.rank

    # ---- Try Megatron-native format first (fast, no conversion needed) ----
    global_rank = dist.get_rank() if dist.is_initialized() else 0
    native_path = adapter_dir / f"adapter_megatron_rank{global_rank}.pt"
    if not native_path.exists():
        if any(adapter_dir.glob("adapter_megatron_rank*.pt")):
            raise FileNotFoundError(
                f"{adapter_dir} holds per-rank adapter shards but none for global rank {global_rank}; "
                "it was saved under a different parallel layout."
            )
        legacy = adapter_dir / f"adapter_megatron_tp{tp_rank}_pp{pp_rank}.pt"
        if legacy.exists():
            logger.warning(f"Using legacy tp/pp-named adapter shard {legacy}; only valid when EP<=TP")
            native_path = legacy
    if native_path.exists():
        state_dict = torch.load(native_path, map_location="cpu", weights_only=True)
        adapter_params = {
            name: param
            for model_chunk in model
            for name, param in model_chunk.named_parameters()
            if _is_adapter_param_name(name)
        }
        missing = adapter_params.keys() - state_dict.keys()
        unexpected = state_dict.keys() - adapter_params.keys()
        if missing or unexpected:
            raise RuntimeError(
                f"Adapter checkpoint {native_path} does not match the model's adapter parameters: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        for name, param in adapter_params.items():
            param.data.copy_(state_dict[name].to(device=param.device))
        logger.info(f"Loaded {len(adapter_params)} adapter tensors from Megatron-native checkpoint: {native_path}")

        iteration, optimizer_restored = _load_training_state(
            adapter_dir, optimizer, opt_param_scheduler, load_optimizer
        )
        return True, iteration, optimizer_restored

    if any((adapter_dir / name).exists() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        logger.warning(
            f"Found HF PEFT adapter at {adapter_dir} but direct HF PEFT loading into "
            f"Megatron is not yet supported. Please save using Megatron-native format "
            f"(adapter_megatron_rank*.pt files) for checkpoint resume."
        )
        return False, None, False

    logger.warning(f"No adapter checkpoint found at {adapter_dir}")
    return False, None, False


def _load_training_state(
    adapter_dir: Path,
    optimizer: Any | None,
    opt_param_scheduler: Any | None,
    load_optimizer: bool = True,
) -> tuple[int | None, bool]:
    """Restore optimizer/scheduler state saved alongside a LoRA adapter checkpoint."""
    if optimizer is None:
        return None, False

    rank = dist.get_rank() if dist.is_initialized() else 0
    state_path = adapter_dir / f"training_state_rank{rank}.pt"
    if not state_path.exists():
        return None, False

    # Optimizer state dicts may contain non-tensor objects (e.g. step counts,
    # param group metadata), so full unpickling is required here.
    training_state = torch.load(state_path, map_location="cpu", weights_only=False)

    optimizer_restored = False
    if not load_optimizer:
        logger.info("--no-load-optim: keeping the freshly initialized optimizer")
    elif training_state.get("optimizer") is not None:
        optimizer.load_state_dict(training_state["optimizer"])
        optimizer_restored = True
        logger.info("Restored optimizer state from LoRA checkpoint")

    if opt_param_scheduler is not None and training_state.get("opt_param_scheduler") is not None:
        opt_param_scheduler.load_state_dict(training_state["opt_param_scheduler"])
        logger.info("Restored LR scheduler state from LoRA checkpoint")

    iteration = training_state.get("iteration")
    if iteration is not None:
        logger.info(f"Resuming LoRA training from iteration {iteration}")
    return iteration, optimizer_restored
