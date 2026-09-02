import logging
import math
from argparse import Namespace
from collections.abc import Sequence
from contextlib import contextmanager

import torch
import torch.distributed as dist

from miles.backends.megatron_utils.api_backends.multi_lora.checkpoint import (
    _slot_children,
    named_adapter_slot_parameters,
)
from miles.backends.training_utils.operation_execution import resolve_adam_params

logger = logging.getLogger(__name__)


def adapter_slot_parameters(model, slot: int) -> list[torch.nn.Parameter]:
    """All parameters belonging to one adapter slot, across model chunks."""
    return [param for _, param in named_adapter_slot_parameters(model, slot)]


def _adam_init_state_fn(opt, config=None):
    for group in opt.param_groups:
        for p in group["params"]:
            if len(opt.state[p]) == 0:
                opt.state[p]["exp_avg"] = torch.zeros_like(p.data)
                opt.state[p]["exp_avg_sq"] = torch.zeros_like(p.data)


@contextmanager
def _only_slot_trainable(model_chunks, slot_params: list[torch.nn.Parameter]):
    slot_ids = {id(p) for p in slot_params}
    frozen = []
    for model_chunk in model_chunks:
        for param in model_chunk.parameters():
            if param.requires_grad and id(param) not in slot_ids:
                param.requires_grad = False
                frozen.append(param)
    try:
        yield
    finally:
        for param in frozen:
            param.requires_grad = True


def build_multi_lora_operation_optimizer(args: Namespace, config, model_chunks: Sequence):
    assert (
        not config.use_distributed_optimizer
    ), "per-slot optimizers require use_distributed_optimizer=False (LayerWise shards; grad retention all-reduces)"
    assert not config.fp16, "tinker per-slot optimizers require bf16 (no dynamic loss scaler)"
    assert (
        config.optimizer or ""
    ).lower() == "adam", (
        f"tinker per-slot optimizers only implement Adam semantics; got optimizer={config.optimizer!r}"
    )

    from megatron.core.optimizer import get_megatron_optimizer
    from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
    from megatron.core.process_groups_config import ProcessGroupCollection

    pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    # Defer bf16 master-weight creation into LayerWise (post-sharding) so fp32 masters exist only for owned params.
    reset_bf16 = config.bf16
    config.bf16 = False

    base_optimizers: list = []
    init_fns: list = []
    slot_child_indices: dict[int, list[int]] = {}
    try:
        for slot in range(args.multi_lora_n_adapters):
            slot_params = adapter_slot_parameters(model_chunks, slot)
            assert slot_params, f"adapter slot {slot} has no parameters; is this a multi-LoRA model?"
            with _only_slot_trainable(model_chunks, slot_params):
                chained = get_megatron_optimizer(
                    config,
                    list(model_chunks),
                    use_gloo_process_groups=args.use_gloo_process_groups,
                )
            children = [
                child
                for child in chained.chained_optimizers
                if getattr(child, "optimizer", None) is not None and child.get_parameters()
            ]
            assert children, f"adapter slot {slot} produced no optimizer children"
            slot_child_indices[slot] = list(range(len(base_optimizers), len(base_optimizers) + len(children)))
            for child in children:
                for group in child.param_groups:
                    group["miles_multi_lora_slot"] = slot
                # LayerWise wraps raw torch optimizers itself; the pinned MCore
                # rejects pre-wrapped children (slot tags survive via the proxy).
                base_optimizers.append(child.optimizer)
                init_fns.append(_adam_init_state_fn)
    finally:
        config.bf16 = reset_bf16

    optimizer = LayerWiseDistributedOptimizer(base_optimizers, config, pg_collection, init_state_fn_list=init_fns)

    # Dense and expert params use independent ownership groups; norm/clip reductions must span the world.
    for child in optimizer.chained_optimizers:
        child.grad_stats_parallel_group = None

    optimizer.miles_slot_child_indices = slot_child_indices
    logger.info(
        f"[tinker] built LayerWise optimizer: {args.multi_lora_n_adapters} slots, "
        f"{len(optimizer.chained_optimizers)} chained children"
    )
    return optimizer


def reload_adapter_slot_model_params(optimizer, slot: int) -> None:
    for child in _slot_children(optimizer, slot):
        child.reload_model_params()


def zero_adapter_slot_grads(model, slot: int) -> None:
    for param in adapter_slot_parameters(model, slot):
        if (main_grad := getattr(param, "main_grad", None)) is not None:
            main_grad.zero_()
        param.grad = None
        if (main_param := getattr(param, "main_param", None)) is not None:
            main_param.grad = None


def _found_inf_anywhere(found_inf: bool) -> bool:
    """The veto must agree on every rank, or the collective step order diverges."""
    if not dist.is_initialized():
        return found_inf
    flag = torch.tensor([1.0 if found_inf else 0.0], device=torch.cuda.current_device())
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return flag.item() > 0


def _norm_source_flags_anywhere(has_norm_source: bool, has_grads: bool) -> tuple[bool, bool]:
    if not dist.is_initialized():
        return has_norm_source, has_grads
    flags = torch.tensor(
        [1.0 if has_norm_source else 0.0, 1.0 if has_grads else 0.0], device=torch.cuda.current_device()
    )
    dist.all_reduce(flags, op=dist.ReduceOp.MAX)
    return bool(flags[0].item() > 0), bool(flags[1].item() > 0)


def apply_adam_params_to_slot(optimizer, slot: int, adam_params: dict | None) -> dict:
    resolved = resolve_adam_params(adam_params)
    for child in _slot_children(optimizer, slot):
        for group in child.param_groups:
            group["lr"] = resolved["learning_rate"]
            group["betas"] = (resolved["beta1"], resolved["beta2"])
            group["eps"] = resolved["eps"]
            group["weight_decay"] = resolved["weight_decay"]
    return resolved


def step_adapter_slots(
    optimizer,
    model,
    adam_params_by_slot: dict[int, dict | None],
) -> tuple[dict[int, float], set[int], set[int]]:
    from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32, get_grad_norm_fp32

    grad_norms: dict[int, float] = {}
    vetoed: set[int] = set()
    norm_blind: set[int] = set()

    for slot in sorted(adam_params_by_slot):
        children = _slot_children(optimizer, slot)
        adam = apply_adam_params_to_slot(optimizer, slot, adam_params_by_slot[slot])

        # Copy accumulated main_grads into the owned masters' grads, untouched.
        found_inf = False
        for child in children:
            found_inf = bool(child.prepare_grads()) or found_inf

        # Reduce per-slot norms across the world to combine dense and expert ownership groups.
        grads_for_norm = []
        slot_params = []
        for child in children:
            grads_for_norm += child.get_main_grads_for_grad_norm()
            slot_params += child.get_parameters()
        slot_norm = get_grad_norm_fp32(grads_for_norm, grad_stats_parallel_group=None)

        # A non-finite step would otherwise be applied AND live-published to
        # every engine; the veto must be unanimous across ranks.
        if _found_inf_anywhere(found_inf) or not math.isfinite(float(slot_norm)):
            logger.error(
                f"[tinker] slot {slot}: non-finite gradients "
                f"(found_inf={found_inf}, grad_norm={float(slot_norm)}); step vetoed, grads cleared"
            )
            vetoed.add(slot)
            zero_adapter_slot_grads(model, slot)
            continue

        has_norm_source, has_grads = _norm_source_flags_anywhere(
            bool(grads_for_norm),
            any(param.grad is not None and bool((param.grad != 0).any().item()) for param in slot_params),
        )
        if has_grads and not has_norm_source:
            logger.error(f"[tinker] slot {slot}: no grad-norm source despite grads (mis-flagged params); step refused")
            norm_blind.add(slot)
            zero_adapter_slot_grads(model, slot)
            continue

        if adam["grad_clip_norm"] > 0.0 and slot_params:
            clip_grad_by_total_norm_fp32(slot_params, adam["grad_clip_norm"], slot_norm, False)
        grad_norms[slot] = float(slot_norm)

        for child in children:
            child.step_with_ready_grads()

        zero_adapter_slot_grads(model, slot)

    if grad_norms:
        optimizer.allgather_params()

    return grad_norms, vetoed, norm_blind
