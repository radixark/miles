"""Per-slot decoupled Adam optimizers for multi-LoRA, chained under Megatron's LayerWiseDistributedOptimizer;
requires plain DDP all-reduce (use_distributed_optimizer OFF) so cross-batch gradient retention stays idempotent."""

import logging
import math
from argparse import Namespace
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Literal

import torch
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32, get_grad_norm_fp32
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection

logger = logging.getLogger(__name__)

# Step policies: request AdamW params applied to the raw gradient sum (tinker) vs per-slot scheduler with batch mean (E2E).
OptimStepPolicy = Literal["explicit_adam_sum", "scheduled_mean"]
EXPLICIT_ADAM_SUM: OptimStepPolicy = "explicit_adam_sum"
SCHEDULED_MEAN: OptimStepPolicy = "scheduled_mean"


def adapter_slot_parameters(model, slot: int) -> list[torch.nn.Parameter]:
    """All parameters belonging to one adapter slot, across model chunks."""
    from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear

    parameters = []
    seen = set()
    model_chunks = model if isinstance(model, (list, tuple)) else [model]
    for model_chunk in model_chunks:
        for module in model_chunk.modules():
            if not isinstance(module, MultiLoRALinear):
                continue
            for param in module.adapters[slot].parameters():
                if id(param) not in seen:
                    parameters.append(param)
                    seen.add(id(param))
    return parameters


def _adam_init_state_fn(opt, config=None):
    for group in opt.param_groups:
        for p in group["params"]:
            if len(opt.state[p]) == 0:
                opt.state[p]["exp_avg"] = torch.zeros_like(p.data)
                opt.state[p]["exp_avg_sq"] = torch.zeros_like(p.data)


@contextmanager
def _only_slot_trainable(model_chunks, slot_params: list[torch.nn.Parameter]):
    """Temporarily freeze every trainable param outside ``slot_params`` so the
    stock param-group builder sees exactly one slot (the Muon construction
    pattern from megatron's ``get_megatron_muon_optimizer``)."""
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


def build_multi_lora_optimizer(
    args: Namespace,
    config: OptimizerConfig,
    model_chunks: Sequence,
) -> MegatronOptimizer:
    """Build one Float16-wrapped Adam per adapter slot under a LayerWiseDistributedOptimizer (ChainedOptimizer);
    each child's param groups are tagged with ``miles_multi_lora_slot`` and narrowed to this rank's shard."""
    assert not config.use_distributed_optimizer, (
        "multi-LoRA per-slot optimizers require use_distributed_optimizer=False: "
        "gradient retention relies on all-reduce idempotency, and LayerWise "
        "sharding replaces byte-level ZeRO"
    )
    assert not config.fp16, "multi-LoRA per-slot optimizers require bf16 (no dynamic loss scaler)"
    assert (config.optimizer or "").lower() == "adam", (
        "multi-LoRA per-slot optimizers only implement Adam semantics (state init, "
        f"slot retirement cleanup, step clocks); got optimizer={config.optimizer!r}"
    )

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
                base_optimizers.append(child)
                init_fns.append(_adam_init_state_fn)
    finally:
        config.bf16 = reset_bf16

    optimizer = LayerWiseDistributedOptimizer(base_optimizers, config, pg_collection, init_state_fn_list=init_fns)

    # Params are scattered whole across DP ranks, so per-child norm/clip reductions must span the world.
    for child in optimizer.chained_optimizers:
        child.grad_stats_parallel_group = None

    optimizer.miles_slot_child_indices = slot_child_indices
    logger.info(
        f"Built multi-LoRA LayerWise optimizer: {args.multi_lora_n_adapters} slots, "
        f"{len(optimizer.chained_optimizers)} chained children"
    )
    return optimizer


def _slot_children(optimizer, slot: int):
    return [optimizer.chained_optimizers[i] for i in optimizer.miles_slot_child_indices[slot]]


def reset_grad_metadata_keep_grads(model_chunks) -> None:
    """Reset DDP per-iteration grad bookkeeping WITHOUT zeroing grad buffers, so per-adapter accumulation
    survives across train batches (replaces ``DistributedDataParallel.zero_grad_buffer``)."""
    for model_chunk in model_chunks:
        if getattr(model_chunk.config, "cuda_graph_impl", "none") != "transformer_engine":
            for param in model_chunk.params_with_grad:
                param.grad_added_to_main_grad = False
        for bucket_group in model_chunk.bucket_groups + model_chunk.expert_parallel_bucket_groups:
            bucket_group.reset()


def zero_adapter_slot_grads(model, slot: int) -> None:
    """Zero one slot's gradients everywhere they live: the DDP ``main_grad`` buffer views
    and any lingering ``grad``/``main_param.grad`` references."""
    for param in adapter_slot_parameters(model, slot):
        if (main_grad := getattr(param, "main_grad", None)) is not None:
            main_grad.zero_()
        param.grad = None
        if (main_param := getattr(param, "main_param", None)) is not None:
            main_param.grad = None


def apply_adam_params(
    optimizer, slot: int, *, lr: float, beta1: float, beta2: float, eps: float, weight_decay: float
) -> None:
    """Write per-call AdamW hyperparameters into one slot's param groups (tinker slots carry no scheduler)."""
    for child in _slot_children(optimizer, slot):
        for group in child.param_groups:
            group["lr"] = lr
            group["betas"] = (beta1, beta2)
            group["eps"] = eps
            group["weight_decay"] = weight_decay


def step_adapter_slots(
    optimizer,
    model,
    step_scales: dict[int, float],
    clip_grads: dict[int, float],
    check_finite: bool = False,
) -> dict[int, float | None]:
    """Step exactly the slots in ``step_scales`` (slot -> grad multiplier: 1/batch for E2E mean, 1.0 for tinker sum),
    retaining other slots' gradients; returns grad norm per slot, or None when check_finite rejects a non-finite
    norm (that slot's grads are zeroed and it does not step)."""
    grad_norms: dict[int, float | None] = {}
    stepped_any = False

    for slot, scale in step_scales.items():
        children = _slot_children(optimizer, slot)
        # Copy accumulated main_grads into the owned masters' grads, then apply the caller's scale to the sum.
        for child in children:
            child.prepare_grads()
            if scale != 1.0:
                for main_param in child.get_parameters():
                    if main_param.grad is not None:
                        main_param.grad.mul_(scale)

        # Per-slot grad norm over the slot's children, reduced across the whole world (whole-param DP scatter).
        grads_for_norm = []
        slot_params = []
        for child in children:
            grads_for_norm += child.get_main_grads_for_grad_norm()
            slot_params += child.get_parameters()
        slot_norm = get_grad_norm_fp32(grads_for_norm, grad_stats_parallel_group=None)
        # The norm is world-reduced, so this local check is a consistent all-rank vote.
        if check_finite and not math.isfinite(float(slot_norm)):
            zero_adapter_slot_grads(model, slot)
            grad_norms[slot] = None
            continue
        clip_grad = clip_grads.get(slot, 0.0)
        if clip_grad > 0.0 and slot_params:
            clip_grad_by_total_norm_fp32(slot_params, clip_grad, slot_norm, False)
        grad_norms[slot] = float(slot_norm)

        for child in children:
            child.step_with_ready_grads()

        zero_adapter_slot_grads(model, slot)
        stepped_any = True

    if stepped_any:
        optimizer.allgather_params()

    return grad_norms
