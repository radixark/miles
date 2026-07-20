"""Per-slot decoupled optimizers for multi-LoRA training.

One base Adam per adapter slot, assembled under Megatron's
``LayerWiseDistributedOptimizer`` (whole-parameter ZeRO-1):

- each slot is an independent chained child with its own Adam state, step
  count, and gradient clipping;
- optimizer state is sharded across DP ranks at whole-parameter granularity;
- gradients flow through plain DDP all-reduce (``use_distributed_optimizer``
  must be OFF), which is also what makes cross-batch gradient retention
  idempotent: the retained, already-reduced portion of the buffer is identical
  on every rank, so re-reducing it is a no-op.

Selective stepping (``step_adapter_slots``) steps only the slots whose
adapter batch completed, zeroes only their gradients, and leaves every
other slot's accumulated gradients and optimizer state untouched.
"""

import logging
from argparse import Namespace
from collections.abc import Sequence
from contextlib import contextmanager

import torch
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32, get_grad_norm_fp32
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection

logger = logging.getLogger(__name__)


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
    """Build one Adam per adapter slot under a LayerWiseDistributedOptimizer.

    The returned optimizer is a ``ChainedOptimizer`` whose ``chained_optimizers``
    hold one Float16-wrapped Adam per slot, in slot order; each child's param
    groups are tagged with ``miles_multi_lora_slot``. Param groups are narrowed
    to this rank's whole-parameter shard by LayerWise, so Adam state exists only
    for owned params.
    """
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

    # Delay master-weight creation into LayerWise (post-sharding), so fp32
    # masters exist only for owned params. LayerWise unwraps FP32Optimizer
    # children and re-wraps with Float16OptimizerWithFloat16Params itself.
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
                    use_gloo_process_groups=args.enable_gloo_process_groups,
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

    # LayerWise aggregates grad stats globally (params are scattered across DP
    # ranks at whole-parameter granularity), so per-child norm/clip reductions
    # must span the world too, not just the model-parallel group.
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
    """Between-batch replacement for ``DistributedDataParallel.zero_grad_buffer``.

    Resets the per-iteration bookkeeping (``grad_added_to_main_grad`` flags and
    bucket-group sync metadata) WITHOUT zeroing the grad buffers, so per-adapter
    gradient accumulation survives across train batches. Slot gradients are
    zeroed selectively at step time instead.
    """
    for model_chunk in model_chunks:
        if getattr(model_chunk.config, "cuda_graph_impl", "none") != "transformer_engine":
            for param in model_chunk.params_with_grad:
                param.grad_added_to_main_grad = False
        for bucket_group in model_chunk.bucket_groups + model_chunk.expert_parallel_bucket_groups:
            bucket_group.reset()


def zero_adapter_slot_grads(model, slot: int) -> None:
    """Zero one slot's gradients everywhere they live: the DDP ``main_grad``
    buffer views (every rank holds the full buffer under plain DDP) and any
    lingering ``grad``/``main_param.grad`` references."""
    for param in adapter_slot_parameters(model, slot):
        if (main_grad := getattr(param, "main_grad", None)) is not None:
            main_grad.zero_()
        param.grad = None
        if (main_param := getattr(param, "main_param", None)) is not None:
            main_param.grad = None


def step_adapter_slots(
    optimizer,
    model,
    step_batch_sizes: dict[int, int],
    clip_grad: float,
) -> dict[int, float]:
    """Step exactly the given slots; retain everyone else's gradients.

    ``step_batch_sizes`` maps slot -> adapter_global_batch_size. Samples enter
    the buffers with weight 1, so the accumulated gradient is a sum over the
    adapter batch; scaling the fresh master-grad copy by 1/batch_size at step
    time yields the adapter-batch mean (the constant is known in advance, and
    the copy is remade by ``prepare_grads`` each step, so the scale applies
    exactly once).

    Per slot: copy accumulated ``main_grad`` into the owned masters' ``grad``,
    scale by 1/batch_size, clip per slot, run Adam on the owned shard, copy
    masters back to model params, and zero the slot's gradient state. One param
    all-gather at the end propagates updated weights (unchanged slots gather
    identical bytes).

    Returns the grad norm per stepped slot. Non-finite gradients get the same
    treatment as baseline bf16 training: they poison that adapter's weights
    (visible as a NaN grad norm in the logs) — co-tenants are unaffected since
    slots' params, gradients, and clipping are disjoint.
    """
    grad_norms: dict[int, float] = {}

    for slot, batch_size in step_batch_sizes.items():
        children = _slot_children(optimizer, slot)
        # Copy model main_grads -> owned masters' grads (bf16 has no loss
        # scaler, so prepare_grads performs no found-inf handling here), then
        # normalize the accumulated sum into the adapter-batch mean.
        for child in children:
            child.prepare_grads()
            for main_param in child.get_parameters():
                if main_param.grad is not None:
                    main_param.grad.mul_(1.0 / batch_size)

        # Per-slot norm over the union of the slot's children, reduced across
        # the world (params are scattered whole across DP ranks; every param is
        # counted exactly once on its owner rank).
        grads_for_norm = []
        slot_params = []
        for child in children:
            grads_for_norm += child.get_main_grads_for_grad_norm()
            slot_params += child.get_parameters()
        slot_norm = get_grad_norm_fp32(grads_for_norm, grad_stats_parallel_group=None)
        if clip_grad > 0.0 and slot_params:
            clip_grad_by_total_norm_fp32(slot_params, clip_grad, slot_norm, False)
        grad_norms[slot] = float(slot_norm)

        for child in children:
            child.step_with_ready_grads()

        zero_adapter_slot_grads(model, slot)

    if step_batch_sizes:
        optimizer.allgather_params()

    return grad_norms
