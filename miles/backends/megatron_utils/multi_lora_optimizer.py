"""Per-slot decoupled optimizers (Adam/AdamW or Muon) for multi-LoRA, chained under Megatron's
LayerWiseDistributedOptimizer; requires plain DDP all-reduce (use_distributed_optimizer OFF) so
cross-batch gradient retention stays idempotent."""

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
    """Build one Float16-wrapped optimizer (Adam/AdamW or Muon) per adapter slot under a
    LayerWiseDistributedOptimizer (ChainedOptimizer); each child's param groups are tagged with
    ``miles_multi_lora_slot`` and narrowed to this rank's shard."""
    assert not config.use_distributed_optimizer, (
        "multi-LoRA per-slot optimizers require use_distributed_optimizer=False: "
        "gradient retention relies on all-reduce idempotency, and LayerWise "
        "sharding replaces byte-level ZeRO"
    )
    assert not config.fp16, "multi-LoRA per-slot optimizers require bf16 (no dynamic loss scaler)"
    assert (config.optimizer or "").lower() in ("adam", "muon", "dist_muon"), (
        "multi-LoRA per-slot optimizers implement Adam/AdamW and Muon state layouts "
        f"(state init, slot retirement cleanup); got optimizer={config.optimizer!r}"
    )

    pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    # Defer bf16 master-weight creation into LayerWise (post-sharding) so fp32 masters exist only
    # for owned params. Children must also come out raw (not layer-wise wrapped): multi-LoRA applies
    # LayerWise ONCE over all slots below, so dist_muon collapses to plain muon here.
    reset_bf16 = config.bf16
    reset_optimizer = config.optimizer
    reset_layer_wise = getattr(config, "use_layer_wise_distributed_optimizer", False)
    config.bf16 = False
    if (config.optimizer or "").lower() == "dist_muon":
        config.optimizer = "muon"
    if hasattr(config, "use_layer_wise_distributed_optimizer"):
        config.use_layer_wise_distributed_optimizer = False

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
                # Each child carries the stock init fn matching its own family
                # (Adam moments, or Muon's _init_group momentum buffers).
                init_fns.append(child.init_state_fn)
    finally:
        config.bf16 = reset_bf16
        config.optimizer = reset_optimizer
        if hasattr(config, "use_layer_wise_distributed_optimizer"):
            config.use_layer_wise_distributed_optimizer = reset_layer_wise

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


def step_adapter_slots(
    optimizer,
    model,
    step_batch_sizes: dict[int, int],
    clip_grad: float,
) -> dict[int, float]:
    """Step exactly the slots in ``step_batch_sizes`` (slot -> batch size), retaining all other slots' gradients;
    scales each slot's accumulated grad sum by 1/batch_size and returns the grad norm per stepped slot."""
    grad_norms: dict[int, float] = {}

    for slot, batch_size in step_batch_sizes.items():
        children = _slot_children(optimizer, slot)
        # Copy accumulated main_grads into the owned masters' grads, then scale the sum to the adapter-batch mean.
        for child in children:
            child.prepare_grads()
            for main_param in child.get_parameters():
                if main_param.grad is not None:
                    main_param.grad.mul_(1.0 / batch_size)

        # Per-slot grad norm over the slot's children, reduced across the whole world (whole-param DP scatter).
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
