"""Per-slot decoupled Adam optimizers for multi-LoRA, chained under Megatron's LayerWiseDistributedOptimizer;
requires plain DDP all-reduce (use_distributed_optimizer OFF) so cross-batch gradient retention stays idempotent."""

import logging
import math
from argparse import Namespace
from collections.abc import Sequence
from contextlib import contextmanager

import torch
import torch.distributed as dist
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
                # LayerWise wraps raw torch optimizers itself; the pinned MCore
                # rejects pre-wrapped children (slot tags survive via the proxy).
                base_optimizers.append(child.optimizer)
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


def reload_adapter_slot_model_params(optimizer, slot: int) -> None:
    """Refresh fp32 masters for ONE slot only — a global reload would quantize
    every other resident slot's masters through bf16."""
    for child in _slot_children(optimizer, slot):
        child.reload_model_params()


def reset_grad_metadata_keep_grads(model_chunks) -> None:
    """Reset DDP grad bookkeeping WITHOUT zeroing buffers, so per-adapter
    accumulation survives (replaces ``zero_grad_buffer``)."""
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


def _found_inf_anywhere(found_inf: bool) -> bool:
    """The veto must agree on every rank, or the collective step order diverges."""
    if not dist.is_initialized():
        return found_inf
    flag = torch.tensor([1.0 if found_inf else 0.0], device=torch.cuda.current_device())
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return flag.item() > 0


# Tinker-compatible per-call Adam defaults (thinker adapters).
_ADAM_PARAM_DEFAULTS = dict(learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-12, weight_decay=0.0, grad_clip_norm=0.0)


def apply_adam_params_to_slot(optimizer, slot: int, adam_params: dict) -> dict:
    """Write one operation's AdamParams onto the slot's param groups; returns
    the resolved values. Thinker adapters install no scheduler, so nothing
    overwrites these between operations."""
    resolved = {**_ADAM_PARAM_DEFAULTS, **{k: v for k, v in (adam_params or {}).items() if v is not None}}
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
    step_counts: dict[int, int],
    clip_grad: float,
    *,
    normalize_by_count: bool = True,
    adam_params_by_slot: dict[int, dict] | None = None,
) -> tuple[dict[int, float], set[int]]:
    """Step exactly the slots in ``step_counts`` (slot -> rollout-execution
    count, the loss's aggregation unit), retaining all other slots' gradients.
    Returns (grad norms, vetoed slots): a found-inf/NaN slot is not stepped,
    its grads are cleared, and the caller must not commit or publish it.

    A slot with an ``adam_params_by_slot`` entry follows the client's contract
    instead of the server's: its AdamParams land on the param groups first,
    its gradient sum is never count-normalized (the loss weights own the
    scale), and its clip is the per-call ``grad_clip_norm`` (0.0 = none)."""
    grad_norms: dict[int, float] = {}
    vetoed: set[int] = set()
    adam_params_by_slot = adam_params_by_slot or {}

    for slot in sorted(step_counts):
        children = _slot_children(optimizer, slot)
        adam = adam_params_by_slot.get(slot)
        if adam is not None:
            adam = apply_adam_params_to_slot(optimizer, slot, adam)
        # Copy accumulated main_grads into the owned masters' grads, then scale the sum to the per-execution mean.
        found_inf = False
        for child in children:
            found_inf = bool(child.prepare_grads()) or found_inf
            if normalize_by_count and adam is None:
                for main_param in child.get_parameters():
                    if main_param.grad is not None:
                        main_param.grad.mul_(1.0 / step_counts[slot])

        # Per-slot grad norm over the slot's children, reduced across the whole world (whole-param DP scatter).
        grads_for_norm = []
        slot_params = []
        for child in children:
            grads_for_norm += child.get_main_grads_for_grad_norm()
            slot_params += child.get_parameters()
        slot_norm = get_grad_norm_fp32(grads_for_norm, grad_stats_parallel_group=None)

        # The gate the single-adapter path has always had (`assert update_successful`)
        # and this path silently lacked: a non-finite step would otherwise be
        # applied AND live-published to every engine.
        if _found_inf_anywhere(found_inf) or not math.isfinite(float(slot_norm)):
            logger.error(
                f"[multilora] slot {slot}: non-finite gradients "
                f"(found_inf={found_inf}, grad_norm={float(slot_norm)}); step vetoed, grads cleared"
            )
            vetoed.add(slot)
            zero_adapter_slot_grads(model, slot)
            continue

        slot_clip = adam["grad_clip_norm"] if adam is not None else clip_grad
        if slot_clip > 0.0 and slot_params:
            clip_grad_by_total_norm_fp32(slot_params, slot_clip, slot_norm, False)
        grad_norms[slot] = float(slot_norm)

        for child in children:
            child.step_with_ready_grads()

        zero_adapter_slot_grads(model, slot)

    if grad_norms:
        optimizer.allgather_params()

    return grad_norms, vetoed
